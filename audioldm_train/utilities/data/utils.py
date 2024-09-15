import contextlib
import importlib
import numpy as np
import torch

from inspect import isfunction
import os
import subprocess
import tempfile
import json
import soundfile as sf
import time
import wave
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from audioldm_train.utilities.audio.lowpass import lowpass

hann_window = {}
mel_basis = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def _locate_cutoff_freq(stft, percentile=0.97):
    def _find_cutoff(x, percentile=0.95):
        percentile = x[-1] * percentile
        for i in range(1, x.shape[0]):
            if x[-i] < percentile:
                return x.shape[0] - i
        return 0

    magnitude = torch.abs(stft)
    energy = torch.cumsum(torch.sum(magnitude, dim=0), dim=0)
    return _find_cutoff(energy, percentile)


def normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5

def pad_wav(waveform, target_length):
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

    if waveform_length == target_length:
        return waveform

    # Pad
    temp_wav = np.zeros((1, target_length), dtype=np.float32)
    rand_start = 0

    temp_wav[:, rand_start : rand_start + waveform_length] = waveform
    return temp_wav


def lowpass_filtering_simulation(dl_output):
    waveform = dl_output["waveform"]  # [1, samples]
    sampling_rate = dl_output["sampling_rate"]
    duration = dl_output["duration"]
    # this is only for inference - to find actual cutoff freq of new data
    # nyq = int(0.5 * sampling_rate)
    # cutoff_freq = (
    #     _locate_cutoff_freq(dl_output["stft"], percentile=0.985) / 1024
    # ) * nyq    
    # # If the audio is almost empty. Give up processing
    # if(cutoff_freq < 1000):
    #     cutoff_freq = nyq - 1000

    # we first perform lowpass filtering to the audio with a cutoff frequency
    # uniformly sampled between 2kHz and 16kHz (for sampling rate 48000)
    #                or between 2kHz and 5.333kHz (for sampling rate 16000)
    cf_max = int(sampling_rate / 3) - 3000
    cutoff_freq = int(np.random.random() * cf_max + 2000)

    # To address the filter generalization problem [3], the type of the lowpass filter
    # is randomly sampled within Chebyshev, Elliptic, Butterworth and Boxcar, 
    ftype = np.random.choice(["butter", "cheby1", "ellip", "bessel"])
    
    # and the order of the lowpass filter is randomly selected between 2 and 10.
    order = np.random.random_integers(2,10)
 
    filtered_audio = lowpass(
        waveform.numpy().squeeze(),
        highcut=cutoff_freq,
        fs=sampling_rate,
        order=order,
        _type=ftype,
    )

    # Add single tone noise
    # Randomly pick a single tone noise frequency between 100 Hz and 15 kHz
    freq = np.random.uniform(100.0, cf_max)

    # Randomly pick an amplitude noise between 0.1 and 1.0
    amplitude = np.random.uniform(0.001, 0.2)

    # Generate the single tone waveform
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    tone_waveform = amplitude * np.sin(2 * np.pi * freq * t)
                
    # Add the single tone to the original waveform
    filtered_audio = filtered_audio + tone_waveform


    filtered_audio = torch.FloatTensor(filtered_audio.copy()).unsqueeze(0)

    if waveform.size(-1) <= filtered_audio.size(-1):
        filtered_audio = filtered_audio[..., : waveform.size(-1)]
    else:
        filtered_audio = torch.functional.pad(
            filtered_audio, (0, waveform.size(-1) - filtered_audio.size(-1))
        )

    return {"waveform_lowpass": filtered_audio}

def read_wav_file(filename):
    # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
    waveform, sr = torchaudio.load(filename)
    duration = waveform.size(-1) / sr

    if(duration > 10.24):
        print("\033[93m {}\033[00m" .format("Warning: audio is longer than 10.24 seconds, may degrade the model performance. It's recommand to truncate your audio to 5.12 seconds before input to AudioSR to get the best performance."))

    if(duration % 5.12 != 0):
        pad_duration = duration + (5.12 - duration % 5.12)
    else:
        pad_duration = duration

    target_frame = int(pad_duration * 100)

    waveform = torchaudio.functional.resample(waveform, sr, 48000)

    waveform = waveform.numpy()[0, ...]

    waveform = normalize_wav(
        waveform
    )  # TODO rescaling the waveform will cause low LSD score

    waveform = waveform[None, ...]
    waveform = pad_wav(waveform, target_length=int(48000 * pad_duration))
    return waveform, target_frame, pad_duration

def read_audio_file(filename):
    waveform, target_frame, duration = read_wav_file(filename)
    log_mel_spec, stft = wav_feature_extraction(waveform, target_frame)
    return log_mel_spec, stft, waveform, duration, target_frame


def mel_spectrogram_train(y, slf):
    global mel_basis, hann_window

    if slf.mel_fmax not in mel_basis:
        mel = librosa_mel_fn(
            sr=slf.sampling_rate, 
            n_fft=slf.filter_length, 
            n_mels=slf.n_mel,
            fmin=slf.mel_fmin, 
            fmax=slf.mel_fmax,
            )
        mel_basis[str(slf.mel_fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(slf.win_length).to(
            y.device
            )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (
            int((slf.filter_length - slf.hop_length) / 2), 
            int((slf.filter_length - slf.hop_length) / 2),
            ),
        mode="reflect",
    )

    y = y.squeeze(1)

    stft_spec = torch.stft(
        y,
        slf.filter_length,
        hop_length=slf.hop_length,
        win_length=slf.win_length,
        window=hann_window[str(y.device)],
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    stft_spec = torch.abs(stft_spec)

    mel = spectral_normalize_torch(
        torch.matmul(mel_basis[str(slf.mel_fmax) +
                               "_" + str(y.device)], stft_spec)
    )

    return mel[0], stft_spec[0]

class Slf():    
    # Slf is temporary surrogate for self until i make a proper class 
    def __init__(self, config) -> None:            
        self.sampling_rate =  config["preprocessing"]["audio"]["sampling_rate"]   #48000
        self.filter_length = config["preprocessing"]["stft"]["filter_length"] #2048
        self.hop_length =  config["preprocessing"]["stft"]["hop_length"] #480
        self.win_length = config["preprocessing"]["stft"]["win_length"] #2048
        self.n_mel = config["preprocessing"]["mel"]["n_mel_channels"] #256
        self.mel_fmin = config["preprocessing"]["mel"]["mel_fmin"] #20
        self.mel_fmax = config["preprocessing"]["mel"]["mel_fmax"] #24000
        self.hopsize = config["preprocessing"]["stft"]["hop_length"]
        self.duration = config["preprocessing"]["audio"]["duration"]
        self.target_length = int(self.duration * self.sampling_rate / self.hopsize)
        

def wav_feature_extraction(waveform, slf):
    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)

    log_mel_spec, stft = mel_spectrogram_train(waveform.unsqueeze(0), slf)

    log_mel_spec = torch.FloatTensor(log_mel_spec.T)
    stft = torch.FloatTensor(stft.T)

    log_mel_spec, stft = pad_spec(log_mel_spec, slf), pad_spec(stft, slf)
    return log_mel_spec, stft

def pad_spec(log_mel_spec, slf):
    n_frames = log_mel_spec.shape[0]
    p = slf.target_length - n_frames
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        log_mel_spec = m(log_mel_spec)
    elif p < 0:
        log_mel_spec = log_mel_spec[0:slf.target_length, :]

    if log_mel_spec.size(-1) % 2 != 0:
        log_mel_spec = log_mel_spec[..., :-1]

    return log_mel_spec

def read_list(fname):
    result = []
    with open(fname, "r", encoding="utf-8") as f:
        for each in f.readlines():
            each = each.strip("\n")
            result.append(each)
    return result


def get_duration(fname):
    with contextlib.closing(wave.open(fname, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)


def get_bit_depth(fname):
    with contextlib.closing(wave.open(fname, "r")) as f:
        bit_depth = f.getsampwidth() * 8
        return bit_depth


def get_time():
    t = time.localtime()
    return time.strftime("%d_%m_%Y_%H_%M_%S", t)


def seed_everything(seed):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def strip_silence(orignal_path, input_path, output_path):
    get_dur = subprocess.run([
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'format=duration',
        '-sexagesimal',
        '-of', 'json',
        orignal_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    duration = json.loads(get_dur.stdout)['format']['duration']
    
    subprocess.run([
        'ffmpeg',
        '-y',
        '-ss', '00:00:00',
        '-i', input_path,
        '-t', duration,
        '-c', 'copy',
        output_path
    ])
    os.remove(input_path)



def save_wave(waveform, inputpath, savepath, name="outwav", samplerate=16000):
    if type(name) is not list:
        name = [name] * waveform.shape[0]

    for i in range(waveform.shape[0]):
        if waveform.shape[0] > 1:
            fname = "%s_%s.wav" % (
                os.path.basename(name[i])
                if (not ".wav" in name[i])
                else os.path.basename(name[i]).split(".")[0],
                i,
            )
        else:
            fname = (
                "%s.wav" % os.path.basename(name[i])
                if (not ".wav" in name[i])
                else os.path.basename(name[i]).split(".")[0]
            )
            # Avoid the file name too long to be saved
            if len(fname) > 255:
                fname = f"{hex(hash(fname))}.wav"

        save_path = os.path.join(savepath, fname)
        temp_path = os.path.join(tempfile.gettempdir(), fname)
        print("\033[98m {}\033[00m" .format("Don't forget to try different seeds by setting --seed <int> so that AudioSR can have optimal performance on your hardware."))
        print("Save audio to %s." % save_path)
        sf.write(temp_path, waveform[i, 0], samplerate=samplerate)
        strip_silence(inputpath, temp_path, save_path)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

