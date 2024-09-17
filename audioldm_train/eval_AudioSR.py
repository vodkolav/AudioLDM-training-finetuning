import os
import yaml
import torch
from audioldm_eval import EvaluationHelper
import json
from pathlib import Path
import pandas as pd

from glob import glob


def list_filtered_subdirectories(directory, pattern):
    return [f.name for f in Path(directory).glob(pattern) if f.is_dir()]


SAMPLE_RATE = 32000
#device = torch.device(f"cuda:{0}")
device = torch.device(f"cpu")
evaluator = EvaluationHelper(SAMPLE_RATE, device)

bp = "/src/AudioLDM-training-finetuning/log/latent_diffusion/2024_08_21_AudioSR/AudioSR/"

# TODO: filter files by date
# dump all to one json
# add step, date and set to metrics

dirs = list_filtered_subdirectories(bp , "val*_09-1*")

#run = "val_48_08-31-17:50_cfg_scale_3.5_ddim_200_n_cand_1/" # beginning of training 

#print(dirs)
print(len(dirs))

pth = "evaluation/" 

os.makedirs(pth, exist_ok=True)

#run = "val_17809_09-01-09:23_cfg_scale_3.5_ddim_200_n_cand_1"  #end of training 
##  val_422621_09-16-02:26_cfg_scale_3.5_ddim_200_n_cand_1

def extract(run):
    tmp = run.split("_")
    res = {"dir": run,
           "set":tmp[0], 
           "step": tmp[1], 
           "date": "2024-"+tmp[2]+":00"  }
    return res


def load_json_to_dataframe(folder_path):
    # Get a list of all JSON files in the folder
    json_files = glob(os.path.join(folder_path, "*.json"))
    
    # Use a list comprehension to load each file and wrap the dict in a list
    data_list = [pd.DataFrame([json.load(open(file))]) for file in json_files]
    
    # Concatenate all DataFrames into a single DataFrame
    df = pd.concat(data_list, ignore_index=True)
    
    return df

runs = []

#AudioLDM-training-finetuning/log/latent_diffusion/2024_08_21_AudioSR/AudioSR/evaluation

#done = pd.concat([pd.read_json(f_name) for f_name in glob(bp + 'evaluation/*.json')])
done = load_json_to_dataframe(bp + 'evaluation')
#done = done.dropna(axis=1)

output_name = "" 

try:
    for run in dirs:
        metad = extract(run)
        if int(metad["step"]) in done.step.values:
            continue

        jfile = glob(bp + run + '/*.json')
        if jfile:
            metrics = json.load(open(jfile[0]))
        else:
            
            generated_files_path = bp + run + "/sr"
            groundtruth_path = bp + run + "/gt"
            same_name = True

            if not os.path.exists(generated_files_path) or not os.path.exists(groundtruth_path):
                continue

            metrics = evaluator.calculate_metrics(generated_files_path, groundtruth_path, same_name ,calculate_lsd=True) # , recalculate=True
            
        metrics.update(metad)
        output_name = bp + 'evaluation/' + run + '.json'
        with open(output_name, 'w') as output:
            json.dump(metrics, output, indent=2)
        #runs.append(metrics)
        print("\rprocessed ", run)

except KeyboardInterrupt as ki:
    print("Keyboard Interrupting evaluation " ,  ki)
except Exception as e:
    print("other error", e)

print("evaluations complete")
    

# to = bp + pth + dirs[-1] + ".json"

# with open( to , 'w') as fl:
#     json.dump(runs, fl, indent=2)






def locate_yaml_file(path):
    for file in os.listdir(path):
        if ".yaml" in file:
            return os.path.join(path, file)
    return None


def is_evaluated(path):
    candidates = []
    for file in os.listdir(
        os.path.dirname(path)
    ):  # all the file inside a experiment folder
        if ".json" in file:
            candidates.append(file)
    folder_name = os.path.basename(path)
    for candidate in candidates:
        if folder_name in candidate:
            return True
    return False


def locate_validation_output(path):
    folders = []
    for file in os.listdir(path):
        dirname = os.path.join(path, file)
        if "val_" in file and os.path.isdir(dirname):
            if not is_evaluated(dirname):
                folders.append(dirname)
    return folders


def evaluate_exp_performance(exp_name):
    abs_path_exp = os.path.join(latent_diffusion_model_log_path, exp_name)
    config_yaml_path = locate_yaml_file(abs_path_exp)

    if config_yaml_path is None:
        print("%s does not contain a yaml configuration file" % exp_name)
        return

    folders_todo = locate_validation_output(abs_path_exp)

    for folder in folders_todo:
        print(folder)

        if len(os.listdir(folder)) == 964:
            test_dataset = "audiocaps"
        elif len(os.listdir(folder)) > 5000:
            test_dataset = "musiccaps"
        else:
            continue

        test_audio_data_folder = os.path.join(test_audio_path, test_dataset)

        evaluator.main(folder, test_audio_data_folder)


def eval(exps):
    for exp in exps:
        try:
            evaluate_exp_performance(exp)
        except Exception as e:
            print(exp, e)


# if __name__ == "__main__":

#     import argparse

#     parser = argparse.ArgumentParser(description="AudioLDM model evaluation")

#     parser.add_argument(
#         "-l", "--log_path", type=str, help="the log path", required=True
#     )
#     parser.add_argument(
#         "-e",
#         "--exp_name",
#         type=str,
#         help="the experiment name",
#         required=False,
#         default=None,
#     )

#     args = parser.parse_args()

#     test_audio_path = "log/testset_data"
#     latent_diffusion_model_log_path = args.log_path

#     if latent_diffusion_model_log_path != "all":
#         exp_name = args.exp_name
#         if exp_name is None:
#             exps = os.listdir(latent_diffusion_model_log_path)
#             eval(exps)
#         else:
#             eval([exp_name])
#     else:
#         todo_list = [os.path.abspath("log/latent_diffusion")]
#         for todo in todo_list:
#             for latent_diffusion_model_log_path in os.listdir(todo):
#                 latent_diffusion_model_log_path = os.path.join(
#                     todo, latent_diffusion_model_log_path
#                 )
#                 if not os.path.isdir(latent_diffusion_model_log_path):
#                     continue
#                 print(latent_diffusion_model_log_path)
#                 exps = os.listdir(latent_diffusion_model_log_path)
#                 eval(exps)
