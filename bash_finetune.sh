
# Train the AudioLDM (latent diffusion part)

python3 audioldm_train/train/latent_diffusion.py -c \
 audioldm_train/config/2024_08_21_AudioSR/AudioSR.yaml \
 --reload_from_ckpt data/checkpoints/haoheliu_audiosr_basic.ckpt

# Train the VAE
# python3 audioldm_train/train/autoencoder.py -c audioldm_train/config/2023_11_13_vae_autoencoder/16k_64.yaml