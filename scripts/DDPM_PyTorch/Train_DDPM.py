# -*- coding: utf-8 -*-
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

def main():
    if torch.cuda.is_available():
        device = "cuda"
        print("GPU available: {}".format(torch.cuda.is_available()))
    else:
        device = "cpu"
        print("GPU available: {}".format(torch.cuda.is_available()))
    

    model = Unet(dim=64, dim_mults=(1,2,4)).to(device)
    #print(model)

    diffusion = GaussianDiffusion(model, image_size=128, timesteps=500, 
                                  beta_schedule="cosine").to(device)

    trainer = Trainer(diffusion, "SEN12MS/",
                      train_batch_size=8, train_lr=2e-5,
                      train_num_steps=62920, # Should be 11 epochs
                      gradient_accumulate_every=2, num_samples=25,
                      ema_decay=0.995, amp=True, calculate_fid=False, 
                      save_and_sample_every=5720,
                      num_fid_samples = 5,
                      )

    trainer.train()

if __name__=="__main__":
    main()

    