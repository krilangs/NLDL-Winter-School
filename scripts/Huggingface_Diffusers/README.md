Denoising diffusion probabilistic model code based upon the Hugging Face's Diffusers library:\
**Diffusers docs**: https://huggingface.co/docs/diffusers/en/index \
**Diffusers GitHub**: https://github.com/huggingface/diffusers

**Code**:\
*sen12ms_dataloader.py*: Loads the SEN12MS dataset (see report).\
*diffusers_training_example.py*: Original code is a tutorial by Hugging Face (Tutorial: https://huggingface.co/docs/diffusers/en/tutorials/basic_training) converted from a Jupyter notebook (Training with 🧨 Diffusers) to Python. Small alterations have been made from the original notebook.

**Results**:\
*ddpm-linearbeta*: Contains the samples made every 10th epoch during training. Trained for 50 epochs with the linear beta-scheduler.\
*ddpm-cosinebeta*: Contains the samples made every 10th epoch during training. Trained for 50 epochs with the cosine beta-scheduler.
