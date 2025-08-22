from diffusers import StableDiffusionPipeline
from peft import PeftModel
import torch
from PIL import Image
import os

def generate_image(prompt):
    base_model_path = "./models/stable-diffusion-v1-5"
    lora_weights = "./models"

    pipe = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float32).to("cpu")
   # Load LoRA weights
    # pipe.load_lora_weights(lora_weights, adapter_name="fashion-lora") # improvised using fashion lora
    # pipe.set_adapters(["fashion-lora"], adapter_weights=[0.7])
    # pipe.fuse_lora()
    
    pipe.enable_attention_slicing()
    result = pipe(prompt, num_inference_steps=10, num_images_per_prompt=5, guidance_scale=7.5)
    images = result.images
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = []                                             

    for i, img in enumerate(images):
        img = img.resize((512, 512), Image.LANCZOS)
        output_paths.append(os.path.join(output_dir, f"img_{i+1}.png"))
        img.save(output_paths[i])
    return output_paths                                    
