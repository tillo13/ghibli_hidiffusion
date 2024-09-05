import os
import time
import torch
from datetime import datetime
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, LMSDiscreteScheduler
from hidiffusion import apply_hidiffusion, remove_hidiffusion

# Load the Ghibli Diffusion model
model_id = "nitrosocke/Ghibli-Diffusion"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Disable safety checker
pipe.safety_checker = None

# Define the prompt and the negative prompt
prompt = (
    "ghibli style portrait of a family, with detailed facial features such as eyes, nose, lips, "
    "and hair, in the style of Studio Ghibli. Realistic shading, expressive eyes, clear skin texture."
)
negative_prompt = "blurry, ugly, duplicate, poorly drawn face, deformed, mosaic, artifacts, bad limbs"

# Seed for reproducibility
seed = 3450349066
torch.manual_seed(seed)  # Set the global seed for reproducibility
generator = torch.Generator(device="cuda").manual_seed(seed)  # Create a reproducible generator

# Load the input image
input_image_path = "./451.jpg"
init_image = Image.open(input_image_path).convert("RGB")

# Ensure image is loaded correctly
assert init_image is not None, "Failed to load input image."

# Resize the image to be square and match Diffusion's preferred input size
init_image = init_image.resize((512, 512))

# Use a single scheduler (LMSDiscreteScheduler for this demo)
scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = scheduler

# Fix for the warning message
if hasattr(pipe.scheduler.config, 'lower_order_final'):
    pipe.scheduler.config.lower_order_final = True

# Ensure output directory exists
output_dir = "./hidiffusion_demo_outputs"
os.makedirs(output_dir, exist_ok=True)

# Set strength and guidance scale
strength = 0.6
guidance_scale = 8.0

# Function to generate images with and without HiDiffusion
def generate_images_comparison():
    # Generate image without HiDiffusion
    remove_hidiffusion(pipe)
    start_time = time.time()
    image_without_hidiffusion = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=100,
        negative_prompt=negative_prompt,
        generator=generator
    ).images[0]
    time_taken_without = time.time() - start_time

    # Save the image without HiDiffusion
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_image_name_without = f"demo_without_hidiffusion_{timestamp}.png"
    output_image_path_without = os.path.join(output_dir, output_image_name_without)
    image_without_hidiffusion.save(output_image_path_without)
    print(f"Generated image without HiDiffusion saved as {output_image_path_without} (time taken: {time_taken_without:.2f}s)")

    # Apply HiDiffusion and generate image
    apply_hidiffusion(pipe)
    start_time = time.time()
    image_with_hidiffusion = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=100,
        negative_prompt=negative_prompt,
        generator=generator
    ).images[0]
    time_taken_with = time.time() - start_time

    # Save the image with HiDiffusion
    output_image_name_with = f"demo_with_hidiffusion_{timestamp}.png"
    output_image_path_with = os.path.join(output_dir, output_image_name_with)
    image_with_hidiffusion.save(output_image_path_with)
    print(f"Generated image with HiDiffusion saved as {output_image_path_with} (time taken: {time_taken_with:.2f}s)")

# Run the function to generate images
generate_images_comparison()