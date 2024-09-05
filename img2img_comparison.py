from hidiffusion import apply_hidiffusion, remove_hidiffusion
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from PIL import Image
import torch
import lpips
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time
import warnings
from datetime import datetime  # For date and time

# Function to load images and convert them to numpy arrays
def load_image_as_array(file_path):
    img = Image.open(file_path)
    img = img.convert("RGB")
    return np.array(img)

# Evaluate PSNR and SSIM
def evaluate_psnr_ssim(image1, image2, win_size=7):
    psnr_value = peak_signal_noise_ratio(image1, image2)
    image_shape = image1.shape
    win_size = min(win_size, image_shape[0], image_shape[1])
    ssim_value, _ = structural_similarity(image1, image2, channel_axis=2, win_size=win_size, full=True)
    return psnr_value, ssim_value

# Evaluate LPIPS
def evaluate_lpips(image1, image2):
    loss_fn = lpips.LPIPS(net='alex')
    image1_tensor = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image2_tensor = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        lpips_value = loss_fn(image1_tensor, image2_tensor)

    return lpips_value.item()

# Function to perform img2img transformation
def generate_img2img(me_image_path, use_hidiffusion=True):
    pretrain_model = "stabilityai/stable-diffusion-xl-base-1.0"
    scheduler = DDIMScheduler.from_pretrained(pretrain_model, subfolder="scheduler")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        pretrain_model, scheduler=scheduler, torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")

    # Remove the xformers-related lines
    # pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()

    if use_hidiffusion:
        apply_hidiffusion(pipe)
        base_filename = "me_with_hidiff_img2img.jpg"
    else:
        base_filename = "me_without_hidiff_img2img.jpg"

    datetime_prefix = datetime.now().strftime("%Y%m%d_%H%M%S_")
    filename = datetime_prefix + base_filename

    me_image = Image.open(me_image_path).convert("RGB")
    me_image = me_image.resize((1024, 1024))  # Ensure the image is the correct size

    prompt = "A highly detailed and realistic image of a person in a beautiful fantasy world, with intricate details and lifelike features. Focus on the face and surroundings."
    negative_prompt = "blurry, ugly, poorly drawn face, deformed, artifacts"

    start_time = time.time()
    try:
        # Ensure valid image input and handle necessary conversion for img2img pipeline
        result = pipe(prompt=prompt, image=me_image, strength=0.75, guidance_scale=7.5, num_inference_steps=50, negative_prompt=negative_prompt)
        image = result.images[0]
    except Exception as e:
        print(f"Error during image generation: {e}")
        return None, 0
    
    image.save(filename)
    end_time = time.time()
    duration = end_time - start_time

    return filename, duration

# Main function to run multiple iterations
def main():
    NUMBER_OF_LOOPS = 3
    durations = []
    metrics = []
    me_image_path = 'me.png'

    total_start_time = time.time()

    for i in range(NUMBER_OF_LOOPS):
        print(f"--- Iteration {i+1} ---")
        image1_file, duration1 = generate_img2img(me_image_path, use_hidiffusion=False)
        if image1_file is None:
            print("Skipping iteration due to error in generating image without HiDiffusion")
            continue
        image2_file, duration2 = generate_img2img(me_image_path, use_hidiffusion=True)
        if image2_file is None:
            print("Skipping iteration due to error in generating image with HiDiffusion")
            continue

        image1_array = load_image_as_array(image1_file)
        image2_array = load_image_as_array(image2_file)
        assert image1_array.shape == image2_array.shape and image1_array.size > 0 and image2_array.size > 0, "Images are not loaded correctly or are empty."

        durations.append((image1_file, duration1))
        durations.append((image2_file, duration2))

        psnr_value, ssim_value = evaluate_psnr_ssim(image1_array, image2_array)
        lpips_value = evaluate_lpips(image1_array, image2_array)

        metrics.append({
            'iteration': i + 1,
            'image1_file': image1_file,
            'image1_duration': duration1,
            'image2_file': image2_file,
            'image2_duration': duration2,
            'psnr': psnr_value,
            'ssim': ssim_value,
            'lpips': lpips_value
        })

        print(f"Image {image1_file} generation took {duration1:.2f} seconds.")
        print(f"Image {image2_file} generation took {duration2:.2f} seconds.")
        print(f"PSNR (Peak Signal-to-Noise Ratio): {psnr_value}")
        print(f"SSIM (Structural Similarity Index): {ssim_value}")
        print(f"LPIPS (Learned Perceptual Image Patch Similarity): {lpips_value}")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    print("\n--- Image Generation and Metrics ---")
    for metric in metrics:
        print(f"===== Iteration {metric['iteration']} =====")
        print(f"Image 1: {metric['image1_file']} took {metric['image1_duration']:.2f} seconds")
        print(f"Image 2: {metric['image2_file']} took {metric['image2_duration']:.2f} seconds")
        print(f"PSNR of images: {metric['psnr']}")
        print(f"SSIM of images: {metric['ssim']}")
        print(f"LPIPS of images: {metric['lpips']}")
        print("====================================")

    print(f"\nTotal duration for {NUMBER_OF_LOOPS} iterations: {total_duration:.2f} seconds.")

if __name__ == "__main__":
    main()