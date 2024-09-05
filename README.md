# HiDiffusion Demos

This repository contains scripts to demonstrate and compare the performance and output quality of the HiDiffusion method applied to Stable Diffusion models.

## Overview

HiDiffusion is designed to enhance the resolution and speed of pre-trained diffusion models. It can be integrated with diffusion pipelines using a single line of code. These scripts will help you understand the impact of HiDiffusion on image generation. 

## Installation

1. **Clone the repository**:
    ```shell
    git clone https://github.com/megvii-model/HiDiffusion.git
    cd HiDiffusion
    ```

2. **Create a virtual environment**:
    ```shell
    python -m venv hidiff_env
    ```

3. **Activate the virtual environment**:
    - On Windows:
      ```shell
      .\hidiff_env\Scripts\activate
      ```
    - On macOS and Linux:
      ```shell
      source hidiff_env/bin/activate
      ```

4. **Install required packages**:
    ```shell
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install diffusers==0.27.0 transformers==4.27.4 accelerate==0.18.0
    pip install lpips scikit-image numpy pillow
    ```

5. **Install HiDiffusion**:
    ```shell
    pip install hidiffusion
    ```

## Scripts

### 1. HiDiffusion Demo Script (`hidiffusion_demo.py`)

This script demonstrates the effect of HiDiffusion on image generation. It generates two images from the same prompt: one with HiDiffusion applied and one without, allowing you to observe the improvements.

#### How it works:
1. **Load the Ghibli Diffusion Model**: Uses the `StableDiffusionImg2ImgPipeline` model with LMSDiscreteScheduler.
2. **Disable the Safety Checker**: Safety checker is disabled for this demo.
3. **Define Prompts**: A prompt and a negative prompt guide the generation.
4. **Load and Resize the Initial Image**: The initial image is loaded and resized.
5. **Generate Images**: 
   - One without HiDiffusion
   - One with HiDiffusion
6. **Save and Compare**: Saves the images with timestamps and prints the time taken for each process.

### Running `hidiffusion_demo.py`

1. **Activate your virtual environment**:
    - On Windows:
      ```shell
      .\hidiff_env\Scripts\activate
      ```
    - On macOS and Linux:
      ```shell
      source hidiff_env/bin/activate
      ```

2. **Run the script**:
    ```shell
    python hidiffusion_demo.py
    ```

### 2. Image-to-Image Comparison Script (`img2img_comparison.py`)

This script compares the image-to-image transformation quality using the Stable Diffusion model both with and without HiDiffusion applied.

#### How it works:
1. **Load the Model and Scheduler**: Uses the `StableDiffusionImg2ImgPipeline` model with the DDIM scheduler.
2. **Apply/Remove HiDiffusion**: Generates images with and without HiDiffusion.
3. **Define Prompts**: A primary text prompt and a negative prompt guide the generation.
4. **Generate Images**:
   - Two images are generated: one with HiDiffusion and one without.
5. **Evaluate Metrics**:
   - PSNR (Peak Signal-to-Noise Ratio)
   - SSIM (Structural Similarity Index)
   - LPIPS (Learned Perceptual Image Patch Similarity)
6. **Print Results**: The script prints and saves the evaluation metrics and time taken for each generation.

### Running `img2img_comparison.py`

1. **Activate your virtual environment**:
    - On Windows:
      ```shell
      .\hidiff_env\Scripts\activate
      ```
    - On macOS and Linux:
      ```shell
      source hidiff_env/bin/activate
      ```

2. **Run the script**:
    ```shell
    python img2img_comparison.py
    ```

## Detailed Script Usage

### `hidiffusion_demo.py`

This script showcases the performance and quality difference in generating images with and without HiDiffusion. It uses the Ghibli Diffusion model to generate images based on a specific prompt.

**Example Prompts**:
- **Prompt**:
ghibli style portrait of a family, with detailed facial features such as eyes, nose, lips, and hair, in the style of Studio Ghibli. Realistic shading, expressive eyes, clear skin texture.

- **Negative Prompt**:
blurry, ugly, duplicate, poorly drawn face, deformed, mosaic, artifacts, bad limbs

### `img2img_comparison.py`

This script compares the quality and performance of image-to-image generation by generating images from an initial reference image, both with and without HiDiffusion. Metrics such as PSNR, SSIM, and LPIPS are calculated to quantitatively assess the differences.

### Output

After running the scripts, you will find generated images and logs providing insights into the performance and quality metrics in the respective output directories (`hidiffusion_demo_outputs` and `img2img_comparison_outputs`).

### Troubleshooting

If you encounter any issues:
- Verify that all required packages are installed.
- Ensure the virtual environment is activated.
- Check GPU compatibility and CUDA installation.

### Acknowledgements

This repository leverages the following:
- [HiDiffusion](https://pypi.org/project/hidiffusion/)
- [Diffusers](https://huggingface.co/docs/diffusers/)
- [PyTorch](https://pytorch.org/)
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity)

### License

This project is licensed under the Apache-2.0 License.

---

For any questions or issues, feel free to raise an issue in the repository or contact the maintainer.
