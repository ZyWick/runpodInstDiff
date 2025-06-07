import runpod
import os
import torch
from diffusers import StableDiffusionINSTDIFFPipeline
from io import BytesIO
import base64
import shutil

# Define volume paths
VOLUME_PATH = "/runpod-volume"
HF_CACHE_DIR = os.path.join(VOLUME_PATH, "huggingface")
MODEL_CACHE_DIR = os.path.join(VOLUME_PATH, "models", "instancediffusion_sd15")

# Use volume for HuggingFace cache
os.environ["HF_HOME"] = HF_CACHE_DIR

# Optional: Check volume space
total, used, free = shutil.disk_usage(VOLUME_PATH)
print(f"Volume - Total: {total // (2**20)} MB | Used: {used // (2**20)} MB | Free: {free // (2**20)} MB")

# Ensure necessary directories exist
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Load or download model
if not os.listdir(MODEL_CACHE_DIR):
    print("Downloading model to volume...")
    pipe = StableDiffusionINSTDIFFPipeline.from_pretrained(
        "kyeongry/instancediffusion_sd15",
        torch_dtype=torch.float16,
        use_safetensors=True,
        cache_dir=MODEL_CACHE_DIR
    )
else:
    print("Loading model from volume cache...")
    pipe = StableDiffusionINSTDIFFPipeline.from_pretrained(
        MODEL_CACHE_DIR,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

pipe = pipe.to("cuda")

def handler(event):
    try:
        # Parse inputs
        prompt = event["input"]["prompt"]
        negative_prompt = event["input"].get("negative_prompt", "")
        phrases = event["input"].get("phrases", [])
        boxes = event["input"].get("boxes", [])

        # Run inference
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            instdiff_phrases=phrases,
            instdiff_boxes=boxes,
            instdiff_scheduled_sampling_alpha=0.8,
            instdiff_scheduled_sampling_beta=0.36,
            guidance_scale=7.5,
            output_type="pil",
            num_inference_steps=50,
        ).images[0]

        # Encode image to base64
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("utf-8")

        return {"image_base64": encoded}
    except Exception as e:
        return {"error": str(e)}

# Start the Serverless function
runpod.serverless.start({"handler": handler})
