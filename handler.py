import runpod
import os
import torch
from diffusers import StableDiffusionINSTDIFFPipeline
from io import BytesIO
import base64

# Use mounted volume to store model cache
os.environ["HF_HOME"] = "/workspace/huggingface"

# Optional: Check free space for debugging
import shutil
total, used, free = shutil.disk_usage("/")
print(f"Disk - Total: {total // (2**20)} MB | Used: {used // (2**20)} MB | Free: {free // (2**20)} MB")

# Use mounted volume to cache model
model_dir = "/models/instancediffusion_sd15"

if not os.path.exists(model_dir):
    print("Downloading model to /models...")
    pipe = StableDiffusionINSTDIFFPipeline.from_pretrained(
        "kyeongry/instancediffusion_sd15",
        torch_dtype=torch.float16,
        use_safetensors=True,
        cache_dir=model_dir
    )
else:
    print("Loading model from /models cache...")
    pipe = StableDiffusionINSTDIFFPipeline.from_pretrained(
        model_dir,
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


# Start the Serverless function when the script is run
runpod.serverless.start({"handler": handler})