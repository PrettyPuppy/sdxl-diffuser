from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
import torch
from fastapi import Body, FastAPI
import uvicorn
from utils import encode_pil_to_base64

# establish stable diffusion xl model pipeline
generator_pipe: StableDiffusionXLPipeline  = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

# move the pipeline to the GPU
generator_pipe.to("cuda")

app = FastAPI()

@app.get("/")
async def root():
    return 'Hello, this is our hands-on stable diffusion xl code'

@app.post("/v1/txt2img")
def inference(
        prompt: str = Body("", title='Prompt'),
        negative_prompt: str = Body("", title='Negative Prompt'),
        width: int = Body(1024, title='Width'), 
        height: int = Body(1024, title='Height'), 
        guidance_scale: int = Body(7, title='Guidance Scale'),
        num_inference_steps : int = Body(60, title='Inference Steps')
    ):
    # generate the image
    generated_image = generator_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=width,
            width=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]

    # convert image to base64
    b64image = encode_pil_to_base64(generated_image)

    return {"image": b64image}

if __name__ == "__main__":
    uvicorn.run(app=app, host='0.0.0.0', port=7000)
