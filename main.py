from fastapi import Body, FastAPI
import uvicorn
from utils import encode_pil_to_base64
from txt2img import sdxl_txt2img
from config import configs, set_configs

config = configs()

app = FastAPI()

@app.get("/")
async def root():
    return 'Hello, this is our hands-on stable diffusion xl code'

@app.post("/v1/txt2img")
def txt2img(
        prompt_1: str = Body("", title='Prompt1'),
        prompt_2: str = Body("", title='Prompt2'),
        negative_prompt_1: str = Body("", title='Negative Prompt1'),
        negative_prompt_2: str = Body("", title='Negative Prompt2'),
        width: int = Body(0, title='Width'), 
        height: int = Body(0, title='Height'), 
        seed: int = Body(-1, title='Seed'), 
        guidance_scale: int = Body(-1, title='Guidance Scale'),
        num_inference_steps : int = Body(0, title='Inference Steps'),
        num_images_per_prompt : int = Body(0, title='Inference Steps')
    ):

    config = set_configs(config, prompt_1, prompt_2, negative_prompt_1, negative_prompt_2, width, height, seed, guidance_scale, num_inference_steps, num_images_per_prompt)

    images = sdxl_txt2img(config=config)

    base64_images = []
    for image in images:
        base64_images.append(encode_pil_to_base64(image=image))

    return {'images': base64_images}


if __name__ == "__main__":
    uvicorn.run('main:app', host='0.0.0.0', port=7000, workers=10)
