import base64, requests, io, os
from PIL import Image

URL = f"https://dnd1a27k2xx850-7000.proxy.runpod.net"      # ai server url

NEGATIVE_PROMPT = """oversaturated, ugly, render, cartoon, grain, low-res, kitsch, anime, painting, bad, disfigured"""
# PROMPT = """a professional photo of a bottle and a box of pills at a tropical beach with a blue sea, award winning photography, beautiful, ultra detailed, ultra quality"""
PROMPT = """christmas theme, beautiful, ultra detailed, ultra quality, realistic, shinny, brightness, detailed objects"""

def get_filepath(root_dir):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    return os.path.join(root_dir, root_dir + str(len(os.listdir(root_dir))) + '.png')

def _main():

    # request body (ai server parameters)
    payload ={
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "width": 1024, 
        "height": 1024, 
        "guidance_scale": 7,
        "num_inference_steps" : 60,
    }

    # send post request to generate image
    print(f'{URL}/v1/txt2img')
    response = requests.post(url=f'{URL}/v1/txt2img', headers={ 'Content-Type' : 'application/json' }, json=payload) 

    print(response.status_code)

    if response.status_code == 200:                                                     # if image is generated
        r = response.json()                                                             # response type : { image: ' image base64 '}
        res_image =  r['image']                                                         # get image as base64
        image = Image.open(io.BytesIO(base64.b64decode(res_image.split(",",1)[0])))     # decode base64 to image
        image.save(get_filepath('txt2img'))                                             # save image to txt2img directory


if __name__ == '__main__':
    _main()