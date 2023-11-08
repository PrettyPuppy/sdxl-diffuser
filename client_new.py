import os
import io
import gc
import time
import shutil
import argparse
from pathlib import Path
import base64
import requests
from PIL import Image
from src.read_json import read_json
from config import get_filepath

def main(args):
    """
    Midjourney main entry point
    """
    # Timer
    start_time = time.time()

    # # Remove the image folder if it exists
    # if os.path.exists(args.image_dir):
    #     shutil.rmtree(args.image_dir)

    # # Create a new image folder
    # os.makedirs(args.image_dir)

    # Payload
    payload_data = read_json(args.payload_dir)

    # send post request to generate image
    response = requests.post(url=f"{args.run_url}/v1/txt2img", headers={ "Content-Type" : "application/json" }, json=payload_data) 

    print(response.status_code)

    if response.status_code == 200:                                                     # if image is generated
        res = response.json()                                                           # response type : { image: " image base64 "}
        res_image =  res["image"]                                                       # get image as base64
        image = Image.open(io.BytesIO(base64.b64decode(res_image.split(",",1)[0])))     # decode base64 to image
        image.save(get_filepath("txt2img"))                                             # save image to txt2img directory


    # Write into log file
    end_time = time.time()
    msg = f"Total processing time: {end_time - start_time} seconds"
    print(msg)

    # Delete class objects and clean the buffer memory using the garbage collection
    gc.collect()


if __name__ == "__main__":
    """
    Form command lines
    """
    # Clean up buffer memory
    gc.collect()

    # Current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    test_name    = "example01"
    payload_file = "payload.json"
    payload_dir  = os.path.join(current_dir, "tests", test_name, payload_file)
    run_url      = f"https://dnd1a27k2xx850-7000.proxy.runpod.net"

    # Add options
    p = argparse.ArgumentParser()
    p.add_argument("--payload_dir", type=Path, default=payload_dir)
    p.add_argument("--run_url", type=str, default=run_url)
    args = p.parse_args()

    main(args)