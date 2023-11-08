import os

URL = f"https://dnd1a27k2xx850-7000.proxy.runpod.net"      # ai server url

NEGATIVE_PROMPT = """oversaturated, ugly, render, cartoon, grain, low-res, kitsch, anime, painting, bad, disfigured"""
# PROMPT = """a professional photo of a bottle and a box of pills at a tropical beach with a blue sea, award winning photography, beautiful, ultra detailed, ultra quality"""
PROMPT = """christmas theme, beautiful, ultra detailed, ultra quality, realistic, shinny, brightness, detailed objects"""

def get_filepath(root_dir):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    return os.path.join(root_dir, root_dir + str(len(os.listdir(root_dir))) + '.png')