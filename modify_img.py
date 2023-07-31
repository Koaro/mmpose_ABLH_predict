import numpy as np
from PIL import Image
import os
from tqdm import tqdm

img_root = 'D:/data/2-feature/33242/'
out_root = 'D:/data/2-feature/184x192_interpolated/'
img_names = os.listdir(img_root)

if not os.path.exists(out_root):
    os.makedirs(out_root)

for n in tqdm(img_names):
    # read image
    img = Image.open(f"{img_root}{n}")
    
    new_img = img.copy()

    # # cut img
    new_img = img.crop((0, 0, 184, 192))

    # ## resize
    # new_img = img.resize((128, 128), Image.Resampling.BICUBIC)

    # ## padding
    # new_img = Image.new(img.mode, (256,256),(0,0,0))
    # new_img.paste(img,(64,0))
    
    # ## edit img
    # width, height = img.size

    # for x in range(width):
    #     new_img.putpixel((x, 0), (0, 0, 0, 255))

    ## saving
    new_img.save(f"{out_root}{n}")


