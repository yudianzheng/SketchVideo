import os
import cv2
import argparse
from pathlib import Path
import re
import numpy as np

def bounding(mask):
    pos = np.argwhere(mask>0)
    widths = pos[:, 1]
    
    min_w = np.min(widths)
    max_w = np.max(widths)

    return int((min_w+max_w)/2)

def crop_img(img, mid_width):

    h, w, c = img.shape

    # Define the cropping area
    x1, y1, x2, y2 = int(mid_width-np.floor(h/2)), 0, int(mid_width+np.ceil(int(h/2))), h 
    
    if x1<0:
        x1 = 0
        x2 = h
    if x2>w:
        x2 = w
        x1 = w - h
    
    # Crop the image using the specified coordinates
    cropped_image = img[y1:y2, x1:x2]
    
    return cropped_image

parser = argparse.ArgumentParser(description='Preprocess image sequence')

parser.add_argument(
    '--image_folder', type=str, default='../data/', help='folder to process')
parser.add_argument(
    '--mask_folder', type=str, default='../data/', help='folder to process')
parser.add_argument(
    '--size', nargs='*', type=int, default=[1080,1080], help='number of frames accepted')


args = parser.parse_args() 

size = args.size

# Specify the input and output folders
img_folder = args.image_folder
mask_folder = args.mask_folder

output = f"{img_folder}/imgs_bg_crop"

for img_name in sorted(os.listdir(Path(f"{img_folder}/imgs_bg"))):

    img = cv2.imread(os.path.join(f"{img_folder}/imgs_bg/{img_name}"))
    mask_name = img_name.replace(".jpg", ".png")
    mask = cv2.imread(os.path.join(f"{img_folder}/masks/{mask_name}"))

    whole_img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    
    width_mid = bounding(mask)
    img = crop_img(whole_img, width_mid)

    img = cv2.resize(img,  (size[1], size[0]))

    if not os.path.exists(output):
        os.makedirs(output)

    cv2.imwrite(os.path.join(f"{output}/{img_name}"), img)
