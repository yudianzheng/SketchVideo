import os
import cv2
import argparse
from pathlib import Path
import re
import numpy as np
from icecream import ic
from tqdm import tqdm 
import glob

def update_mask(img):
    mask = np.any(img > 0, axis=-1)
    
    # Set all channels to 255 where the mask is True
    img[mask] = [255, 255, 255]

    return img

def bounding(mask):
    pos = np.argwhere(mask>0)
    widths = pos[:, 1]
    
    min_w = np.min(widths)
    max_w = np.max(widths)

    return int((min_w+max_w)/2)

def crop_mask(mask, mid_width):

    h, w, c = mask.shape

    # Define the cropping area
    x1, y1, x2, y2 = int(mid_width-np.floor(h/2)), 0, int(mid_width+np.ceil(int(h/2))), h

    if x1<0:
        x1 = 0
        x2 = h
    if x2>w:
        x2 = w
        x1 = w - h

    # Crop the image using the specified coordinates
    cropped_image = mask[y1:y2, x1:x2]
    # cropped_image = np.any(cropped_image > 128, axis=-1)
    mask = np.any(cropped_image > 0, axis=-1)
    
    # Set all channels to 255 where the mask is True
    cropped_image[mask] = [255, 255, 255]
    # ic(cropped_image.shape)
    return cropped_image

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
    '--output_folder', type=str, default='../data/', help='folder to save')
# parser.add_argument(
#     '--num_frames', type=int, default=50, help='number of frames accepted')
parser.add_argument(
    '--size', nargs='*', type=int, default=[1080,1080], help='number of frames accepted')


args = parser.parse_args() 

size = args.size

# Specify the input and output folders
img_folder = args.image_folder
mask_folder = args.mask_folder
output_folder = args.output_folder



if not os.path.exists(f"{output_folder}"):
    os.makedirs(f"{output_folder}")

img_output = f"{output_folder}/imgs"
mask_output = f"{output_folder}/masks"
# num_frames = args.num_frames

imgs_folder_names = [f for f in os.listdir(img_folder) if os.path.isdir(os.path.join(img_folder, f))]

for video_name in imgs_folder_names:
    
    num_frames = len(glob.glob(os.path.join(f"{img_folder}/{video_name}", '*.jpg'))) + len(glob.glob(os.path.join(f"{img_folder}/{video_name}", '*.png')))

    ic(f"{img_folder}/{video_name}")
    ic(num_frames)

    if not os.path.exists(f"{output_folder}/{video_name}"):
        os.makedirs(f"{output_folder}/{video_name}")

    for img_name in tqdm(sorted(os.listdir(Path(f"{img_folder}/{video_name}"))), video_name):
        
        number = re.findall('\d+', img_name)[0]
        if int(number) > num_frames:
            break

        whole_img = cv2.imread(os.path.join(f"{img_folder}/{video_name}/{img_name}"))
        
        mask_name = img_name.replace(".jpg", ".png")
        # ic(f"{mask_folder}/{video_name}/{mask_name}")
        mask = cv2.imread(os.path.join(f"{mask_folder}/{video_name}/{mask_name}"))

        whole_mask = update_mask(mask)
        try:
            width_mid = bounding(mask)
        except:
            continue
        mask = crop_mask(mask, width_mid)
        img = crop_img(whole_img, width_mid)
        cropped = img * (mask / 255)
        filling = np.any(cropped == 0, axis=-1)
        cropped[filling] = [255,255,255]

        mask = cv2.resize(mask, (size[0], size[1]))[:,:,0]
        img = cv2.resize(img,  (size[0], size[1]))
        cropped = cv2.resize(cropped, (size[0], size[1]))


        if not os.path.exists(f"{output_folder}/{video_name}/imgs"):
            os.makedirs(f"{output_folder}/{video_name}/imgs")
        if not os.path.exists(f"{output_folder}/{video_name}/masks"):
            os.makedirs(f"{output_folder}/{video_name}/masks")
        if not os.path.exists(f"{output_folder}/{video_name}/imgs_crop"):
            os.makedirs(f"{output_folder}/{video_name}/imgs_crop")
        if not os.path.exists(f"{output_folder}/{video_name}/imgs_crop_fore"):
            os.makedirs(f"{output_folder}/{video_name}/imgs_crop_fore")
        if not os.path.exists(f"{output_folder}/{video_name}/masks_crop"):
            os.makedirs(f"{output_folder}/{video_name}/masks_crop")
        if not os.path.exists(f"{output_folder}/{video_name}/imgs_crop_maskrcnn"):
            os.makedirs(f"{output_folder}/{video_name}/imgs_crop_maskrcnn")

        cv2.imwrite(os.path.join(f"{output_folder}/{video_name}/masks/{mask_name}"), whole_mask)
        cv2.imwrite(os.path.join(f"{output_folder}/{video_name}/imgs/{img_name}"), whole_img)
        cv2.imwrite(os.path.join(f"{output_folder}/{video_name}/imgs_crop/{img_name}"), img)
        cv2.imwrite(os.path.join(f"{output_folder}/{video_name}/masks_crop/{img_name}"), mask)
        cv2.imwrite(os.path.join(f"{output_folder}/{video_name}/imgs_crop_maskrcnn/{img_name}"), mask)
        cv2.imwrite(os.path.join(f"{output_folder}/{video_name}/imgs_crop_fore/{img_name}"), cropped)
