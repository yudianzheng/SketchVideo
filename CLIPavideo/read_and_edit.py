import glob
from math import sqrt
import os
import pydiffvg
import torch
from natsort import natsorted
from PIL import Image
import numpy as np 
import torchvision
from sketch_utils import read_svg as read_svg_old


svgs = glob.glob('/apdcephfs/private_shadowcun/CLIPavideo/CLIPavideo/output_sketches/car-turn3000/*.svg')
masks = glob.glob('/apdcephfs_cq2/share_1290939/shadowcun/sketching-video-49/car-turn/masks_crop/*.jpg')

svgs = natsorted(svgs)
masks = natsorted(masks)

threshold = 0.8

# def similarity():
#     # given two path, return the similarity of these two points.

def read_svg(path_svg, path_mask, device, multiply=False):
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        path_svg)
    
    if multiply:
        canvas_width *= 2
        canvas_height *= 2
        for path in shapes:
            path.points *= 2
            path.stroke_width *= 2
    
    _render = pydiffvg.RenderFunction.apply

    #### remove the points unnecessary.
    # 1. lines are gather together.
    # 2. short line
    # 3. out of mask

    mask = np.array(Image.open(path_mask).convert('L').resize((224,224)))/255.
    
    # lines are gather together
    # Brute force to find the similar lines.

    # for i_point in range(len(shapes)):
    #     for j_point in range(i_point, len(shapes)):
             
    #         if similarity(shapes[i_point], shapes[j_point]) > threshold:  # their are two lines similar in this frame, may consider to remove it.
                


    for i_point in range(len(shapes)):

        len_of_points = 0
        for i_idx in range(len(shapes[i_point].points) - 1 ): # for each point,  (4 points)
            ix1, iy1 = [ int(x) for x in shapes[i_point].points[i_idx] ] # the location of each points
            ix2, iy2 = [ int(x) for x in shapes[i_point].points[i_idx + 1] ] # the location of each points

            distance_ = sqrt((iy2 - iy1)**2 + (ix2 - ix1)**2)

            len_of_points += distance_

        area_ratio = np.sum(mask)/(canvas_height*canvas_width)
        # print(len_of_points, (1 - area_ratio )**2 * 25 ) # larger mask, small threshhold; small mask, large threshold but smoother.
        if len_of_points < ( 1 - area_ratio )**2 * 25 : # 300 - np.sum(mask*0.01)**2:
            shape_groups[i_point].stroke_color = torch.tensor([0,0,0,0])

    # out of mask
    for i_point in range(len(shapes)):
        mm = 0
        for i_idx in range(len(shapes[i_point].points)): # for each point,  (4 points)
            ix, iy = [ int(x) for x in shapes[i_point].points[i_idx] ]

            if iy < canvas_height and ix < canvas_width:
                mm += mask[iy, ix] # inner of mask
        
        if mm == 0:
            shape_groups[i_point].stroke_color = torch.tensor([0,0,0,0])

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    
    img = _render(canvas_width,  # width
                  canvas_height,  # height
                  2,   # num_samples_x
                  2,   # num_samples_y
                  0,   # seed
                  None,
                  *scene_args)

    img = img[:, :, 3:4] * img[:, :, :3] + \
        torch.ones(img.shape[0], img.shape[1], 3,
                   device=device) * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    return img


img_list = []

for svg, mask in zip(svgs, masks):
    img_old = read_svg_old(svg, 'cuda')
    img = read_svg(svg, mask, 'cuda')
    img_list.append( np.concatenate([np.array(img.cpu().numpy()*255), np.array(img_old.cpu().numpy()*255)], 1))

torchvision.io.write_video('car-turn.mp4', np.array(img_list), fps=10)