import svgpathtools as svg
from icecream import ic
import cv2
import numpy as np
import re
import argparse
import os
import cairosvg


def combinesvgs(fore_dir, back_dir, mask_dir, result_folder, frame):
    bg_paths, bg_attributes, svg_attributes = svg.svg2paths2(back_dir)

    w = int(svg_attributes['width'])
    h = int(svg_attributes['height'])
    mask = cv2.imread(mask_dir)
    # mask = cv2.imread('00049.jpg')
    mask = cv2.resize(mask, (w,h))
    cv2.imwrite("new.jpg", mask)
    ic(len(bg_paths))

    def get_coordinate(path_str):
        pattern = r'\d+\.?\d*'
        matches = re.findall(pattern, path_str)
        return matches

    def neighbor_positive(x,y):
        if x >= 0 and x <= (w-1) and y >= 0 and y <= (h-1) \
        and (mask[int(np.floor(y)),int(np.floor(x)),0] > 0 \
        or mask[int(np.floor(y)),int(np.floor(x))+1,0] > 0 \
        or mask[int(np.floor(y))+1,int(np.floor(x)),0] > 0 \
        or mask[int(np.floor(y))+1,int(np.floor(x))+1,0] > 0):
            return 1
        else:
            return 0
        

    index = []
    index_ = []
    for i, path in enumerate(bg_paths):
        str_path = get_coordinate(path.d())
        # num = 0
        for axis in range(0,8,2):
            pos = np.array((float(str_path[axis]),float(str_path[axis+1])))
            if neighbor_positive(pos[0], pos[1]):
                # num += 1
                if (index==[] or index[-1] != i):
                    index.append(i)
                

        if neighbor_positive(float(str_path[0]), float(str_path[1])):
            bg_paths[i] = bg_paths[i].reversed()
            if neighbor_positive(float(str_path[6]), float(str_path[7])):
                index.remove(i)
                index_.append(bg_paths[i])
        
            # elif num == 4:
            #     index.remove(i)
            #     index_.append(bg_paths[i])

    # svg.wsvg(bg_paths, attributes=bg_attributes, svg_attributes=svg_attributes, filename='output2.svg')


    fore_paths, fore_attributes, _ = svg.svg2paths2(fore_dir)


    num = 0

    rms = []
    for i in index:
        bg_path = bg_paths[i]
        
        min = 2
        result = bg_path
        for fore_path in fore_paths:

            for (T1, seg1, t1), (T2, seg2, t2) in bg_path.intersect(fore_path):
                if min > T1:
                    min = T1

        if min <= 1:
            bg_paths[i] = svg.Path(svg.CubicBezier(start=(bg_paths[i].point(0)), control1=(bg_paths[i].point(T1/3)), control2=(bg_paths[i].point(T1*2/3)), end=(bg_paths[i].point(T1))))
        else:
            rms.append(bg_paths[i])

    ic(len(bg_paths))
    ic(len(bg_attributes))

    for path in index_:
        bg_paths.remove(path)
        bg_attributes.pop(0)

    for rm in rms:
        bg_attributes.pop(0)
        bg_paths.remove(rm)
        # bg_attributes.remove(rms.index(rm))

    for fore_path,fore_attribute in zip(fore_paths,fore_attributes):
        bg_paths.append(fore_path)
        bg_attributes.append(fore_attribute)

    svg.wsvg(bg_paths, attributes=bg_attributes, svg_attributes=svg_attributes, filename=f'{result_folder}/{frame}.svg')

# svg.wsvg(bg_paths, attributes=bg_attributes, svg_attributes=svg_attributes, filename='output3.svg')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocess image sequence')

    parser.add_argument(
        '--foresvgs_folder', type=str, default='../data/', help='foreground svg folder')
    parser.add_argument(
        '--backsvgs_folder', type=str, default='../data/', help='background svg folder')
    parser.add_argument(
        '--masks_folder', type=str, default='../data/', help='mask folder')
    parser.add_argument(
        '--results_folder', type=str, default='../data/', help='result folder')
    
    args = parser.parse_args() 
    
    fore_list = sorted(os.listdir(args.foresvgs_folder))
    ic(fore_list)
    back_list = sorted(os.listdir(args.backsvgs_folder))
    mask_list = sorted(os.listdir(args.masks_folder))
    for i in range(len(fore_list)):
        combinesvgs(f"{args.foresvgs_folder}/{fore_list[i]}", f"{args.backsvgs_folder}/{back_list[i]}", f"{args.masks_folder}/{mask_list[i]}", args.results_folder, i)

    


# image = cairosvg.svg2png("output3.svg")
# ic(image)