import random
import CLIP_.clip as clip
import numpy as np
import pydiffvg
import sketch_utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torchvision import transforms

import matplotlib.pyplot as plt

import copy
import random
from icecream import ic
from MLP_models.MLP import WidthMLP, init_weights
import os


class Painter(torch.nn.Module):
    def __init__(self, args,
                num_strokes=4,
                num_segments=4,
                imsize=224,
                device=None,
                target_im=None,
                ratio=None,
                atlas=None,
                optic_flow=None,
                mask=None):
        super(Painter, self).__init__()

        self.args = args

#         self.optic_flow = torch.from_numpy(optic_flow).to(args.device)
        self.num_frames = len(target_im)
        self.atlas = atlas
        self.ratio = ratio

        self.num_paths = num_strokes
        self.num_segments = num_segments
        self.width = args.width
        self.control_points_per_seg = args.control_points_per_seg
        self.opacity_optim = args.force_sparse
        self.num_stages = args.num_stages
        self.add_random_noise = "noise" in args.augemntations
        self.noise_thresh = args.noise_thresh
        self.softmax_temp = args.softmax_temp

        # self.test_shapes = torch.nn.ModuleList([])
        # self.frames_shapes = torch.nn.ModuleList()
        self.shapes_frames_index = []
        self.frames_shapes = []
        self.frames_shape_groups = []

        # self.addition_shapes = []
        # self.addition_shape_groups = []
        # self.addition_frames_shapes = []
        # self.addition_frames_shape_groups = []
        # self.addition_frames_optimize_flag = []
        # self.addition_optimize_flag = []

        self.shapes = []
        self.shape_groups = []
        self.device = device
        self.canvas_width, self.canvas_height = imsize, imsize
        self.points_vars = []
        self.color_vars = []
        self.color_vars_threshold = args.color_vars_threshold

        self.path_svg = args.path_svg
        self.strokes_per_stage = self.num_paths
        self.frames_optimize_flag = []
        self.optimize_flag = []

        # attention related for strokes initialisation
        self.attention_init = args.attention_init
        self.target_path = args.target
        self.saliency_model = args.saliency_model
        self.xdog_intersec = args.xdog_intersec
        self.mask_object = args.mask_object_attention
        
        self.text_target = args.text_target # for clip gradients
        self.saliency_clip_model = args.saliency_clip_model
        # self.define_attention_input(target_im[0])
        # self.define_attention_input(target_im)
        # self.mask = mask
        # self.attention_map = self.set_attention_map() if self.attention_init else None
        
        # self.thresh = self.set_attention_threshold_map() if self.attention_init else None
        # self.strokes_counter = 0 # counts the number of calls to "get_path"        
        # self.epoch = 0
        # self.final_epoch = args.num_iter - 1
        # tensor_image = ((atlas[0] - atlas[0].min()) / (atlas[0].max() - atlas[0].min())).squeeze(0).cpu()
        # pil_image = Image.fromarray((tensor_image * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy())
        # pil_image.save(f"{self.args.output_dir}/mp4_logs_{self.args.text}_{self.args.clip_model_name}_{''.join(str(x) for x in self.args.clip_conv_layer_weights)}_{self.args.clip_fc_loss_weight}_{self.args.width}/atlas1.jpg")


        self.image_atlas_attn_clip = self.define_attention_input(atlas)
        self.image_input_attn_clip = self.define_attention_input(target_im)
        self.mask = mask
        
        # tensor_image = ((self.image_atlas_attn_clip[0] - self.image_atlas_attn_clip[0].min()) / (self.image_atlas_attn_clip[0].max() - self.image_atlas_attn_clip[0].min())).squeeze(0).cpu()
        # pil_image = Image.fromarray((tensor_image * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy())
        # pil_image.save(f"{self.args.output_dir}/mp4_logs_{self.args.text}_{self.args.clip_model_name}_{''.join(str(x) for x in self.args.clip_conv_layer_weights)}_{self.args.clip_fc_loss_weight}_{self.args.width}/atlas2.jpg")

        # self.attention_map = self.set_attention_map() if self.attention_init else None
        # ic(self.image_atlas_attn_clip)
        self.attention_atlas_map = self.clip_attn(self.image_atlas_attn_clip) if self.attention_init else None
        self.attention_input_map = self.clip_attn(self.image_input_attn_clip) if self.attention_init else None


        # self.thresh = self.set_attention_threshold_map() if self.attention_init else None
        # self.thresh, _, self.inds_normalised_atlas = self.set_inds_clip(self.attenion_atlas_map, "atlas") if self.attention_init else None
        # self.thresh, self.inds_normalised_input_index, self.inds_normalised_input = self.set_inds_clip(self.attenion_input_map, "input") if self.attention_init else None
        self.thresh, self.inds_normalised_list_atlas = self.set_inds_clip(self.attention_atlas_map, "atlas") if self.attention_init else None
        self.thresh, self.inds_normalised_list_input = self.set_inds_clip(self.attention_input_map, "input") if self.attention_init else None
        
        # self.inds_normalised_list_input = self.set_random_init()
        
        self.strokes_counter = 0 # counts the number of calls to "get_path"        
        self.epoch = 0
        self.final_epoch = args.num_iter - 1
        
        self.width_on = 0
        self.loss_sparse_con = torch.zeros((self.num_frames,1)).to(self.device)
        self.addition_on = 0
        self.main_on = 1
        self.paths4Frames = torch.zeros((self.num_frames)).to(device)

        self.plot(45)

    def plot(self,i):
        plt.figure(figsize=(10, 5))
        threshold_map = self.thresh[i]
        threshold_map_ = (threshold_map - threshold_map.min()) / \
        (threshold_map.max() - threshold_map.min())
        plt.imshow(threshold_map_, interpolation='nearest', vmin=0, vmax=1)
        plt.title("prob softmax")
        # plt.scatter(self.inds[:, 1], self.inds[:, 0], s=10, c='red', marker='o')
        plt.axis("off")
        plt.savefig(self.args.output_dir)
        plt.close()
    
    def mlp_width_weight_init(self):
        if self.mlp_width_weights_path == "none":
            self.mlp_width.apply(init_weights)
        else:
            checkpoint = torch.load(self.mlp_width_weights_path)
            self.mlp_width.load_state_dict(checkpoint['model_state_dict'])
            print("mlp checkpoint loaded from ", self.mlp_width_weights_path)
    
    def init_width(self):
        self.init_widths = torch.ones((self.num_paths)).to(self.device) * self.width
        self.mlp_width = WidthMLP(num_strokes=self.num_paths, num_cp=self.control_points_per_seg, width_optim=self.width_on).to(self.device)
        self.mlp_width_weights_path = self.args.mlp_width_weights_path
        self.mlp_width_weight_init()
        
        self.width_on = 1
    
    
    def init_image(self, stage=0):
        if stage > 0:
            # if multi stages training than add new strokes on existing ones
            # don't optimize on previous strokes
            self.optimize_flag = [False for i in range(len(self.shapes))]
            for i in range(self.strokes_per_stage):
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                path = self.get_path()
                self.shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                    fill_color = None,
                                                    stroke_color = stroke_color)
                self.shape_groups.append(path_group)
                self.optimize_flag.append(True)

        else:
            num_paths_exists = 0
            if self.path_svg != "none":
                self.canvas_width, self.canvas_height, self.shapes, self.shape_groups = utils.load_svg(self.path_svg)
                # if you want to add more strokes to existing ones and optimize on all of them
                num_paths_exists = len(self.shapes)

            for i in range(num_paths_exists, self.num_paths):
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                path = self.get_img_path()
                self.shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                    fill_color = None,
                                                    stroke_color = stroke_color)
                self.shape_groups.append(path_group)        
            self.optimize_flag = [True for i in range(len(self.shapes))]
        
        img = self.render_warp()
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = self.device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device) # NHWC -> NCHW
        return img
        # utils.imwrite(img.cpu(), '{}/init.png'.format(args.output_dir), gamma=args.gamma, use_wandb=args.use_wandb, wandb_name="init")

    def interpolate_key_frames(self):

        # for i in range(len(frames_shapes)):
        for i in range(len(self.frames_shapes)):
            n = int(np.floor(i/self.args.interval_num))
            w1 = (i - n*self.args.interval_num)/self.args.interval_num
            w2 = 1-w1
            for j in range(len(self.frames_shapes[0])):
                if (n+1)*self.args.interval_num <= len(self.frames_shapes)-1:
                    self.frames_shapes[i][j].points = (w2*self.frames_shapes[n*self.args.interval_num][j].points + w1*self.frames_shapes[(n+1)*self.args.interval_num][j].points).detach()
                else:
                    interval = (len(self.frames_shapes)-1) % self.args.interval_num
                    w1 = (i - n*self.args.interval_num)/interval
                    w2 = 1-w1
                    self.frames_shapes[i][j].points = (w2*self.frames_shapes[n*self.args.interval_num][j].points + w1*self.frames_shapes[len(self.frames_shapes)-1][j].points).detach()

    def apply_optic_flow(self):
        # import random
        for i in range(self.args.num_of_frames-1):
        # for i in range(10):
            for path in self.shapes:
                for point in path.points:
                    # x_int = max(0, min(223, int(torch.round(point[0]))))
                    # y_int = max(0, min(223, int(torch.round(point[1]))))
                    x_int = int(torch.floor(point[0]))
                    x_float = point[0] - x_int
                    y_int = int(torch.floor(point[1]))
                    y_float = point[1] - y_int
                    # point[0] = torch.max(torch.tensor(0), torch.min(torch.tensor(223), point[0]+self.optic_flow[x_int, y_int, 0, i, 1]))
                    point[0] = point[0]-self.optic_flow[x_int, y_int, 0, i, 0]
                    # point[1] = torch.max(torch.tensor(0), torch.min(torch.tensor(223), point[1]+self.optic_flow[x_int, y_int, 1, i, 1]))
                    point[1] = point[1]-self.optic_flow[x_int, y_int, 1, i, 0]
            
            self.optimize_flag = [True for i in range(len(self.shapes))]

            # self.frames_shapes = torch.nn.moduleList(self.Frames)
            self.frames_shapes.append(copy.deepcopy(self.shapes))
            self.frames_shape_groups.append(copy.deepcopy(self.shape_groups))
            self.frames_optimize_flag.append(copy.deepcopy(self.optimize_flag))

    def apply_same(self):
        for i in range(self.args.num_of_frames-1):
            self.optimize_flag = [True for i in range(len(self.shapes))]
            self.frames_shapes.append(copy.deepcopy(self.shapes))
            self.frames_shape_groups.append(copy.deepcopy(self.shape_groups))
            self.frames_optimize_flag.append(copy.deepcopy(self.optimize_flag))

    def apply_addition(self):
        for i in range(self.args.num_of_frames-1):
            self.addition_optimize_flag = [True for i in range(len(self.addition_shapes))]
            self.addition_frames_shapes.append(copy.deepcopy(self.addition_shapes))
            self.addition_frames_shape_groups.append(copy.deepcopy(self.addition_shape_groups))
            self.addition_frames_optimize_flag.append(copy.deepcopy(self.addition_optimize_flag))

    def init_frames_flow(self):
        for i in range(self.num_paths):
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            path = self.get_img_path(0)
            self.shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                fill_color = None,
                                                stroke_color = stroke_color)
            self.shape_groups.append(path_group)

        self.optimize_flag = [True for i in range(len(self.shapes))]

        # self.frames_shapes.append(self.shapes)
        self.frames_shapes.append(copy.deepcopy(self.shapes))
        self.frames_shape_groups.append(copy.deepcopy(self.shape_groups))
        self.frames_optimize_flag.append(copy.deepcopy(self.optimize_flag))

        self.apply_optic_flow()

    def init_points(self):
        frames_points = []
        points = []
        for frame in range(self.args.num_of_frames):
            self.strokes_counter = 0
            for i in range(self.num_paths):
                self.num_control_points = torch.zeros(self.num_segments, dtype = torch.int32) + (self.control_points_per_seg - 2)
                p0 = self.inds_normalised_list[frame][self.strokes_counter] if self.attention_init else (random.random(), random.random())
                points.append(copy.deepcopy(p0))
                self.strokes_counter += 1
            frames_points.append(copy.deepcopy(points))
            points = []
        return frames_points
    
    def align_frames(self, frames_points):
        result_frames = []
        result_points = []
        result_frames.append(copy.deepcopy(frames_points[0]))
        for frame_num in range(1, self.args.num_of_frames):
            for i in range(self.num_paths):
                pos = min(range(len(frames_points[frame_num])), 
                        key=lambda num:((frames_points[frame_num][num][0]-frames_points[frame_num-1][0][0])**2+(frames_points[frame_num][0][1]-frames_points[frame_num-1][0][1])**2))
                result_points.append(copy.deepcopy(frames_points[frame_num][pos]))
                frames_points[frame_num-1].remove(frames_points[frame_num-1][0])
                frames_points[frame_num].remove(frames_points[frame_num][pos])
            result_frames.append(copy.deepcopy(result_points))
            frames_points[frame_num] = copy.deepcopy(result_points)
            result_points = []

        return result_frames
    
    def random_path(self):
        points_offset = []
        paths_offset = []
        radius = 0.05
        stroke_colors = []
        for i in range(self.num_paths):
            p0 = [0,0]
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            # stroke_colors.append(torch.tensor([random.random(), random.random(), random.random(), random.random()]))
            points_offset.append((p0[0],p0[1]))
            for j in range(self.control_points_per_seg - 1):
                offset = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                # offset = (0,0)
                points_offset.append(offset)
                p0 = offset
            paths_offset.append(points_offset)
            points_offset = []
        
        return paths_offset, stroke_colors


    def init_frames_align(self):
        frames_points = self.init_points()
        result_frames = self.align_frames(frames_points)
        offsets, stroke_colors = self.random_path()
        stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
        for i in range(self.args.num_of_frames):
            for j in range(self.num_paths):
                points = []
                for k in range(self.control_points_per_seg):
                    point = [sum(x) for x in zip(result_frames[i][j], offsets[j][k])]
                    points.append(point)

                points = torch.tensor(points).to(self.device)
                points[:, 0] *= self.canvas_width
                points[:, 1] *= self.canvas_height
        
                path = pydiffvg.Path(num_control_points = self.num_control_points,
                                        points = points,
                                        stroke_width = torch.tensor(self.width),
                                        is_closed = False)
                self.shapes.append(path)

                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                    fill_color = None,
                                                    stroke_color = stroke_color)
                self.shape_groups.append(path_group)
            
            self.optimize_flag = [True for i in range(len(self.shapes))]

            self.frames_shapes.append(copy.deepcopy(self.shapes))
            self.shapes = []
            self.frames_shape_groups.append(copy.deepcopy(self.shape_groups))
            self.shape_groups = []
            self.frames_optimize_flag.append(copy.deepcopy(self.optimize_flag))
            self.optimize_flag = []

    def init_atlas_points(self):
        frames_paths = []
        paths = []
        points = []
        radius = 0.05
        for frame in range(len(self.atlas)):
            self.strokes_counter = 0
            for i in range(self.num_paths):
                self.num_control_points = torch.zeros(self.num_segments, dtype = torch.int32) + (self.control_points_per_seg - 2)
                p0 = self.inds_normalised_list_atlas[frame][self.strokes_counter] if self.attention_init else (random.random(), random.random())
                points.append(copy.deepcopy(p0))
                for j in range(self.control_points_per_seg - 1):
                    # p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                    p1 = [p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5)]
                    points.append(p1)
                    p0 = p1
                paths.append(copy.deepcopy(points))
                points = []
                self.strokes_counter += 1
            frames_paths.append(copy.deepcopy(paths))
            paths = []
        return frames_paths

    # def init_addition_path(self, num):
    #     self.strokes_counter = 0
    #     for i in range(num):
    
    #         # stroke_color = torch.tensor([1.0, 0.0, 0.0, 0.0])
    #         stroke_color = torch.tensor([random.random(), random.random(), random.random(), random.random()])
            
    #         path = self.get_addition_path()

    #         self.addition_shapes.append(path)
    #         path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.addition_shapes) - 1]),
    #                                             fill_color = None,
    #                                             stroke_color = stroke_color)
    #         self.addition_shape_groups.append(path_group)

    #     self.addition_optimize_flag = [True for i in range(len(self.addition_shapes))]

    #     # self.frames_shapes.append(self.shapes)
    #     self.addition_frames_shapes.append(copy.deepcopy(self.addition_shapes))
    #     self.addition_frames_shape_groups.append(copy.deepcopy(self.addition_shape_groups))
    #     self.addition_frames_optimize_flag.append(copy.deepcopy(self.addition_optimize_flag))

    #     self.apply_addition()

    def init_addition(self, num):
        self.num_addition = num
        self.strokes_counter = 0
        for i in range(num):
            # stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            stroke_color = torch.tensor([random.random(), random.random(), random.random(), random.random()])
            path, index = self.get_images_path()
            for j in range(self.num_frames):
                self.frames_shapes[j].append(copy.deepcopy(path))
                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.frames_shapes[j]) - 1]),
                                                    fill_color = None,
                                                    stroke_color = stroke_color)
                self.frames_shape_groups[j].append(copy.deepcopy(path_group))

                self.frames_optimize_flag[j].append(True)

    # def init_frames_atlas(self, bg):
    #     frames_paths = self.init_atlas_points()
    #     if bg==1:
    #         for i in range(self.num_paths):
    #             # stroke_color = torch.tensor([random.random(), random.random(), random.random(), random.random()])
    #             stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
    #             points = frames_paths[1][i]
    #             points = torch.tensor(points).to(self.device)
    #             points[:, 0] *= self.canvas_width
    #             points[:, 1] *= self.canvas_height
        
    #             path = pydiffvg.Path(num_control_points = self.num_control_points,
    #                                     points = points,
    #                                     stroke_width = torch.tensor(self.width),
    #                                     is_closed = False)
    #             self.shapes.append(path)

    #             path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
    #                                                 fill_color = None,
    #                                                 stroke_color = stroke_color)
    #             self.shape_groups.append(path_group)
            
    #         self.optimize_flag = [True for i in range(len(self.shapes))]

    #         self.frames_shapes.append(copy.deepcopy(self.shapes))
    #         self.frames_shape_groups.append(copy.deepcopy(self.shape_groups))
    #         self.frames_optimize_flag.append(copy.deepcopy(self.optimize_flag))

    #         self.apply_same()

    #         for i in range(self.num_frames):
    #             for j in path in range(self.num_paths):
    #                 self.frames_shapes[i][j].stroke_width = self.ratio

    #     return frames_paths[bg]

    def init_frames_atlas(self):
        frames_paths = self.init_atlas_points()
        for i in range(self.num_paths):
            # stroke_color = torch.tensor([random.random(), random.random(), random.random(), random.random()])
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            points = frames_paths[i]
            points = torch.tensor(points).to(self.device)
            points[:, 0] *= self.canvas_width
            points[:, 1] *= self.canvas_height
    
            path = pydiffvg.Path(num_control_points = self.num_control_points,
                                    points = points,
                                    stroke_width = torch.tensor(self.width),
                                    is_closed = False)
            self.shapes.append(path)

            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                fill_color = None,
                                                stroke_color = stroke_color)
            self.shape_groups.append(path_group)
        
        self.optimize_flag = [True for i in range(len(self.shapes))]

        self.frames_shapes.append(copy.deepcopy(self.shapes))
        self.frames_shape_groups.append(copy.deepcopy(self.shape_groups))
        self.frames_optimize_flag.append(copy.deepcopy(self.optimize_flag))

        self.apply_same()

        for i in range(self.num_frames):
            for j in range(self.num_paths):
                self.frames_shapes[i][j].stroke_width = torch.tensor(self.ratio[i]*self.args.width)

        return frames_paths

    def init_atlas_points(self):
        # frames_paths = []
        paths = []
        points = []
        radius = 0.05
        # for frame in range(len(self.atlas)):
        #     self.strokes_counter = 0
        for i in range(self.num_paths):
            self.num_control_points = torch.zeros(self.num_segments, dtype = torch.int32) + (self.control_points_per_seg - 2)
            # p0 = self.inds_normalised_list_atlas[frame][self.strokes_counter] if self.attention_init else (random.random(), random.random())
            p0 = self.inds_normalised_list_atlas[0][self.strokes_counter] if self.attention_init else (random.random(), random.random())
            points.append(copy.deepcopy(p0))
            for j in range(self.control_points_per_seg - 1):
                # p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p1 = [p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5)]
                points.append(p1)
                p0 = p1
            paths.append(copy.deepcopy(points))
            points = []
            self.strokes_counter += 1
        # frames_paths.append(copy.deepcopy(paths))
        # paths = []
        # return frames_paths
        return paths
    
    def mark_shapes(self):
        points = []
        paths = []
        frames = [] 
        for i in range(len(self.num_frames)):
            for j in range(len(self.num_paths)):
                for k in range(self.control_points_per_seg):
                    if (self.frames_shapes[i][j].points[k][0] > 0 or self.frames_shapes[i][j].points[k][0] < 1) \
                        and (self.frames_shapes[i][j].points[k][1] > 0 or self.frames_shapes[i][j].points[k][0] < 1):
                        points.append(1)
                    else:
                        points.append(0)
                paths.append(points)
                points = []
            frames.append(paths)
            paths = []

        return frames

    def init_frames_all(self):
        indexes = []
        for i in range(self.num_paths):
            # stroke_color = torch.tensor([0.0+(int(i)%3+int(i)/float(self.num_paths))/10., 0.0+(int(i)%3+int(i)/float(self.num_paths))/10., 0.0+(int(i)%3+int(i)/float(self.num_paths))/10., 1.0])
            # stroke_color = torch.tensor([0.0+int(i)/10., 0.0, 0.0, 1.0])
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            # stroke_color = torch.tensor([random.random(), random.random(), random.random(), random.random()])
            # stroke_color = torch.tensor([0.0+int(i)/20., 0.0+int(i)/10., 0.0+int(i)/15., 1.0])
            # print("------------")
            # print(stroke_color)
            
            path, index = self.get_images_path()
            indexes.append(index)
            self.shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                fill_color = None,
                                                stroke_color = stroke_color)
            self.shape_groups.append(path_group)

        self.optimize_flag = [True for i in range(len(self.shapes))]

        # self.frames_shapes.append(self.shapes)
        self.frames_shapes.append(copy.deepcopy(self.shapes))
        self.frames_shape_groups.append(copy.deepcopy(self.shape_groups))
        self.frames_optimize_flag.append(copy.deepcopy(self.optimize_flag))

        self.apply_same()

        for i in range(self.num_frames):
            for j in range(self.num_paths):
                self.frames_shapes[i][j].stroke_width = torch.tensor(self.ratio[i]*self.args.width)

        return indexes
    
    def init_frames_random(self):
        # indexes = []
        for num in range(self.num_frames):
            self.strokes_counter = 0
            self.shape_groups = []
            self.shapes = []
            indexes = []
            for i in range(self.num_paths):
                # stroke_color = torch.tensor([0.0+(int(i)%3+int(i)/float(self.num_paths))/10., 0.0+(int(i)%3+int(i)/float(self.num_paths))/10., 0.0+(int(i)%3+int(i)/float(self.num_paths))/10., 1.0])
                # stroke_color = torch.tensor([0.0+int(i)/10., 0.0, 0.0, 1.0])
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                # stroke_color = torch.tensor([random.random(), random.random(), random.random(), random.random()])
                # stroke_color = torch.tensor([0.0+int(i)/20., 0.0+int(i)/10., 0.0+int(i)/15., 1.0])
                # print("------------")
                # print(stroke_color)
                
                path, index = self.get_inds_path(num)
                indexes.append(index)
                self.shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                    fill_color = None,
                                                    stroke_color = stroke_color)
                self.shape_groups.append(path_group)

            self.optimize_flag = [True for i in range(len(self.shapes))]

            # self.frames_shapes.append(self.shapes)
            self.frames_shapes.append(copy.deepcopy(self.shapes))
            self.frames_shape_groups.append(copy.deepcopy(self.shape_groups))
            self.frames_optimize_flag.append(copy.deepcopy(self.optimize_flag))

        # self.apply_same()

        for i in range(self.num_frames):
            for j in range(self.num_paths):
                self.frames_shapes[i][j].stroke_width = torch.tensor(self.ratio[i]*self.args.width)

        return indexes

    def init_frames_ind(self):
        for f in range(self.num_frames):
            self.strokes_counter = 0
            self.shapes = []
            self.shape_groups = []
            for i in range(self.num_paths):
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                path = self.get_img_path(f)
                self.shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                    fill_color = None,
                                                    stroke_color = stroke_color)
                self.shape_groups.append(path_group)

            self.optimize_flag = [True for i in range(len(self.shapes))]

            # self.frames_shapes.append(self.shapes)
            self.frames_shapes.append(copy.deepcopy(self.shapes))
            self.frames_shape_groups.append(copy.deepcopy(self.shape_groups))
            self.frames_optimize_flag.append(copy.deepcopy(self.optimize_flag))
          
    def set_sparse(self, num):
        index_list = random.choices(range(0, self.num_paths), k=2*num)
        index_list = np.array(index_list).reshape(num, 2)
        area = []
        for i in range(self.num_frames):
            dis_x = 0; dis_y = 0
            for index in index_list:
                dis_x += torch.abs(self.frames_shapes[i][index[0]].points[0][0] - self.frames_shapes[i][index[1]].points[0][0]).cpu().detach()
                dis_y += torch.abs(self.frames_shapes[i][index[0]].points[0][1] - self.frames_shapes[i][index[1]].points[0][1]).cpu().detach()
            area.append((dis_x * dis_y) / num**2)
        
        # area = area / max(area)

        # ratio = [torch.sqrt(x / max(area)) for x in area]
        ratio = [torch.pow((x / max(area)),1/3) for x in area]

        # ic(ratio)

        for i in range(self.num_frames):
            for j in range(self.num_paths):
                self.frames_shapes[i][j].stroke_width = torch.tensor(ratio[i]*self.args.width)

    def get_frames(self):
        
        if self.width_on:
            imgs = self.frames_mlp_render_warp()
        else:
            imgs = self.frames_render_warp()
        
        images = []
        for img in imgs:
            opacity = img[:, :, 3:4]
            img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = self.device) * (1 - opacity)
            img = img[:, :, :3]
            # Convert img from HWC to NCHW
            img = img.unsqueeze(0)
            # print(self.device)
            img = img.permute(0, 3, 1, 2).to(self.device) # NHWC -> NCHW
            images.append(img)
        return images

    def get_image(self):
        img = self.render_warp()
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = self.device) * (1 - opacity)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device) # NHWC -> NCHW
        return img

    def get_path(self):
        points = []
        self.num_control_points = torch.zeros(self.num_segments, dtype = torch.int32) + (self.control_points_per_seg - 2)
        p0 = self.inds_normalised[self.strokes_counter] if self.attention_init else (random.random(), random.random())
        points.append(p0)

        for j in range(self.num_segments):
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height
        
        path = pydiffvg.Path(num_control_points = self.num_control_points,
                                points = points,
                                stroke_width = torch.tensor(self.width),
                                is_closed = False)
        self.strokes_counter += 1
        return path
    
    def get_img_path(self, num):
        points = []
        self.num_control_points = torch.zeros(self.num_segments, dtype = torch.int32) + (self.control_points_per_seg - 2)
        p0 = self.inds_normalised_list[num][self.strokes_counter] if self.attention_init else (random.random(), random.random())
        points.append(p0)

        for j in range(self.num_segments):
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height
        
        path = pydiffvg.Path(num_control_points = self.num_control_points,
                                points = points,
                                stroke_width = torch.tensor(self.width),
                                is_closed = False)
        self.strokes_counter += 1
        return path
    
    def get_addition_path(self):
        points = []
        self.num_control_points = torch.zeros(self.num_segments, dtype = torch.int32) + (self.control_points_per_seg - 2)

        num = np.random.choice(self.args.num_of_frames, replace=True)
        p0 = self.inds_normalised_list_input[num][self.strokes_counter] if self.attention_init else (random.random(), random.random())
        points.append(p0)

        for j in range(self.num_segments):
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height
        
        path = pydiffvg.Path(num_control_points = self.num_control_points,
                                points = points,
                                stroke_width = torch.tensor(self.width),
                                is_closed = False)
        self.strokes_counter += 1
        return path

    def get_inds_path(self,num):
        points = []
        self.num_control_points = torch.zeros(self.num_segments, dtype = torch.int32) + (self.control_points_per_seg - 2)
        # domain = np.append(np.array(range(self.args.num_of_frames)),np.array(self.args.num_of_frames-1).repeat(5))
        # num = self.args.num_of_frames-1
        # num = np.random.choice(domain, replace=True)
        # num = np.random.choice(self.args.num_of_frames, replace=True)
        # self.shapes_frames_index.append(num)
        p0 = self.inds_normalised_list_input[num][self.strokes_counter] if self.attention_init else (random.random(), random.random())
        points.append(p0)

        for j in range(self.num_segments):
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height
        
        path = pydiffvg.Path(num_control_points = self.num_control_points,
                                points = points,
                                stroke_width = torch.tensor(self.width),
                                is_closed = False)
        self.strokes_counter += 1
        return path, num

    def get_images_path(self):
        points = []
        self.num_control_points = torch.zeros(self.num_segments, dtype = torch.int32) + (self.control_points_per_seg - 2)
        # domain = np.append(np.array(range(self.args.num_of_frames)),np.array(self.args.num_of_frames-1).repeat(5))
        # num = self.args.num_of_frames-1
        # num = np.random.choice(domain, replace=True)
        num = np.random.choice(self.args.num_of_frames, replace=True)
        # self.shapes_frames_index.append(num)
        p0 = self.inds_normalised_list_input[num][self.strokes_counter] if self.attention_init else (random.random(), random.random())
        points.append(p0)

        for j in range(self.num_segments):
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height
        
        path = pydiffvg.Path(num_control_points = self.num_control_points,
                                points = points,
                                stroke_width = torch.tensor(self.width),
                                is_closed = False)
        self.strokes_counter += 1
        return path, num

    # def get_images_path(self):
    #     points = []
    #     self.num_control_points = torch.zeros(self.num_segments, dtype = torch.int32) + (self.control_points_per_seg - 2)
    #     # domain = np.append(np.array(range(self.args.num_of_frames)),np.array(self.args.num_of_frames-1).repeat(5))
    #     # num = self.args.num_of_frames-1
    #     # num = np.random.choice(domain, replace=True)
    #     # num = np.random.choice(self.args.num_of_frames, replace=True)
    #     # self.shapes_frames_index.append(num)
    #     # index = self.inds_normalised_input_index[self.strokes_counter]
    #     p0 = self.inds_normalised_input[self.strokes_counter] if self.attention_init else (random.random(), random.random())
    #     index = int(self.inds_normalised_input_index[int(p0[0]*224), int(p0[1]*224)])
    #     points.append(p0)

    #     for j in range(self.num_segments):
    #         radius = 0.05
    #         for k in range(self.control_points_per_seg - 1):
    #             p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
    #             points.append(p1)
    #             p0 = p1
    #     points = torch.tensor(points).to(self.device)
    #     points[:, 0] *= self.canvas_width
    #     points[:, 1] *= self.canvas_height
        
    #     path = pydiffvg.Path(num_control_points = self.num_control_points,
    #                             points = points,
    #                             stroke_width = torch.tensor(self.width),
    #                             is_closed = False)
    #     self.strokes_counter += 1
    #     return path, index
        
    def frames_mlp_render_warp(self):

        # self.widths = []

        # # for i in range(self.num_frames):
        # #     init_param = torch.hstack((self.init_widths,torch.tensor([i]).to(self.device)))
        # #     # widths_  = self.mlp_width(init_param).clamp(min=1e-8)
        # #     widths_  = self.mlp_width(torch.tensor(i/self.num_frames).unsqueeze(0).to(self.device)).clamp(min=1e-8)
        # #     mask_flipped = (1 - widths_).clamp(min=1e-8)
        # #     v = torch.stack((torch.log(widths_), torch.log(mask_flipped)), dim=-1)
        # #     hard_mask = torch.nn.functional.gumbel_softmax(v, 0.2, False)
        # #     self.stroke_probs = hard_mask[:, 0] * torch.ones((self.num_paths)).to(self.device)
        # #     self.widths.append(self.stroke_probs * self.init_widths)
        
        # # self.loss_sparse_con = torch.sum(torch.abs(torch.vstack(self.widths[1:-1]).to(self.device) - torch.vstack(self.widths[0:-2]).to(self.device)) + torch.abs(torch.vstack(self.widths[1:-1]).to(self.device) - torch.vstack(self.widths[2:]).to(self.device)))
        
        f = (torch.tensor(range(self.num_frames))/(self.num_frames-1)).reshape(-1,1).to(self.device)
        # f = torch.tensor([1.]).to(self.device)
        widths_  = self.mlp_width(f).clamp(min=1e-8)
        mask_flipped = (1 - widths_).clamp(min=1e-8)
        v = torch.stack((torch.log(widths_), torch.log(mask_flipped)), dim=-1)
        hard_mask_ = []
        for v_ in v:
            hard_mask_.append(torch.nn.functional.gumbel_softmax(v_, 0.2, False).unsqueeze(0))
        hard_mask = torch.cat(hard_mask_, dim=0)
        # hard_mask = torch.nn.functional.gumbel_softmax(v[:,:,], 0.2, False)
        self.stroke_probs = hard_mask[:,:, 0] * (torch.ones((self.num_paths)).repeat(self.num_frames,1)).to(self.device)
        self.widths = self.stroke_probs * self.init_widths

        self.loss_sparse_con = torch.sum(torch.abs(self.stroke_probs),dim=1)/self.num_paths
        ic(self.loss_sparse_con)
        self.paths4Frames = torch.sum((self.widths>0.5).reshape(self.num_frames,-1), dim=1)
        ic(self.paths4Frames)
        # self.loss_sparse_con = torch.sum(torch.abs((self.widths[1:-1,:] - self.widths[2:,:])) + torch.abs((self.widths[1:-1,:] - self.widths[:-2,:])))
        # ic(self.loss_sparse_con)

        
        # # define new primitives to render
        # for i in range(self.num_frames):
        #     for p, path in enumerate(self.frames_shapes[i]):
        #         width = self.widths[i][p]
        #         path.stroke_width = width

        # imgs = []
        # _render = pydiffvg.RenderFunction.apply
        # for shapes, shape_groups in zip(self.frames_shapes, self.frames_shape_groups):                
        #     scene_args = pydiffvg.RenderFunction.serialize_scene(\
        #         self.canvas_width, self.canvas_height, shapes, shape_groups)
        #     img = _render(self.canvas_width, # width
        #                 self.canvas_height, # height
        #                 2,   # num_samples_x
        #                 2,   # num_samples_y
        #                 0,   # seed
        #                 None,
        #                 *scene_args)
        #     imgs.append(img)

        shapes = []
        shape_groups = []
        imgs = []
        for f in range(self.num_frames):
            for p in range(self.num_paths):
                width = self.widths[f][p]
                path = pydiffvg.Path(
                    num_control_points=self.num_control_points, points=self.frames_shapes[f][p].points,
                    stroke_width=width, is_closed=False)
                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1]),
                    fill_color=None,
                    stroke_color=torch.tensor([0,0,0,1]))
                shape_groups.append(path_group)
            
            _render = pydiffvg.RenderFunction.apply
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                self.canvas_width, self.canvas_height, shapes, shape_groups)
            img = _render(self.canvas_width, # width
                        self.canvas_height, # height
                        2,   # num_samples_x
                        2,   # num_samples_y
                        0,   # seed
                        None,
                        *scene_args)
            imgs.append(img)
        
        return imgs
    # def frames_render_warp(self):
    #     if self.opacity_optim:
    #         for shape_groups in self.frames_shape_groups:
    #             for group in shape_groups:
    #                 group.stroke_color.data[:3].clamp_(0., 0.) # to force black stroke
    #                 group.stroke_color.data[-1].clamp_(0., 1.) # opacity
    #                 # group.stroke_color.data[-1] = (group.stroke_color.data[-1] >= self.color_vars_threshold).float()
    #     # _render = pydiffvg.RenderFunction.apply
    #     # uncomment if you want to add random noise
    #     if self.add_random_noise:
    #         if random.random() > self.noise_thresh:
    #             eps = 0.01 * min(self.canvas_width, self.canvas_height)
    #             for shapes in self.frames_shapes:
    #                 for path in shapes:
    #                     path.points.data.add_(eps * torch.randn_like(path.points))
    #     if self.addition_on:
    #         for shape_groups in self.addition_frames_shape_groups:
    #             for group in shape_groups:
    #                 group.stroke_color.data[:3].clamp_(0., 0.) # to force black stroke
    #                 group.stroke_color.data[-1].clamp_(0., 1.) # opacity
    #                 # group.stroke_color.data[-1] = (group.stroke_color.data[-1] >= self.color_vars_threshold).float()
    #         if self.add_random_noise:
    #             if random.random() > self.noise_thresh:
    #                 eps = 0.01 * min(self.canvas_width, self.canvas_height)
    #                 for shapes in self.addition_frames_shapes:
    #                     for path in shapes:
    #                         path.points.data.add_(eps * torch.randn_like(path.points))

    #     _render = pydiffvg.RenderFunction.apply

    #     imgs = []
    #     # for shapes, shape_groups in zip(self.frames_shapes, self.frames_shape_groups):  
    #     for i in range(self.num_frames): 
    #         render_shapes = self.frames_shapes[i]
    #         render_shape_groups = self.frames_shape_groups[i]
    #         # render_shapes = []
    #         # render_shape_groups = []
    #         if self.addition_on:
    #             render_shapes = render_shapes + self.addition_frames_shapes[i]
    #             render_shape_groups = render_shape_groups + self.addition_frames_shape_groups[i]
    #             # ic(render_shapes[-1].points[1])
    #         scene_args = pydiffvg.RenderFunction.serialize_scene(\
    #             self.canvas_width, self.canvas_height, render_shapes, render_shape_groups)
    
    #         img = _render(self.canvas_width, # width
    #                     self.canvas_height, # height
    #                     2,   # num_samples_x
    #                     2,   # num_samples_y
    #                     0,   # seed
    #                     None,
    #                     *scene_args)

    #         imgs.append(img)
        
    #     return imgs
    
    def frames_render_warp(self):
        if self.opacity_optim:
            for shape_groups in self.frames_shape_groups:
                for group in shape_groups:
                    group.stroke_color.data[:3].clamp_(0., 0.) # to force black stroke
                    group.stroke_color.data[-1].clamp_(0., 1.) # opacity
                    # group.stroke_color.data[-1] = (group.stroke_color.data[-1] >= self.color_vars_threshold).float()
        _render = pydiffvg.RenderFunction.apply
        # uncomment if you want to add random noise
        if self.add_random_noise:
            if random.random() > self.noise_thresh:
                eps = 0.01 * min(self.canvas_width, self.canvas_height)
                for shapes in self.frames_shapes:
                    for path in shapes:
                        path.points.data.add_(eps * torch.randn_like(path.points))
        imgs = []
        for shapes, shape_groups in zip(self.frames_shapes, self.frames_shape_groups):                
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                self.canvas_width, self.canvas_height, shapes, shape_groups)
            img = _render(self.canvas_width, # width
                        self.canvas_height, # height
                        2,   # num_samples_x
                        2,   # num_samples_y
                        0,   # seed
                        None,
                        *scene_args)
            imgs.append(img)
        
        return imgs
    
    def addition_render_warp(self):
        if self.opacity_optim:
            for shape_groups in self.addition_frames_shape_groups:
                for group in shape_groups:
                    group.stroke_color.data[:3].clamp_(0., 0.) # to force black stroke
                    group.stroke_color.data[-1].clamp_(0., 1.) # opacity
                    # group.stroke_color.data[-1] = (group.stroke_color.data[-1] >= self.color_vars_threshold).float()
        _render = pydiffvg.RenderFunction.apply
        # uncomment if you want to add random noise
        if self.add_random_noise:
            if random.random() > self.noise_thresh:
                eps = 0.01 * min(self.canvas_width, self.canvas_height)
                for shapes in self.addition_frames_shapes:
                    for path in shapes:
                        path.points.data.add_(eps * torch.randn_like(path.points))
        imgs = []
        for shapes, shape_groups in zip(self.addition_frames_shapes, self.addition_frames_shape_groups):                
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                self.canvas_width, self.canvas_height, shapes, shape_groups)
            img = _render(self.canvas_width, # width
                        self.canvas_height, # height
                        2,   # num_samples_x
                        2,   # num_samples_y
                        0,   # seed
                        None,
                        *scene_args)
            imgs.append(img)
        
        return imgs

    def render_warp(self):
        if self.opacity_optim:
            for group in self.shape_groups:
                group.stroke_color.data[:3].clamp_(0., 0.) # to force black stroke
                group.stroke_color.data[-1].clamp_(0., 1.) # opacity
                # group.stroke_color.data[-1] = (group.stroke_color.data[-1] >= self.color_vars_threshold).float()
        _render = pydiffvg.RenderFunction.apply
        # uncomment if you want to add random noise
        if self.add_random_noise:
            if random.random() > self.noise_thresh:
                eps = 0.01 * min(self.canvas_width, self.canvas_height)
                for path in self.shapes:
                    path.points.data.add_(eps * torch.randn_like(path.points))
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
        img = _render(self.canvas_width, # width
                    self.canvas_height, # height
                    2,   # num_samples_x
                    2,   # num_samples_y
                    0,   # seed
                    None,
                    *scene_args)
        return img
        
    def frames_shapes(self):
        return self.frames_shapes

    def shapes_frames_index(self):
        return self.shapes_frames_index
    
    def frames_shape_groups(self):
        return self.frames_shape_groups

    def width_parameters(self):
        return self.mlp_width.parameters()
    
    def loss_sparse(self):
        return self.loss_sparse_con
   
    def turn_on_addition(self):
        self.addition_on = 1
    
    def width_is_on(self):
        return self.width_on
    def frames_paths(self):
        return self.paths4Frames

    def addition_is_on(self):
        return self.addition_on
   
    def turn_off_main(self):
        self.main_on = 0
        ic(self.num_paths)
        for i in range(self.num_frames):
            for j in range(self.num_paths):
                self.frames_optimize_flag[i][j] = False
                self.frames_shapes[i][j].points.requires_grad = False
   
    def parameters(self):
        ic(self.num_paths)
        ic(len(self.frames_shapes[0]))
        self.points_vars = []
        # storkes' location optimization
        for i, shapes in enumerate(self.frames_shapes):
            for j, path in enumerate(shapes):
                if self.frames_optimize_flag[i][j]:
                    path.points.requires_grad = True
                    self.points_vars.append(path.points)
        return self.points_vars

    def get_points_parans(self):
        return self.points_vars
    
    def get_width_params(self):
        return self.width_vars
    
    def set_color_parameters(self):
        # for storkes' color optimization (opacity)
        self.color_vars = []
        for i, shape_groups in enumerate(self.frames_shape_groups):
            for j, group in enumerate(shape_groups):
                if self.frames_optimize_flag[i][j]:
                    group.stroke_color.requires_grad = True
                    self.color_vars.append(group.stroke_color)
        return self.color_vars

    def get_color_parameters(self):
        return self.color_vars
        
    # def save_svg(self, output_dir, name):
    #     pydiffvg.save_svg('{}/{}.svg'.format(output_dir, name), self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
    
    # def save_svgs(self, output_dir, name):
    #    for i in range(self.num_frames):
    #         pydiffvg.save_svg('{}/{}_{}.svg'.format(output_dir, name, i), self.canvas_width, self.canvas_height, self.frames_shapes[i], self.frames_shape_groups[i])

    def save_svg(self, output_dir, name):
        if not self.width_on:
            pydiffvg.save_svg('{}/{}.svg'.format(output_dir, name), self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
        else:
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            new_shapes, new_shape_groups = [], []
            for path in self.shapes:
                w = path.stroke_width / 1.5
                if w > 0.7:
                    new_shapes.append(path)
                    path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(new_shapes) - 1]),
                                                fill_color = None,
                                                stroke_color = stroke_color)
                    new_shape_groups.append(path_group)
            pydiffvg.save_svg('{}/{}.svg'.format(output_dir, name), self.canvas_width, self.canvas_height, new_shapes, new_shape_groups)
    
    def save_svgs(self, output_dir, name):
        if not self.width_on:
            for i in range(self.num_frames):
                pydiffvg.save_svg('{}/{}_{}.svg'.format(output_dir, name, i), self.canvas_width, self.canvas_height, self.frames_shapes[i], self.frames_shape_groups[i])
        else:
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            for i in range(self.num_frames):
                new_shapes, new_shape_groups = [], []
                for path in self.frames_shapes[i]:
                    w = path.stroke_width / 1.5
                    if w > 0.7:
                        new_shapes.append(path)
                        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(new_shapes) - 1]),
                                                    fill_color = None,
                                                    stroke_color = stroke_color)
                        new_shape_groups.append(path_group)
                pydiffvg.save_svg('{}/{}_{}.svg'.format(output_dir, name, i), self.canvas_width, self.canvas_height, new_shapes, new_shape_groups)


    def dino_attn(self):
        patch_size=8 # dino hyperparameter
        threshold=0.6

        # for dino model
        mean_imagenet = torch.Tensor([0.485, 0.456, 0.406])[None,:,None,None].to(self.device)
        std_imagenet = torch.Tensor([0.229, 0.224, 0.225])[None,:,None,None].to(self.device)
        totens = transforms.Compose([
            transforms.Resize((self.canvas_height, self.canvas_width)),
            transforms.ToTensor()
            ])

        dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').eval().to(self.device)
        
        self.main_im = Image.open(self.target_path).convert("RGB")
        main_im_tensor = totens(self.main_im).to(self.device)
        img = (main_im_tensor.unsqueeze(0) - mean_imagenet) / std_imagenet
        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size
        
        with torch.no_grad():
            attn = dino_model.get_last_selfattention(img).detach().cpu()[0]

        nh = attn.shape[0]
        attn = attn[:,0,1:].reshape(nh,-1)
        val, idx = torch.sort(attn)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()
        
        attn = attn.reshape(nh, w_featmap, h_featmap).float()
        attn = nn.functional.interpolate(attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()
        
        return attn


    # def define_attention_input(self, target_im):
    #     model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
    #     model.eval().to(self.device)
    #     data_transforms = transforms.Compose([
    #                 preprocess.transforms[-1],
    #             ])
    #     self.image_input_attn_clip = data_transforms(target_im).to(self.device)

    # def define_attention_input(self, target_im):
    #     model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
    #     model.eval().to(self.device)
    #     data_transforms = transforms.Compose([
    #                 preprocess.transforms[-1],
    #             ])
    #     self.image_input_attn_clip = []
    #     for img in target_im:
    #         self.image_input_attn_clip.append(data_transforms(img).to(self.device))
        
    def define_attention_input(self, target_im):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        data_transforms = transforms.Compose([
                    preprocess.transforms[-1],
                ])
        images_attn_clip = []
        for img in target_im:
            images_attn_clip.append(data_transforms(img).to(self.device))
        return images_attn_clip    

    # def clip_attn(self):
    #     model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
    #     model.eval().to(self.device)
    #     text_input = clip.tokenize([self.text_target]).to(self.device)

    #     if "RN" in self.saliency_clip_model:
    #         saliency_layer = "layer4"
    #         attn_map = gradCAM(
    #             model.visual,
    #             self.image_input_attn_clip,
    #             model.encode_text(text_input).float(),
    #             getattr(model.visual, saliency_layer)
    #         )
    #         attn_map = attn_map.squeeze().detach().cpu().numpy()
    #         attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

    #     else:
    #         # attn_map = interpret(self.image_input_attn_clip, text_input, model, device=self.device, index=0).astype(np.float32)
    #         attn_map = interpret(self.image_input_attn_clip, text_input, model, device=self.device)
            
    #     del model
    #     return attn_map

    def clip_attn(self, image_input_attn_clip):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        text_input = clip.tokenize([self.text_target]).to(self.device)

        if "RN" in self.saliency_clip_model:
            saliency_layer = "layer4"
            attn_map = gradCAM(
                model.visual,
                image_input_attn_clip,
                model.encode_text(text_input).float(),
                getattr(model.visual, saliency_layer)
            )
            attn_map = attn_map.squeeze().detach().cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

        else:
            # attn_map = interpret(self.image_input_attn_clip, text_input, model, device=self.device, index=0).astype(np.float32)
            attn_map = interpret(image_input_attn_clip, text_input, model, device=self.device)
            
        del model
        return attn_map

    def set_attention_map(self):
        assert self.saliency_model in ["dino", "clip"]
        if self.saliency_model == "dino":
            return self.dino_attn()
        elif self.saliency_model == "clip":
            return self.clip_attn()
        

    def softmax(self, x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum() 
    
    def avg(self, x):
        return 1 / len(x)

    # def set_inds_clip(self):
    #     attn_map = (self.attention_map - self.attention_map.min()) / (self.attention_map.max() - self.attention_map.min())
    #     if self.xdog_intersec:
    #         xdog = XDoG_()
    #         im_xdog = xdog(self.image_input_attn_clip[0].permute(1,2,0).cpu().numpy(), k=10)
    #         intersec_map = (1 - im_xdog) * attn_map
    #         attn_map = intersec_map
            
    #     attn_map_soft = np.copy(attn_map)
    #     attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], tau=self.softmax_temp)
        
    #     k = self.num_stages * self.num_paths
    #     self.inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=False, p=attn_map_soft.flatten())
    #     self.inds = np.array(np.unravel_index(self.inds, attn_map.shape)).T
    
    #     self.inds_normalised = np.zeros(self.inds.shape)
    #     self.inds_normalised[:, 0] =  self.inds[:, 1] / self.canvas_width
    #     self.inds_normalised[:, 1] =  self.inds[:, 0] / self.canvas_height
    #     self.inds_normalised = self.inds_normalised.tolist()
    #     return attn_map_soft

    # def set_inds_clip(self):
    #     # self.inds_list = []
    #     self.inds_normalised_list = []
    #     attn_maps_soft = []
    #     for i, attention in enumerate(self.attention_map):
    #         attn_map = (attention - attention.min()) / (attention.max() - attention.min())
    #         if self.xdog_intersec:
    #             xdog = XDoG_()
    #             im_xdog = xdog(self.image_input_attn_clip[i][0].permute(1,2,0).cpu().numpy(), k=10)
    #             intersec_map = (1 - im_xdog) * attn_map
    #             attn_map = intersec_map
                
    #         attn_map_soft = np.copy(attn_map)
    #         attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], tau=self.softmax_temp)
    #         attn_maps_soft.append(attn_map_soft)

    #         k = self.num_stages * self.num_paths
    #         self.inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=False, p=attn_map_soft.flatten())
    #         self.inds = np.array(np.unravel_index(self.inds, attn_map.shape)).T
    #         # self.inds_list.append(copy.deepcopy(self.inds))
        
    #         self.inds_normalised = np.zeros(self.inds.shape)
    #         self.inds_normalised[:, 0] =  self.inds[:, 1] / self.canvas_width
    #         self.inds_normalised[:, 1] =  self.inds[:, 0] / self.canvas_height
    #         self.inds_normalised = self.inds_normalised.tolist()
    #         self.inds_normalised_list.append(copy.deepcopy(self.inds_normalised))
    #     return attn_maps_soft

# plt.figure(figsize=(10, 5))
#         threshold_map = self.attention_atlas_map[0]
#         threshold_map_ = (threshold_map - threshold_map.min()) / \
#         (threshold_map.max() - threshold_map.min())
#         plt.imshow(threshold_map_, interpolation='nearest', vmin=0, vmax=1)
#         plt.title("attnetion_atlas")
#         # plt.scatter(self.inds[:, 1], self.inds[:, 0], s=10, c='red', marker='o')
#         # plt.axis("off")
#         plt.savefig(self.args.output_dir)
#         plt.close()

    def set_inds_clip(self, attention_map, mode):
        # self.inds_list = []
        inds_normalised_list = []
        attn_maps_soft = []
        # attention_map 
        for i, attention in enumerate(attention_map):
            attn_map = (attention - attention.min()) / (attention.max() - attention.min())
            if self.xdog_intersec:
                xdog = XDoG_()
                if mode == "atlas":
                    im_xdog = xdog(self.image_atlas_attn_clip[i][0].permute(1,2,0).cpu().numpy(), k=10)
                elif mode == "input":
                    im_xdog = xdog(self.image_input_attn_clip[i][0].permute(1,2,0).cpu().numpy(), k=10)
                intersec_map = (1 - im_xdog) * attn_map
                attn_map = intersec_map
                
            attn_map_soft = np.copy(attn_map)
            attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], tau=self.softmax_temp)
            attn_maps_soft.append(attn_map_soft)

            k = self.num_stages * self.num_paths
            # ic(sum(attn_map_soft.flatten()>0))
            # ic(self.args.output_dir)
            # if not os.path.exists(self.args.output_dir):
            #     os.mkdir(self.args.output_dir)
            if sum(attn_map_soft.flatten()>0) < 100:
                inds_normalised_list.append(copy.deepcopy(self.inds_normalised))
                continue


            # plt.figure(figsize=(10, 5))
            # threshold_map = attn_map_soft
            # threshold_map_ = (threshold_map - threshold_map.min()) / \
            # (threshold_map.max() - threshold_map.min())
            # plt.imshow(threshold_map_, interpolation='nearest', vmin=0, vmax=1)
            # plt.savefig(self.args.output_dir)
            # plt.close()

            self.inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=False, p=attn_map_soft.flatten())
            self.inds = np.array(np.unravel_index(self.inds, attn_map.shape)).T
            # self.inds_list.append(copy.deepcopy(self.inds))

            # plt.figure(figsize=(10, 5))
            # threshold_map = attn_map_soft
            # threshold_map_ = (threshold_map - threshold_map.min()) / \
            # (threshold_map.max() - threshold_map.min())
            # plt.imshow(threshold_map_, interpolation='nearest', vmin=0, vmax=1)
            # plt.title("newest")
            # plt.scatter(self.inds[:, 1], self.inds[:, 0], s=10, c='red', marker='o')
            # plt.axis("off")
            # plt.savefig(self.args.output_dir)
            # plt.close()

        
            self.inds_normalised = np.zeros(self.inds.shape)
            self.inds_normalised[:, 0] =  self.inds[:, 1] / self.canvas_width
            self.inds_normalised[:, 1] =  self.inds[:, 0] / self.canvas_height
            self.inds_normalised = self.inds_normalised.tolist()
            inds_normalised_list.append(copy.deepcopy(self.inds_normalised))
        return attn_maps_soft, inds_normalised_list
    
    def set_random_init(self):
        inds_normalised_list = []
        for i in range(self.args.num_of_frames):
            inds_normalised = np.zeros(self.inds.shape)
            inds_normalised[:,0] = np.random.randint(0, 224, self.num_paths) / self.canvas_width
            inds_normalised[:,1] = np.random.randint(0, 224, self.num_paths) / self.canvas_height
            inds_normalised = inds_normalised.tolist()
            inds_normalised_list.append(copy.deepcopy(inds_normalised))

        return inds_normalised_list

    # def set_inds_clip(self, attention_map, mode):
    #     # self.inds_list = []
    #     inds_normalised_list = []
    #     attn_maps_soft = []
    #     index_map = np.zeros((attention_map[0].shape))
    #     # global_map = np.zeros((attention_map[0].shape))

    #     if mode == "atlas":
    #         global_map = np.ones((attention_map[0].shape))
    #         attention_map = attention_map[0]
    #         attn_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    #         if self.xdog_intersec:
    #             xdog = XDoG_()
    #             im_xdog = xdog(self.image_atlas_attn_clip[0][0].permute(1,2,0).cpu().numpy(), k=10)
    #             intersec_map = (1 - im_xdog) * attn_map
    #             global_map = intersec_map
        
    #     if mode == "input":
    #         # index_map = np.zeros((attention_map[0].shape))
    #         global_map = np.zeros((attention_map[0].shape))
    #         ic(len(attention_map))
    #         for i, attention in enumerate(attention_map):
    #             attn_map = (attention - attention.min()) / (attention.max() - attention.min())
    #             if self.xdog_intersec:
    #                 xdog = XDoG_()
    #                 im_xdog = xdog(self.image_input_attn_clip[i][0].permute(1,2,0).cpu().numpy(), k=10)
    #                 intersec_map = (1 - im_xdog) * attn_map
    #                 global_map = np.maximum(intersec_map, global_map)
    #                 is_frames_larger = global_map == intersec_map
    #                 index_map = is_frames_larger * i + index_map * (1-is_frames_larger)
    #                 # if i == 51:
    #                 #     print("1")
    #                 #     break

    #     # attn_map_soft = np.copy(intersec_map)
    #     # # attn_map_soft = np.copy(attn_map)
    #     # attn_map_soft[intersec_map > 0] = self.softmax(intersec_map[intersec_map > 0], tau=self.softmax_temp)
    #     attn_map_soft = np.copy(global_map)
    #     # attn_map_soft = np.copy(attn_map)
    #     attn_map_soft[global_map > 0] = self.softmax(global_map[global_map > 0], tau=self.softmax_temp)
    #     # attn_map_soft[global_map > 0] = self.avg(global_map[global_map > 0])
    #     # self.thresh = attn_map_soft

    #     k = self.num_stages * self.num_paths 
    #     self.inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=False, p=attn_map_soft.flatten())
    #     self.inds = np.array(np.unravel_index(self.inds, attn_map.shape)).T
    
    #     inds_normalised = np.zeros(self.inds.shape)
    #     inds_normalised[:, 0] =  self.inds[:, 1] / self.canvas_width
    #     inds_normalised[:, 1] =  self.inds[:, 0] / self.canvas_height
    #     inds_normalised = inds_normalised.tolist()
        
    #     return attn_map_soft, index_map, inds_normalised

    def set_inds_dino(self):
        k = max(3, (self.num_stages * self.num_paths) // 6 + 1) # sample top 3 three points from each attention head
        num_heads = self.attention_map.shape[0]
        self.inds = np.zeros((k * num_heads, 2))
        # "thresh" is used for visualisaiton purposes only
        thresh = torch.zeros(num_heads + 1, self.attention_map.shape[1], self.attention_map.shape[2])
        softmax = nn.Softmax(dim=1)
        for i in range(num_heads):
            # replace "self.attention_map[i]" with "self.attention_map" to get the highest values among
            # all heads. 
            topk, indices = np.unique(self.attention_map[i].numpy(), return_index=True)
            topk = topk[::-1][:k]
            cur_attn_map = self.attention_map[i].numpy()
            # prob function for uniform sampling
            prob = cur_attn_map.flatten()
            prob[prob > topk[-1]] = 1
            prob[prob <= topk[-1]] = 0
            prob = prob / prob.sum()
            thresh[i] = torch.Tensor(prob.reshape(cur_attn_map.shape))

            # choose k pixels from each head            
            inds = np.random.choice(range(cur_attn_map.flatten().shape[0]), size=k, replace=False, p=prob)
            inds = np.unravel_index(inds, cur_attn_map.shape)
            self.inds[i * k: i * k + k, 0] = inds[0]
            self.inds[i * k: i * k + k, 1] = inds[1]
        
        # for visualisaiton
        sum_attn = self.attention_map.sum(0).numpy()
        mask = np.zeros(sum_attn.shape)
        mask[thresh[:-1].sum(0) > 0] = 1
        sum_attn = sum_attn * mask
        sum_attn = sum_attn / sum_attn.sum()
        thresh[-1] = torch.Tensor(sum_attn)

        # sample num_paths from the chosen pixels.
        prob_sum = sum_attn[self.inds[:,0].astype(np.int), self.inds[:,1].astype(np.int)]
        prob_sum = prob_sum / prob_sum.sum()
        new_inds = []
        for i in range(self.num_stages):
            new_inds.extend(np.random.choice(range(self.inds.shape[0]), size=self.num_paths, replace=False, p=prob_sum))
        self.inds = self.inds[new_inds]
        print("self.inds",self.inds.shape)
    
        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] =  self.inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] =  self.inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()
        return thresh

    def set_attention_threshold_map(self):
        assert self.saliency_model in ["dino", "clip"]
        if self.saliency_model == "dino":
            return self.set_inds_dino()
        elif self.saliency_model == "clip":
            return self.set_inds_clip()
        

    def get_attn(self):
        return self.attention_map
    
    def get_thresh(self):
        return self.thresh

    def get_inds(self):
        return self.inds
    
    def get_mask(self):
        return self.mask

    def set_random_noise(self, epoch):
        if epoch % self.args.save_interval == 0:
            self.add_random_noise = False
        else:
            self.add_random_noise = "noise" in self.args.augemntations

class PainterOptimizer:
    def __init__(self, args, renderer):
        self.renderer = renderer
        self.points_lr = args.lr
        self.color_lr = args.color_lr
        self.args = args
        self.optim_color = args.force_sparse
        
        self.width_on = 0
        self.width_lr = args.width_lr
        self.width_optimizer = None
        self.mlp_width_weights_path = args.mlp_width_weights_path

    def init_point_lr(self, lr):
        self.points_lr = lr
        
    def init_width_lr(self, lr):
        self.width_lr = lr
        
    def turn_on_width(self):
        self.width_on = 1

    # def init_optimizers(self):
    #     self.points_optim = torchself.optim.Adam(self.renderer.parameters(), lr=self.points_lr)
    #     if self.optim_color:
    #         self.color_optim = torch.optim.Adam(self.renderer.set_color_parameters(), lr=self.color_lr)

    def init_optimizers(self):
        self.points_optim = torch.optim.Adam(self.renderer.parameters(), lr=self.points_lr)
        if self.optim_color:
            self.color_optim = torch.optim.Adam(self.renderer.set_color_parameters(), lr=self.color_lr)
    
    def init_width_optimizers(self):
        self.width_optim = torch.optim.Adam(self.renderer.width_parameters(), lr=self.width_lr)
        if self.mlp_width_weights_path != "none":
            checkpoint = torch.load(self.mlp_width_weights_path)
            self.width_optim.load_state_dict(checkpoint['optimizer_state_dict'])
            print("optimizer checkpoint loaded from ", self.mlp_width_weights_path)

    # def update_lr(self, counter):
    #     new_lr = utils.get_epoch_lr(counter, self.args)
    #     for param_group in self.points_optim.param_groups:
    #         param_group["lr"] = new_lr
    
    def zero_grad_(self):
        self.points_optim.zero_grad()
        if self.optim_color:
            self.color_optim.zero_grad()
        if self.width_on:    
            self.width_optim.zero_grad()
    
    def step_(self):
        self.points_optim.step()
        if self.optim_color:
            self.color_optim.step()
        if self.width_on:    
            self.width_optim.step()
    
    def get_lr(self):
        return self.points_optim.param_groups[0]['lr']
        
    def get_width_optim(self):
        return self.width_optimizer


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad




# def interpret(image, texts, model, device):

#     images = image.repeat(1, 1, 1, 1)
#     res = model.encode_image(images)
#     model.zero_grad()
#     image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
#     num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
#     R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
#     R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
#     cams = [] # there are 12 attention blocks
#     for i, blk in enumerate(image_attn_blocks):
#         cam = blk.attn_probs.detach() #attn_probs shape is 12, 50, 50
#         # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
#         cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
#         cam = cam.clamp(min=0)
#         cam = cam.clamp(min=0).mean(dim=1) # mean of the 12 something
#         cams.append(cam)  
#         R = R + torch.bmm(cam, R)
              
#     cams_avg = torch.cat(cams) # 12, 50, 50
#     cams_avg = cams_avg[:, 0, 1:] # 12, 1, 49
#     image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
#     image_relevance = image_relevance.reshape(1, 1, 7, 7)
#     image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
#     image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
#     image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
#     return image_relevance

def interpret(video, texts, model, device):
    images_relevance = []
    for frame in video:
        images = frame.repeat(1, 1, 1, 1)
        res = model.encode_image(images)
        model.zero_grad()
        image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
        R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
        cams = [] # there are 12 attention blocks
        for i, blk in enumerate(image_attn_blocks):
            cam = blk.attn_probs.detach() #attn_probs shape is 12, 50, 50
            # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
            cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0)
            cam = cam.clamp(min=0).mean(dim=1) # mean of the 12 something
            cams.append(cam)  
            R = R + torch.bmm(cam, R)
                
        cams_avg = torch.cat(cams) # 12, 50, 50
        cams_avg = cams_avg[:, 0, 1:] # 12, 1, 49
        image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
        image_relevance = image_relevance.reshape(1, 1, 7, 7)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
        image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
        images_relevance.append(image_relevance)
    return images_relevance


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()
        
    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
        
    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:        
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()
    
        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)
    
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
        
    return gradcam    


class XDoG_(object):
    def __init__(self):
        super(XDoG_, self).__init__()
        self.gamma=0.98
        self.phi=200
        self.eps=-0.1
        self.sigma=0.8
        self.binarize=True
        
    def __call__(self, im, k=10):
        if im.shape[2] == 3:
            im = rgb2gray(im)
        imf1 = gaussian_filter(im, self.sigma)
        imf2 = gaussian_filter(im, self.sigma * k)
        imdiff = imf1 - self.gamma * imf2
        imdiff = (imdiff < self.eps) * 1.0  + (imdiff >= self.eps) * (1.0 + np.tanh(self.phi * imdiff))
        imdiff -= imdiff.min()
        imdiff /= imdiff.max()
        if self.binarize:
            th = threshold_otsu(imdiff)
            imdiff = imdiff >= th
        imdiff = imdiff.astype('float32')
        return imdiff
