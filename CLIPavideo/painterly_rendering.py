import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import argparse
import math
import os
import sys
import time
import traceback

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from torchvision import models, transforms
from tqdm.auto import tqdm, trange

import config
import sketch_utils as utils
from models.loss import Loss
from models.painter_params import Painter, PainterOptimizer
from IPython.display import display, SVG

from MLP_models.MLP import IMLP
import cv2
from pathlib import Path
import random
import loss_addition
import logging

from icecream import ic
from torch import nn
from natsort import natsorted

def load_renderer(args, target_im=None, ratio=None, atlas=None, optic_flow=None, mask=None):
    renderer = Painter(num_strokes=args.num_paths, args=args,
                       num_segments=args.num_segments,
                       imsize=args.image_scale,
                       device=args.device,
                       target_im=target_im,
                       ratio=ratio,
                       atlas = atlas,
                       optic_flow = optic_flow,
                       mask=mask)
    renderer = renderer.to(args.device)
    return renderer


def get_frames(args):
    frames = []
    im_list = sorted(os.listdir(args.frames_dir))[:args.num_of_frames]
    # print(im_list)
    img_size = (224, 224)
    for i in im_list:
        im_name = os.path.join(args.frames_dir+i)
        # frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        frame = cv2.imread(im_name) / 255
        frame = cv2.resize(frame, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        frame = np.transpose(frame, (2,0,1))
        frames.append(torch.from_numpy(frame).unsqueeze(0).to(args.device))
        # frames.append(frame)
    # frames = np.transpose(np.stack(frames), (0,3,1,2))
    # return torch.from_numpy(frames).to(args.device)
    return frames

def get_masks(args):
    masks = []
    im_list = natsorted(os.listdir(args.masks_dir))[:args.num_of_frames]
    # print(im_list)
    img_size = (224, 224)
    for i in im_list:
        im_name = os.path.join(args.masks_dir+i)
        # frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        mask = cv2.imread(im_name) / 255
        mask = cv2.resize(mask, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        mask = (mask[:,:,0]>0).astype(np.uint8)
        # mask = np.transpose(mask, (2,0,1))
        # mask = mask(mask[:,:,0]>0).astype(np.uint8)
        
        masks.append(torch.from_numpy(mask).unsqueeze(0).to(args.device))

    return masks

def resize_flow(flow, newh, neww):
    oldh, oldw = flow.shape[0:2]
    flow[:, :, 0] *= newh / oldh
    flow[:, :, 1] *= neww / oldw
    flow = cv2.resize(flow, (neww, newh), interpolation=cv2.INTER_LINEAR)
    return flow


def get_optic_flow(args):
    # out_flow_dir = vid_root / f'{vid_name}_flow'
    # frames = []
    # im_list = sorted(os.listdir(args.frames_dir))
    out_flow_dir = Path(args.flow_dir)
    data_folder = Path(args.frames_dir)
    input_files = sorted(os.listdir(args.frames_dir))
    
    optical_flows = np.zeros((224, 224, 2, args.num_of_frames, 2))

    for i in range(args.num_of_frames - 1):
        file1 = input_files[i]
        j = i + 1
        file2 = input_files[j]
        # ic(j)

        fn1 = str(file1)
        fn2 = str(file2)

        flow12_fn = out_flow_dir / f'{fn1}_{fn2}.npy'
        flow21_fn = out_flow_dir / f'{fn2}_{fn1}.npy'
        flow12 = np.load(flow12_fn)
        flow21 = np.load(flow21_fn)

        if flow12.shape[0] != 224 or flow12.shape[1] != 224:
            flow12 = resize_flow(flow12, newh=224, neww=224)
            flow21 = resize_flow(flow21, newh=224, neww=224)

        optical_flows[:, :, :, i, 0] = flow12
        optical_flows[:, :, :, i, 1] = flow21
    
    return optical_flows

def take_key_frames(frames, interval_num):
    key_frames = []
    index = []
    for i, frame in enumerate(frames):
        if i%interval_num == 0:
            key_frames.append(frame)
            index.append(i)
    if (len(frames) - 1)%interval_num != 0:
        key_frames.append(frames[-1])
        index.append(len(frames) - 1)
        ic(len(frames) - 1)
    
    return key_frames, index

def get_atlas(args):
    atlas = []
    # atlas_bg = Image.open(args.atlas_bg_dir)
    atlas_fore = Image.open(args.atlas_fore_dir)
    ic(args.atlas_fore_dir)
    if atlas_fore.mode == "RGBA":
        # Create a white rgba background
        new_fore = Image.new("RGBA", atlas_fore.size, "WHITE")
        # new_bg = Image.new("RGBA", atlas_bg.size, "WHITE")
        # Paste the image on the background.
        new_fore.paste(atlas_fore, (0, 0), atlas_fore)
        # new_bg.paste(atlas_bg, (0, 0), atlas_bg)
        atlas_fore = new_fore
        # atlas_bg = new_bg
    atlas_fore = atlas_fore.convert("RGB")
    # atlas_bg = atlas_bg.convert("RGB")

    transforms_ = []
    if atlas_fore.size[0] != atlas_fore.size[1]:
        transforms_.append(transforms.Resize(
            (args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(
            args.image_scale, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    atlas_fore_ = data_transforms(atlas_fore).unsqueeze(0).to(args.device)
    # atlas_bg_ = data_transforms(atlas_bg).unsqueeze(0).to(args.device)
    # if (atlas_fore_[:, :, :, :] < 5).any(dim=1):
    #     atlas_fore_[:, :, :, :] = 255
    mask = (atlas_fore_.mean(dim=1, keepdim=True) < 0.01)
    # atlas_fore_ = atlas_fore_ * (1-mask) + 1 * mask
    atlas_fore_[mask.expand_as(atlas_fore_)] = 1
    # # atlas_fore_

    atlas.append(atlas_fore_)
    # atlas.append(atlas_bg_)
    return atlas

def main(args):
    if args.text == "None":
        args.text = None
        args.clip_text_guide = 1
    else:
        args.clip_text_guide = 0
    ic(args.text)
    ic(args.clip_conv_layer_weights)
    ic(type(args.clip_conv_layer_weights))
    ic(args.clip_conv_layer_weights[0])
    # ic(args.text_layer_weights)
    ic(args.frames_dir)
    ic(args.checkpoint_path)
    ic(args.consist_param)
    ic(args.num_paths)
    ic(args.clip_model_name)
    # ic(args.text_layer_weights)
    ic(args.clip_text_weight)
    ic(args.clip_fc_loss_weight)
    ic(args.width)
    
    visual_point = random.choices(range(0, args.num_paths), k=15)
    colors = []
    for _ in range(len(visual_point)):
        color = [random.randrange(0, 256) for _ in range(3)]
        # color = [255,0,0]
        colors.append(color)

    loss_func = Loss(args)
    # inputs = torch.vstack(get_frames(args))
    inputs = get_frames(args)
    atlas = get_atlas(args)
    masks = get_masks(args)
    max_mask = torch.max(torch.sum((torch.vstack(inputs)<1).reshape(args.num_of_frames,-1), dim=1))
    mask_ratio = torch.sqrt(torch.sum((torch.vstack(inputs)<1).reshape(args.num_of_frames,-1),dim=1) / max_mask)
    mask_ratio = torch.ones((mask_ratio.shape))

    # args.interval_num = 10
    # key_frames, index = take_key_frames(inputs, args.interval_num)
    # optic_flow = get_optic_flow(args)
    
    # logging.basicConfig(filename='log.txt', filemode='w', level=logging.INFO)
    logging.basicConfig(format='%(message)s', filename=f'{args.output_dir}/log.txt', filemode='w', level=logging.INFO)
    logging.info(f'Consist_param: {args.consist_param}')
    logging.info(f'Frames_param: {args.frames_param}')
    logging.info(f'CLIP_param: {args.clip_param}')
    logging.info(f'Num_of_stroks: {args.num_paths}')

    utils.log_input(inputs[0], args.output_dir)
    # print(inputs.shape)
    # renderer = load_renderer(args, inputs, atlas, optic_flow=None)
    renderer = load_renderer(args, inputs, mask_ratio, atlas, optic_flow=None)
    # renderer = load_renderer(args, inputs, optic_flow)
    # renderer = load_renderer(args, inputs)

    optimizer = PainterOptimizer(args, renderer)
    counter = 0
    configs_to_save = {"loss_eval": []}
    best_loss, best_fc_loss = 100, 100
    best_iter, best_iter_fc = 0, 0
    min_delta = 1e-5
    terminate = False

    # renderer.set_random_noise(0)
    # img = renderer.init_image(stage=0)
    # renderer.init_frames()
    # renderer.init_frames_flow()
    
    if args.focus == "foreground":
        index = renderer.init_frames_all()
        # index = renderer.init_frames_random()
        atlas_paths=None

    if args.focus == "background" or args.focus == 'atlas':
        index = None
        atlas_paths = renderer.init_frames_atlas()

    optimizer.init_optimizers()

    # print("--------")
    print(args.device)

    init_file = torch.load(args.checkpoint_path)
    ic(args.checkpoint_path)

    if args.focus == "foreground" or args.focus == 'atlas':
        ic(args.focus)
        model_F_mapping = IMLP(
        input_dim=3,
        output_dim=2,
        hidden_dim=256,
        use_positional=False,
        positional_dim=4,
        num_layers=6,
        skip_layers=[]).to(args.device)
        
        model_F_mapping.load_state_dict(init_file["model_F_mapping1_state_dict"])

    elif args.focus == "background":
        ic(args.focus)
        model_F_mapping = IMLP(
            input_dim=3,
            output_dim=2,
            hidden_dim=256,
            use_positional=False,
            positional_dim=2,
            num_layers=4,
            skip_layers=[]).to(args.device)
        
        model_F_mapping.load_state_dict(init_file["model_F_mapping2_state_dict"])

    model_F_mapping.eval()

    Loss_add = loss_addition.LossAddition(args,
                                          model = model_F_mapping,
                                          frames_shapes = renderer.frames_shapes,
                                          frames_paths = atlas_paths,
                                          index = index,
                                          masks = masks,
                                          imsize=224,
                                          device=args.device)


    sketches = renderer.get_frames()

    utils.save_sketches_video(sketches, f"{args.output_dir}/mp4_logs_{args.text}_{args.clip_model_name}_{''.join(str(x) for x in args.clip_conv_layer_weights)}_{args.clip_fc_loss_weight}_{args.width}", title=f"test.mp4")


    # tensor_image = ((atlas[0] - atlas[0].min()) / (atlas[0].max() - atlas[0].min())).squeeze(0).cpu()
    # pil_image = Image.fromarray((tensor_image * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy())
    # pil_image.save(f"{args.output_dir}/mp4_logs_{args.text}_{args.clip_model_name}_{''.join(str(x) for x in args.clip_conv_layer_weights)}_{args.clip_fc_loss_weight}_{args.width}/atlas.jpg")
    
    directory = Path(f"{args.output_dir}/atlas_logs_{args.text}_{args.clip_model_name}_{''.join(str(x) for x in args.clip_conv_layer_weights)}_{args.clip_fc_loss_weight}_{args.width}")
    if not os.path.exists(directory):
        os.makedirs(directory)
    utils.save_atlas(atlas, model_F_mapping, renderer.frames_shapes, visual_point, colors, 2000, f"{args.output_dir}/atlas_logs_{args.text}_{args.clip_model_name}_{''.join(str(x) for x in args.clip_conv_layer_weights)}_{args.clip_fc_loss_weight}_{args.width}/init.jpg", args.device)

    for i in tqdm(range(300)):
        loss = torch.tensor([0.]).to(args.device)

        # renderer.set_random_noise(i)
        if args.lr_scheduler:
            optimizer.update_lr(counter)
        
        optimizer.zero_grad_()
        # start0 = time.time()
        # loss += loss_addition.loss_init_path(renderer.frames_shapes, renderer.shapes_frames_index, model_F_mapping1, args.device)
        # loss += loss_addition.loss_init_atlas(renderer.frames_shapes, atlas_paths, model_F_mapping1, args.device)
        if args.focus == "foreground":
            loss += Loss_add.loss_init_inputs()
        if args.focus == "background" or args.focus == "atlas":
            loss += Loss_add.loss_init_atlas()
        loss.backward()
        optimizer.step_()

        # if i%50 == 0 and i>0:
        #     logging.info(f"losses_consist_init: {loss}")
        #     print(f"losses_consist_init: {loss}")
        #     sketches = renderer.get_frames()
        #     utils.save_sketches_video(sketches, f"{args.output_dir}/mp4_logs", title=f"init_{i}.mp4") 
        #     directory=f"{args.output_dir}/svg_logs/{i}"
        #     if not os.path.exists(directory):
        #         os.makedirs(directory)
        #     renderer.save_svgs(directory, f"svg_iter{i}")

    renderer.set_sparse(10)


    utils.save_atlas(atlas, model_F_mapping, renderer.frames_shapes, visual_point, colors, 5000, f"{args.output_dir}/atlas_logs_{args.text}_{args.clip_model_name}_{''.join(str(x) for x in args.clip_conv_layer_weights)}_{args.clip_fc_loss_weight}_{args.width}/warmup.jpg", args.device)

    sketches = renderer.get_frames()
    utils.save_sketches_video(sketches, f"{args.output_dir}/mp4_logs_{args.text}_{args.clip_model_name}_{''.join(str(x) for x in args.clip_conv_layer_weights)}_{args.clip_fc_loss_weight}_{args.width}", title=f"init.mp4") 

    # mark_points = renderer.mark_shapes()
    # max_mask = torch.max(torch.sum((torch.vstack(inputs)<1).reshape(args.num_of_frames,-1), dim=1))
    epoch_range = tqdm(range(args.num_iter))
    # epoch_range = tqdm(range(10001))
    start_width = 100000
    for epoch in epoch_range:
        if epoch > start_width and not renderer.width_is_on():
            renderer.init_width()
            optimizer.turn_on_width()
            optimizer.init_width_optimizers()
            
        if not args.display:
            epoch_range.refresh()
        renderer.set_random_noise(epoch)
        if args.lr_scheduler:
            optimizer.update_lr(counter)

        # if epoch > 1000 and not renderer.addition_on:
        #     renderer.init_addition(5)
        #     renderer.turn_on_addition()
        #     renderer.turn_off_main()
        #     optimizer.init_optimizers()

        optimizer.zero_grad_()
        # start0 = time.time()
        sketches = renderer.get_frames()
        # end0 = time.time()
        # print(f"frames time: {end0-start0}")
        loss = torch.tensor([0.]).to(args.device)
        losses_consist = torch.tensor([0.]).to(args.device)
        losses_clip = torch.tensor([0.]).to(args.device)
        # losses_frames = torch.tensor([0.]).to(args.device)
        # losses_MSE = torch.tensor([0.]).to(args.device)
        # losses_ratio = torch.tensor([0.]).to(args.device)
        # losses_sparse = torch.tensor([0.]).to(args.device)
        
        # if epoch < 5000:
        # losses_consist += loss_addition.loss_key_con(renderer.frames_shapes, index, model_F_mapping1, args.device)
        # for i in index:
        #     sketches_batch = torch.vstack((sketches_batch, sketches[i]))
        #     inputs_batch = torch.vstack((inputs_batch, inputs[i]))
        # else:
        # losses_consist += loss_addition.loss_con_(renderer.frames_shapes, model_F_mapping1, args.device)
        # losses_consist += loss_addition.loss_mark_con(renderer.frames_shapes, mark_points, model_F_mapping1, args.device)
        
        # if not renderer.addition_on:
        #     renderer.init_addition_path(5)
        #     # renderer.turn_off_main()
        #     renderer.turn_on_addition()
        #     optimizer.init_optimizers()

        losses_consist += Loss_add.loss_consist()
        # losses_consist += Loss_add.loss_consist_wMask()

            # losses_frames += Loss_add.loss_frame_wise()
            # losses_consist += Loss_add.loss_con_()
        
        if args.clip_model_name == "RN101":
            num = 22
        if args.clip_model_name == "ViT-B/32":
            num = 49

        # losses_clips = []
        # torch.max(torch.vstack(sketches)
        # losses_MSE += nn.MSELoss()(torch.vstack(sketches[1:-1]), torch.vstack(sketches[0:-2])) + nn.MSELoss()(torch.vstack(sketches[1:-1]), torch.vstack(sketches[2:]))
        # sketches_batch = torch.vstack(sketches)

        sketches_batch = []
        inputs_batch = []
        for i in np.random.choice(len(sketches), num, replace=False):
        # for i in np.random.choice(len(sketches), 20, replace=False):
            sketches_batch.append(sketches[i])
            inputs_batch.append(inputs[i])

        losses_dict = loss_func(torch.vstack(sketches_batch), torch.vstack(inputs_batch), args.text, renderer.get_color_parameters(), renderer, counter, optimizer)
            
        losses_clip = num * sum(list(losses_dict.values())) 

            # losses_dict.append(loss_func(sketches[i], inputs[i], renderer.get_color_parameters(), renderer, counter, optimizer))
        # losses_dict = loss_func(sketches_batch[1:], inputs_batch[1:].detach(
        # ), renderer.get_color_parameters(), renderer, counter, optimizer)

        if renderer.width_is_on():
            losses_sparse = torch.sum(renderer.loss_sparse())
            # losses_ratio = torch.tensor(sum[1:]>0)
            # ic((inputs_batch[1:]<1).reshape(num,-1).shape)
            # ic(torch.sum((inputs_batch[1:]<1).reshape(num,-1),dim=1).shape)
            # count = torch.sum((inputs_batch[1:]<1).reshape(num,-1),dim=1)

            # ic(torch.vstack(inputs).shape)
            ratio = torch.sqrt(torch.sum((torch.vstack(inputs)<1).reshape(args.num_of_frames,-1),dim=1) / max_mask)
            # losses_ratio = torch.tensor((range(1,5)))
            # losses_ratio = torch.sum(torch.abs(renderer.loss_sparse()[num_index] / torch.vstack(losses_clips)- ratio[num_index]**3))
            # losses_ratio = torch.sum(torch.abs(renderer.loss_sparse()[num_index]))
            # ic(losses_ratio)
            # ic(ratio[num_index]**3)
            # ic(renderer.loss_sparse()[num_index] / torch.vstack(losses_clips))
            # ic(renderer.loss_sparse()[num_index])
            # losses_ratio = torch.sum(torch.abs(renderer.loss_sparse[num_index] / losses_clip) - ratio)
        # end3 = time.time()
        # print(f"clip time: {end3-start3}")
        # loss += args.dist_param * losses_dist
        loss += args.consist_param * losses_consist
        loss += args.clip_param * losses_clip
        # loss += args.frames_param * losses_frame
        # if not renderer.width_is_on():
        # loss += losses_ratio
        # loss += 0.005*losses_sparse
        # loss += 500 * losses_MSE
        loss.backward()
        optimizer.step_()

        # if epoch % 499 == 0 and epoch != 0 and epoch < 5000:
        #     renderer.interpolate_key_frames()

        # j = random.randint(0,1)

        # if epoch == 0:
        #     utils.plot_batch(inputs[0], sketches[0], f"{args.output_dir}", counter,
        #                      use_wandb=args.use_wandb, title=f"the_first.jpg")
        if epoch > 0 and epoch % 10 == 0:
            logging.info(f"----------{epoch}------------")
            logging.info(f"loss: {loss}")
            logging.info(f"losses_consist: {losses_consist}")
            logging.info(f"losses_clip: {losses_clip}")

        if epoch > 0 and epoch % args.save_interval == 0:
            # logging.info(f"----------{epoch}------------")
            print(f"----------{epoch}------------")
            # logging.info(f"loss: {loss}")
            print(f"loss: {loss}")
            # print(f"losses_dist: {args.dist_param*losses_dist}")
            # logging.info(f"losses_consist: {losses_consist}")
            print(f"losses_consist: {losses_consist}")
            # logging.info(f"losses_clip: {losses_clip}")
            print(f"losses_clip: {losses_clip}")
            # logging.info(f"losses_frames: {losses_frames}")
            # print(f"losses_frames: {losses_frames}")
            # logging.info(f"losses_MSE: {losses_MSE}")
            # print(f"losses_MSE: {losses_MSE}")
            # logging.info(f"losses_sparse: {losses_sparse}")
            # print(f"losses_sparse: {losses_sparse}")
            # print(f"losses_clip: {loss - args.consist_param*losses_consist - args.dist_param*losses_dist}")
            # utils.plot_batch(inputs[j], sketches[j], f"{args.output_dir}/jpg_logs", counter,
            #                  use_wandb=args.use_wandb, title=f"iter{epoch}.jpg")
            # renderer.save_svg(
            #     f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")


            # utils.save_atlas(atlas, model_F_mapping, renderer.frames_shapes(), visual_point, 5000, f"{args.output_dir}/atlas_logs_{args.text}_{args.clip_model_name}_{''.join(str(x) for x in args.clip_conv_layer_weights)}_{args.clip_fc_loss_weight}_{args.width}/{epoch}.jpg" )

            utils.save_atlas(atlas, model_F_mapping, renderer.frames_shapes, visual_point, colors, 2000, f"{args.output_dir}/atlas_logs_{args.text}_{args.clip_model_name}_{''.join(str(x) for x in args.clip_conv_layer_weights)}_{args.clip_fc_loss_weight}_{args.width}/{epoch}.jpg", args.device)


            utils.save_sketches_video(sketches, f"{args.output_dir}/mp4_logs_{args.text}_{args.clip_model_name}_{''.join(str(x) for x in args.clip_conv_layer_weights)}_{args.clip_fc_loss_weight}_{args.width}", title=f"iter{epoch}.mp4")
            directory=f"{args.output_dir}/svg_logs_{args.text}_{args.clip_model_name}_{''.join(str(x) for x in args.clip_conv_layer_weights)}_{args.clip_fc_loss_weight}_{args.width}/{epoch}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            renderer.save_svgs(directory, f"svg_iter{epoch}")
            # if epoch >= 2000:
            #     for i in tqdm(range(100)):
            #         loss = torch.tensor([0.]).to(args.device)
            #         # renderer.set_random_noise(i)
            #         if args.lr_scheduler:
            #             optimizer.update_lr(counter)
                    
            #         optimizer.zero_grad_()
            #         # start0 = time.time()
            #         losses_consist = loss_addition.loss_con_(renderer.frames_shapes, model_F_mapping1, args.device)
            #         loss += losses_consist 
            #         loss.backward()
            #         optimizer.step_()

            #     sketches = renderer.get_frames()
            #     logging.info(f"losses_consist_after: {loss}")
            #     print(f"losses_consist_after: {loss}")
            #     utils.save_sketches_video(sketches, f"{args.output_dir}/mp4_logs", title=f"iter{epoch}_.mp4") 


        # if epoch % args.eval_interval == 0:
        #     with torch.no_grad():
                # losses_dict_eval = loss_func(sketches[0], inputs[0], renderer.get_color_parameters(
                # ), renderer.get_points_parans(), counter, optimizer, mode="eval")
                # loss_eval = sum(list(losses_dict_eval.values()))
                # configs_to_save["loss_eval"].append(loss_eval.item())
                # for k in losses_dict_eval.keys():
                #     if k not in configs_to_save.keys():
                #         configs_to_save[k] = []
                #     configs_to_save[k].append(losses_dict_eval[k].item())
                # if args.clip_fc_loss_weight:
                #     if losses_dict_eval["fc"].item() < best_fc_loss:
                #         best_fc_loss = losses_dict_eval["fc"].item(
                #         ) / args.clip_fc_loss_weight
                #         best_iter_fc = epoch
                # print(
                #     f"eval iter[{epoch}/{args.num_iter}] loss[{loss.item()}] time[{time.time() - start}]")

                # cur_delta = loss_eval.item() - best_loss
                # if abs(cur_delta) > min_delta:
                #     if cur_delta < 0:
                #         best_loss = loss_eval.item()
                #         best_iter = epoch
                #         terminate = False
                #         utils.save_sketches_video(sketches, args.output_dir, title="best_iter.mp4")
                        # utils.plot_batch(
                        #     inputs[j], sketches[j], args.output_dir, counter, use_wandb=args.use_wandb, title="best_iter.jpg")
                        # renderer.save_svg(args.output_dir, "best_iter")

                # if abs(cur_delta) <= min_delta:
                #     if terminate:
                #         break
                #     terminate = True

        # if counter == 0 and args.attention_init:
        #     utils.plot_atten(renderer.get_attn(), renderer.get_thresh(), inputs[0], renderer.get_inds(),
        #                      args.use_wandb, "{}/{}.jpg".format(
        #                          args.output_dir, "attention_map"),
        #                      args.saliency_model, args.display_logs)

        counter += 1

    # renderer.save_svg(args.output_dir, "final_svg")
    # path_svg = os.path.join(args.output_dir, "best_iter.svg")
    # utils.log_sketch_summary_final(
    #     path_svg, args.use_wandb, args.device, best_iter, best_loss, "best total")

    # utils.save_sketches_video(sketches[0], args.output_dir, title="best_total.mp4")

    for i in tqdm(range(500)):
        loss = torch.tensor([0.]).to(args.device)
        # renderer.set_random_noise(i)
        if args.lr_scheduler:
            optimizer.update_lr(counter)
        
        optimizer.zero_grad_()
        # start0 = time.time()
        # loss += loss_addition.loss_init_path(renderer.frames_shapes, renderer.shapes_frames_index, model_F_mapping1, args.device)
        # loss += loss_addition.loss_init_atlas(renderer.frames_shapes, atlas_paths, model_F_mapping1, args.device)
        loss = Loss_add.loss_consist()

        loss.backward()
        optimizer.step_()

        if i%50 == 0 and i>0:
            logging.info(f"losses_consist_init: {loss}")
            print(f"losses_consist_init: {loss}")
            


            sketches = renderer.get_frames()
            utils.save_sketches_video(sketches, f"{args.output_dir}/mp4_logs_{args.text}_{args.clip_model_name}_{''.join(str(x) for x in args.clip_conv_layer_weights)}_{args.clip_fc_loss_weight}_{args.width}", title=f"iter{epoch}.mp4")
            directory=f"{args.output_dir}/svg_logs_{args.text}_{args.clip_model_name}_{''.join(str(x) for x in args.clip_conv_layer_weights)}_{args.clip_fc_loss_weight}_{args.width}/{epoch}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            renderer.save_svgs(directory, f"svg_iter_consist_{epoch}")



    return configs_to_save

if __name__ == "__main__":
    args = config.parse_arguments()
    final_config = vars(args)
    # args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    try:
        configs_to_save = main(args)
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    for k in configs_to_save.keys():
        final_config[k] = configs_to_save[k]
    np.save(f"{args.output_dir}/config.npy", final_config)
    if args.use_wandb:
        wandb.finish()
