import collections
import CLIP_.clip as clip
import torch
import torch.nn as nn
from torchvision import models, transforms

import random
from icecream import ic
import copy


class LossAddition(nn.Module):
    def __init__(self, args,
                model=None,
                frames_shapes=None,
                frames_paths=None,
                index=None,
                masks=None,
                imsize=224,
                device=None):
        super(LossAddition, self).__init__()

        self.args = args
        # self.masks = torch.stack(masks, axis=-1).squeeze(0)
        self.masks = torch.cat(masks)
        self.model = model
        self.frames_shapes = frames_shapes
        self.frames_num = len(frames_shapes)
        self.paths_num = len(frames_shapes[0])
        self.points_num = len(frames_shapes[0][0].points)
        self.device = device
        self.frames_paths = frames_paths
        self.index = index
        self.all_points, self.all_f = self.build_init()
        if frames_paths != None:
            self.all_atlas = self.build_atlas_gt()
        if index != None:
            self.all_inputs_uv = self.build_inputs_gt()
        self.xy_current_consist, self.f_current_consist, \
        self.xy_near_consist, self.f_near_consist = self.build_consist_atlas()

        self.points_pre, self.points_mid, self.points_aft = self.build_frames_wise()
        # print(self.points_pre)

    def build_frames_wise(self):
        paths_mid = [element for sublist in self.frames_shapes[1:-1] for element in sublist]
        points_mid = [path.points for path in paths_mid]

        paths_aft = [element for sublist in self.frames_shapes[2:] for element in sublist]
        points_aft = [path.points for path in paths_aft]

        paths_pre = [element for sublist in self.frames_shapes[0:-2] for element in sublist]
        points_pre = [path.points for path in paths_pre]

        return points_pre, points_mid, points_aft


    def build_init(self):
        paths = [element for sublist in self.frames_shapes for element in sublist]
        points = [path.points for path in paths]

        f = torch.tensor([[i]*(self.paths_num*self.points_num) for i in range(self.frames_num)]).view(-1).to(self.device).unsqueeze(1)

        return points, f
    
    def build_atlas_gt(self):
        return torch.tensor(self.frames_paths).reshape(-1,2).repeat(self.frames_num,1).to(self.device)
    
    def build_inputs_gt(self):
        all_shapes = copy.deepcopy(self.frames_shapes)[0]
        all_points = torch.vstack([path.points for path in all_shapes])
        f = torch.tensor(copy.deepcopy(self.index)).repeat_interleave(self.points_num, dim=0).to(self.device)
        xyf_current = torch.stack((all_points[:,0]/(224/2)-1, all_points[:,1]/(224/2)-1, (f/(self.frames_num/2)-1).detach()),dim=1).to(self.device)
        uv_current = self.model(xyf_current)
        return uv_current.repeat(self.frames_num,1)
    
    def build_consist_atlas(self):
        xy_current_consist = []
        f_current_consist = []
        xy_near_consist = []
        f_near_consist = []
        for i in range(self.frames_num):
            for j in range(self.paths_num):
                    xy = self.frames_shapes[i][j].points
                    z = i * torch.ones((4)).to(self.device)
                    if i+1 <= self.frames_num-1:
                        xy_p1 = self.frames_shapes[i+1][j].points
                        z_p1 = (i+1)* torch.ones((4)).to(self.device)

                        xy_current_consist.append(xy)
                        f_current_consist.append(z)
                        xy_near_consist.append(xy_p1)
                        f_near_consist.append(z_p1)

                    if i-1 >= 0:
                        xy_n1 = self.frames_shapes[i-1][j].points
                        z_n1 = (i-1)* torch.ones((4)).to(self.device)

                        xy_current_consist.append(xy)
                        f_current_consist.append(z)
                        xy_near_consist.append(xy_n1)
                        f_near_consist.append(z_n1)

        return xy_current_consist, torch.cat(f_current_consist), xy_near_consist, torch.cat(f_near_consist)

    def loss_init_atlas(self):
        loss = torch.tensor([0.]).to(self.device)
        xy_current = torch.vstack(self.all_points)
        xyf_current = torch.stack((xy_current[:,0]/(224/2)-1, xy_current[:,1]/(224/2)-1, (self.all_f/(self.frames_num/2)-1).detach().squeeze(1)),dim=1).to(self.device)

        uv_current = self.model(xyf_current)
        loss += torch.sum(torch.abs(uv_current[:,0] - (self.all_atlas*2-1).detach()[:,0]))
        loss += torch.sum(torch.abs(uv_current[:,1] - (self.all_atlas*2-1).detach()[:,1]))

        return loss
    
    def loss_init_inputs(self):
        loss = torch.tensor([0.]).to(self.device)
        xy_current = torch.vstack(self.all_points)
        xyf_current = torch.stack((xy_current[:,0]/(224/2)-1, xy_current[:,1]/(224/2)-1, (self.all_f/(self.frames_num/2)-1).detach().squeeze(1)),dim=1).to(self.device)

        uv_current = self.model(xyf_current)
        loss += torch.sum(torch.abs(uv_current[:,0] - self.all_inputs_uv.detach()[:,0]))
        loss += torch.sum(torch.abs(uv_current[:,1] - self.all_inputs_uv.detach()[:,1]))

        return loss

    def loss_frame_wise(self):
        loss = torch.sum(torch.abs(torch.hstack(self.points_pre) - torch.hstack(self.points_mid)) + torch.abs(torch.hstack(self.points_aft) - torch.hstack(self.points_mid)) )
        return loss
    
    def loss_consist(self):
        loss = torch.tensor([0.]).to(self.device)

        xy_current = torch.vstack(self.xy_current_consist)
        xyf_current = torch.stack((xy_current[:,0]/(224/2)-1, xy_current[:,1]/(224/2)-1, (self.f_current_consist/(self.frames_num/2)-1).detach()),dim=1).to(self.device)

        xy_near = torch.vstack(self.xy_near_consist)
        xyf_near = torch.stack((xy_near[:,0]/(224/2)-1, xy_near[:,1]/(224/2)-1, (self.f_near_consist/(self.frames_num/2)-1).detach()),dim=1).to(self.device)

        uv_current = self.model(xyf_current)
        uv_near = self.model(xyf_near)
        loss += torch.sum(torch.abs(uv_current[:,0] - uv_near[:,0]))
        loss += torch.sum(torch.abs(uv_current[:,1] - uv_near[:,1]))

        return loss
    
    # def loss_consist_wMask(self):
    #     # loss_1 = torch.tensor([0.]).to(self.device)
    #     loss = torch.tensor([0.]).to(self.device)

    #     xy_current = torch.vstack(self.xy_current_consist)
    #     xyf_current = torch.stack((xy_current[:,0]/(224/2)-1, xy_current[:,1]/(224/2)-1, (self.f_current_consist/(self.frames_num/2)-1).detach()),dim=1).to(self.device)

    #     xy_near = torch.vstack(self.xy_near_consist)
    #     xyf_near = torch.stack((xy_near[:,0]/(224/2)-1, xy_near[:,1]/(224/2)-1, (self.f_near_consist/(self.frames_num/2)-1).detach()),dim=1).to(self.device)

    #     consist_mask = self.masks[(self.f_near_consist).long(), xy_near[:,1].long(), xy_near[:,0].long()]
    #     # neighbor_mask = torch.ones((consist_mask).shape).to(self.device) - consist_mask

    #     xyf = xyf_current[consist_mask!=0]
    #     xyf_ = xyf_near[consist_mask!=0]
    #     uv_current = self.model(xyf)
    #     uv_near = self.model(xyf_)

    #     # loss += torch.sum(torch.abs(uv_current[:,0] - uv_near[:,0])*consist_mask)
    #     # loss += torch.sum(torch.abs(uv_current[:,1] - uv_near[:,1])*consist_mask)

    #     loss += torch.sum(torch.abs(uv_current[:,0] - uv_near[:,0].detach()))
    #     loss += torch.sum(torch.abs(uv_current[:,1] - uv_near[:,1].detach()))
    #     # ic(loss_1)
    #     # ic(loss)
    #     # loss += torch.sum(torch.abs(torch.vstack(self.xy_current_consist)[:,0]-torch.vstack(self.xy_near_consist)[:,0].detach())*neighbor_mask)
    #     # loss += torch.sum(torch.abs(torch.vstack(self.xy_current_consist)[:,1]-torch.vstack(self.xy_near_consist)[:,1].detach())*neighbor_mask)

    #     return loss
    
    def loss_con_(self):
        loss = torch.tensor([0.]).to(self.device)
        xyz_near_all = torch.zeros(1,3).to(self.device)
        xyz_current_all = torch.zeros(1,3).to(self.device)
        for i in range(self.frames_num):
            for j in range(self.paths_num):
                    xy = self.frames_shapes[i][j].points/(224/2)-1
                    z = (i/(self.frames_num/2)-1) * torch.ones((4)).to(self.device)
                    xyz_current = torch.stack((xy[:,0],xy[:,1],z), dim=1)

                    if i+1 <= self.frames_num-1:
                        xy = self.frames_shapes[i+1][j].points/(224/2)-1
                        z = ((i+1)/(self.frames_num/2)-1) * torch.ones((4)).to(self.device)
                        xyz_next = torch.stack((xy[:,0],xy[:,1],z), dim=1)
                        xyz_near_all = torch.vstack((xyz_near_all, xyz_next))
                        xyz_current_all = torch.vstack((xyz_current_all, xyz_current))

                    if i-1 >= 0:
                        xy = self.frames_shapes[i-1][j].points/(224/2)-1
                        z = ((i-1)/(self.frames_num/2)-1) * torch.ones((4)).to(self.device)
                        xyz_pre = torch.stack((xy[:,0],xy[:,1],z), dim=1)
                        xyz_near_all = torch.vstack((xyz_near_all, xyz_pre))
                        xyz_current_all = torch.vstack((xyz_current_all, xyz_current))

        xy_current_all = self.model(xyz_current_all[1:])
        xy_near_all = self.model(xyz_near_all[1:])
        loss += torch.sum(torch.abs(xy_current_all[:,0] - xy_near_all.detach()[:,0]))
        loss += torch.sum(torch.abs(xy_current_all[:,1] - xy_near_all.detach()[:,1]))
        return loss


def loss_dist(frames_shapes):
    loss = 0
    for i in range(len(frames_shapes)):
        for j in range(len(frames_shapes[0])):
            for k in range(len(frames_shapes[0][0].points)):
                if i+1 <= len(frames_shapes) - 1:
                    loss += torch.abs(frames_shapes[i][j].points[k][0] - frames_shapes[i+1][j].points[k][0])
                    loss += torch.abs(frames_shapes[i][j].points[k][1] - frames_shapes[i+1][j].points[k][1])
                if i-1 >= 0:
                    loss += torch.abs(frames_shapes[i][j].points[k][0] - frames_shapes[i-1][j].points[k][0])
                    loss += torch.abs(frames_shapes[i][j].points[k][1] - frames_shapes[i-1][j].points[k][1])
    return loss

def loss_con(frames_shapes, model_F_mapping1, device):
    loss = 0

    for i in range(len(frames_shapes)):
        for j in range(len(frames_shapes[0])):
            for k in range(len(frames_shapes[0][0].points)):
                x = frames_shapes[i][j].points[k][0]/(224/2)-1
                y = frames_shapes[i][j].points[k][1]/(224/2)-1
                z = torch.tensor(i/(49/2)-1).to(device)
                xy_current = torch.stack((x,y,z))

                x_current, y_current = model_F_mapping1(xy_current)

                x_ = frames_shapes[0][j].points[k][0]/(224/2)-1
                y_ = frames_shapes[0][j].points[k][1]/(224/2)-1
                z_ = torch.tensor(0/(49/2)-1).to(device)
                xy_true = torch.stack((x_,y_,z_))

                x_true, y_true = model_F_mapping1(xy_true)

                loss += torch.abs(x_current - x_true.detach())
                loss += torch.abs(y_current - y_true.detach())

    return loss

def loss_con(frames_shapes, model_F_mapping1, num, device):
    loss = 0

    for i in range(len(frames_shapes)):
        for j in range(len(frames_shapes[0])):
            for k in range(len(frames_shapes[0][0].points)):
                x = frames_shapes[i][j].points[k][0]/(224/2)-1
                y = frames_shapes[i][j].points[k][1]/(224/2)-1
                z = torch.tensor(i/(49/2)-1).to(device)
                xy_current = torch.stack((x,y,z))

                x_current, y_current = model_F_mapping1(xy_current)

                x_ = frames_shapes[num][j].points[k][0]/(224/2)-1
                y_ = frames_shapes[num][j].points[k][1]/(224/2)-1
                z_ = torch.tensor(num/(49/2)-1).to(device)
                xy_true = torch.stack((x_,y_,z_))

                x_true, y_true = model_F_mapping1(xy_true)

                loss += torch.abs(x_current - x_true.detach())
                loss += torch.abs(y_current - y_true.detach())

    return loss
  
def loss_con_(frames_shapes, model, device):
    loss = torch.tensor([0.]).to(device)
    xyz_near_all = torch.zeros(1,3).to(device)
    xyz_current_all = torch.zeros(1,3).to(device)
    for i in range(len(frames_shapes)):
        for j in range(len(frames_shapes[0])):
            # for k in range(len(frames_shapes[0][0].points)):
                xy = frames_shapes[i][j].points/(224/2)-1
                z = (i/(49/2)-1) * torch.ones((4)).to(device)
                xyz_current = torch.stack((xy[:,0],xy[:,1],z), dim=1)

                # x_current, y_current = model(xyz_current)
                if i+1 <= len(frames_shapes)-1:
                    xy = frames_shapes[i+1][j].points/(224/2)-1
                    z = ((i+1)/(49/2)-1) * torch.ones((4)).to(device)
                    xyz_next = torch.stack((xy[:,0],xy[:,1],z), dim=1)
                    xyz_near_all = torch.vstack((xyz_near_all, xyz_next))
                    xyz_current_all = torch.vstack((xyz_current_all, xyz_current))

                if i-1 >= 0:
                    xy = frames_shapes[i-1][j].points/(224/2)-1
                    z = ((i-1)/(49/2)-1) * torch.ones((4)).to(device)
                    xyz_pre = torch.stack((xy[:,0],xy[:,1],z), dim=1)
                    xyz_near_all = torch.vstack((xyz_near_all, xyz_pre))
                    xyz_current_all = torch.vstack((xyz_current_all, xyz_current))

    xy_current_all = model(xyz_current_all[1:])
    xy_near_all = model(xyz_near_all[1:])
    loss += torch.sum(torch.abs(xy_current_all[:,0] - xy_near_all.detach()[:,0]))
    loss += torch.sum(torch.abs(xy_current_all[:,1] - xy_near_all.detach()[:,1]))
    return loss

def loss_mark_con(frames_shapes, mark, model, device):
    loss = torch.tensor([0.]).to(device)
    xyz_near_all = torch.zeros(1,3).to(device)
    xyz_current_all = torch.zeros(1,3).to(device)
    for i in range(len(frames_shapes)):
        z = (i/(49/2)-1) * torch.ones((1)).to(device)
        for j in range(len(frames_shapes[0])):
            for k in range(len(frames_shapes[0][0].points)):
                if mark[i][j][k] != 0:
                    xy = frames_shapes[i][j].points[k]/(224/2)-1
                    # z = (i/(49/2)-1) * torch.ones((4)).to(device)
                    xyz_current = torch.tensor([xy[0],xy[1],z]).to(device)
                    if i+1 <= len(frames_shapes)-1:
                        if mark[i+1][j][k] != 0:
                            xy = frames_shapes[i+1][j].points[k]/(224/2)-1
                            z = ((i+1)/(49/2)-1) * torch.ones((1)).to(device)
                            xyz_next = torch.tensor([xy[0],xy[1],z]).to(device)
                            xyz_near_all = torch.vstack((xyz_near_all, xyz_next))
                            xyz_current_all = torch.vstack((xyz_current_all, xyz_current))

                    if i-1 >= 0:
                        if mark[i-1][j][k] != 0:
                            xy = frames_shapes[i-1][j].points[k]/(224/2)-1
                            z = ((i-1)/(49/2)-1) * torch.ones((1)).to(device)
                            xyz_pre = torch.tensor([xy[0],xy[1],z]).to(device)
                            xyz_near_all = torch.vstack((xyz_near_all, xyz_pre))
                            xyz_current_all = torch.vstack((xyz_current_all, xyz_current))

    xy_current_all = model(xyz_current_all[1:])
    xy_near_all = model(xyz_near_all[1:])
    loss += torch.sum(torch.abs(xy_current_all[:,0] - xy_near_all.detach()[:,0]))
    loss += torch.sum(torch.abs(xy_current_all[:,1] - xy_near_all.detach()[:,1]))
    return loss

def loss_key_con(frames_shapes, index, model, device):
    loss = torch.tensor([0.]).to(device)
    xyz_near_all = torch.zeros(1,3).to(device)
    xyz_current_all = torch.zeros(1,3).to(device)
    # for i in range(len(frames_shapes)):
    for i in range(len(index)):
        # ic(index[i])
        # ic(index[i+1])
        for j in range(len(frames_shapes[0])):
            # for k in range(len(frames_shapes[0][0].points)):
                xy = frames_shapes[index[i]][j].points/(224/2)-1
                z = (index[i]/(49/2)-1) * torch.ones((4)).to(device)
                xyz_current = torch.stack((xy[:,0],xy[:,1],z), dim=1)

                # x_current, y_current = model(xyz_current)
                if index[i]+1 <= len(frames_shapes)-1:
                    xy = frames_shapes[index[i+1]][j].points/(224/2)-1
                    z = ((index[i+1])/(49/2)-1) * torch.ones((4)).to(device)
                    xyz_next = torch.stack((xy[:,0],xy[:,1],z), dim=1)
                    xyz_near_all = torch.vstack((xyz_near_all, xyz_next))
                    xyz_current_all = torch.vstack((xyz_current_all, xyz_current))

                if index[i]-1 >= 0:
                    xy = frames_shapes[index[i-1]][j].points/(224/2)-1
                    z = ((index[i-1])/(49/2)-1) * torch.ones((4)).to(device)
                    xyz_pre = torch.stack((xy[:,0],xy[:,1],z), dim=1)
                    xyz_near_all = torch.vstack((xyz_near_all, xyz_pre))
                    xyz_current_all = torch.vstack((xyz_current_all, xyz_current))

    xy_current_all = model(xyz_current_all[1:])
    xy_near_all = model(xyz_near_all[1:])
    loss += torch.sum(torch.abs(xy_current_all[:,0] - xy_near_all.detach()[:,0]))
    loss += torch.sum(torch.abs(xy_current_all[:,1] - xy_near_all.detach()[:,1]))
    return loss

def loss_init_path(frames_shapes, shapes_index, model, device):
    loss = torch.tensor([0.]).to(device)
    xyz_true_all = torch.zeros(1,3).to(device)
    xyz_current_all = torch.zeros(1,3).to(device)
    for shapes in shapes_index:
    # for shapes in random.sample(shapes_index, 32):
        for num in range(len(frames_shapes[0])):
            xy_ = frames_shapes[shapes][num].points/(224/2)-1
            # y_ = frames_shapes[shapes][num].points/(224/2)-1
            z_ = (shapes/(49/2)-1) * torch.ones((4)).to(device)
            xyz_true = torch.stack((xy_[:,0],xy_[:,1],z_),dim=1)
            # xyz_true_all = torch.vstack((xyz_true_all, xyz_true))
            # xy_true = model(xyz_true)
            
            for i in range(len(frames_shapes)):
                xy = frames_shapes[i][num].points/(224/2)-1
                # y = frames_shapes[i][num].points/(224/2)-1
                z = (i/(49/2)-1) * torch.ones((4)).to(device)
                xyz_current = torch.stack((xy[:,0],xy[:,1],z),dim=1)
                xyz_current_all = torch.vstack((xyz_current_all, xyz_current))

                xyz_true_all = torch.vstack((xyz_true_all, xyz_true))
                # xy_current = model(xyz_current)
                # loss += torch.sum(torch.abs(xy_current[:,0] - xy_true.detach()[:,0]))
                # loss += torch.sum(torch.abs(xy_current[:,1] - xy_true.detach()[:,1]))
    xy_current_all = model(xyz_current_all)
    xy_true_all = model(xyz_true_all)
    loss += torch.sum(torch.abs(xy_current_all[:,0] - xy_true_all.detach()[:,0]))
    loss += torch.sum(torch.abs(xy_current_all[:,1] - xy_true_all.detach()[:,1]))


    return loss

def loss_init_atlas(frames_shapes, frame_path, model, device):
    loss = torch.tensor([0.]).to(device)
    xyz_current_all = torch.zeros(1,3).to(device)
    xy_atlas_all = torch.zeros(1,2).to(device)
    for i in range(len(frames_shapes)):
        for j in range(len(frames_shapes[0])):
            xy_ = frames_shapes[i][j].points/(224/2)-1
            # y_ = frames_shapes[shapes][num].points/(224/2)-1
            z_ = (i/(49/2)-1) * torch.ones((4)).to(device)
            xyz_current = torch.stack((xy_[:,0],xy_[:,1],z_),dim=1)
            xyz_current_all = torch.vstack((xyz_current_all, xyz_current))

            # xy_atlas = torch.stack(frame_path[j],)
            xy_atlas_all = torch.vstack((xy_atlas_all, (2*torch.tensor(frame_path[j])-1).to(device)))
                # xy_current = model(xyz_current)
                # loss += torch.sum(torch.abs(xy_current[:,0] - xy_true.detach()[:,0]))
                # loss += torch.sum(torch.abs(xy_current[:,1] - xy_true.detach()[:,1]))
    xy_current_all = model(xyz_current_all)
    loss += torch.sum(torch.abs(xy_current_all[1:,0] - xy_atlas_all.detach()[1:,0]))
    loss += torch.sum(torch.abs(xy_current_all[1:,1] - xy_atlas_all.detach()[1:,1]))


    return loss


class LossAddition_(nn.Module):
    def __init__(self, args,
                model = None,
                frames_shapes = None,
                shapes_frames_index = None,
                imsize=224,
                device=None):
        super(LossAddition_, self).__init__()

        self.args = args
        self.model = model
        self.frames_shapes = frames_shapes
        self.shapes_index = shapes_frames_index
        self.frames_num = len(frames_shapes)
        self.paths_num = len(frames_shapes[0])
        self.points_num = len(frames_shapes[0][0].points)
        self.device = device
        self.xyz_index_all, self.xyz_true_all = self.init_path()

        self.xyz_index_points, self.xyz_true_frame, self.xyz_true_points, self.xyz_index_frame = self.init_path_()

        self.xyz_current_all, self.xyz_near_all = self.init_consist()

    def init_path_(self):
        xyz_true_points = []
        xyz_true_frame = [] 
        xyz_index_points = []
        xyz_index_frame = []

        for shapes in self.shapes_index:
            for num in range(len(self.frames_shapes[0])):
                # xyz_true_points.append(self.frames_shapes[shapes][num].points)
                # # z_ = (shapes/(49/2)-1) * torch.ones((4)).to(self.device)
                # # xyz_true = torch.stack((xy_[:,0],xy_[:,1],z_),dim=1)
                
                for i in range(len(self.frames_shapes)):
                    xyz_true_points.append(self.frames_shapes[shapes][num].points.to(self.device))
                    xyz_true_frame.append((shapes/(49/2)-1) * torch.ones((4)).to(self.device))
                    xyz_index_points.append(self.frames_shapes[i][num].points.to(self.device))
                    xyz_index_frame.append((i/(49/2)-1) * torch.ones((4)).to(self.device))

        return xyz_index_points, xyz_true_frame, xyz_true_points, xyz_index_frame

    def init_path(self):
        xyz_true_all = torch.zeros(1,3).to(self.device)
        xyz_index_all = torch.zeros(1,3).to(self.device)

        for shapes in self.shapes_index:
            for num in range(len(self.frames_shapes[0])):
                xy_ = self.frames_shapes[shapes][num].points/(224/2)-1
                z_ = (shapes/(49/2)-1) * torch.ones((4)).to(self.device)
                xyz_true = torch.stack((xy_[:,0],xy_[:,1],z_),dim=1)
                
                for i in range(len(self.frames_shapes)):
                    xy = self.frames_shapes[i][num].points/(224/2)-1
                    z = (i/(49/2)-1) * torch.ones((4)).to(self.device)
                    xyz_index = torch.stack((xy[:,0],xy[:,1],z),dim=1)
                    xyz_index_all = torch.vstack((xyz_index_all, xyz_index))
                    # xyz_index_all.requires_grad = True

                    xyz_true_all = torch.vstack((xyz_true_all, xyz_true))
                    # xyz_true_all.requires_grad = True

        return xyz_index_all, xyz_true_all

    def init_consist(self):
        xyz_near_all = torch.zeros(1,3).to(self.device)
        xyz_current_all = torch.zeros(1,3).to(self.device)

        for i in range(len(self.frames_shapes)):
            for j in range(len(self.frames_shapes[0])):
                    xy = self.frames_shapes[i][j].points/(224/2)-1
                    z = (i/(49/2)-1) * torch.ones((4)).to(self.device)
                    xyz_current = torch.stack((xy[:,0],xy[:,1],z), dim=1)

                    if i+1 <= len(self.frames_shapes)-1:
                        xy = self.frames_shapes[i+1][j].points/(224/2)-1
                        z = ((i+1)/(49/2)-1) * torch.ones((4)).to(self.device)
                        xyz_next = torch.stack((xy[:,0],xy[:,1],z), dim=1)
                        xyz_near_all = torch.vstack((xyz_near_all, xyz_next))
                        xyz_current_all = torch.vstack((xyz_current_all, xyz_current))

                    if i-1 >= 0:
                        xy = self.frames_shapes[i-1][j].points/(224/2)-1
                        z = ((i-1)/(49/2)-1) * torch.ones((4)).to(self.device)
                        xyz_pre = torch.stack((xy[:,0],xy[:,1],z), dim=1)
                        xyz_near_all = torch.vstack((xyz_near_all, xyz_pre))
                        xyz_current_all = torch.vstack((xyz_current_all, xyz_current))

        return xyz_current_all, xyz_near_all
    
    def consist_loss(self):
        xy_current_all = self.model(self.xyz_current_all)
        xy_near_all = self.model(self.xyz_near_all)
        loss = torch.sum(torch.abs(xy_near_all[:,0] - xy_current_all.detach()[:,0]))
        loss += torch.sum(torch.abs(xy_current_all[:,1] - xy_current_all.detach()[:,1]))
        return loss
    
    def init_loss(self):
        # n = random.randint(0,10)
        xy_index_all = self.model(self.xyz_index_all)
        xy_true_all = self.model(self.xyz_true_all)
        loss = torch.sum(torch.abs(xy_true_all[:,0] - xy_index_all.detach()[:,0]))
        loss += torch.sum(torch.abs(xy_true_all[:,1] - xy_index_all.detach()[:,1]))

        # self.xyz_index_all  = self.xyz_index_all.detach()
        # self.xyz_true_all = self.xyz_true_all.detach()

        return loss
    
    def init_loss_(self):
        loss = torch.tensor([0.]).to(self.device)
        xyz_true_all = torch.zeros(1,3).to(self.device)
        xyz_current_all = torch.zeros(1,3).to(self.device)
        for i in range(len(self.xyz_index_points)):
            xy_ = self.xyz_index_points[i]/(224/2)-1
            z_ = self.xyz_index_frame[i]
            xyz_true = torch.stack((xy_[:,0],xy_[:,1],z_),dim=1)
            # xyz_true_all = torch.vstack((xyz_true_all, xyz_true))
            # xy_true = model(xyz_true)
            # for i in range(len(frames_shapes)):
            xy = self.xyz_true_points[i]/(224/2)-1
            # y = frames_shapes[i][num].points/(224/2)-1
            z = self.xyz_true_frame[i]
            xyz_current = torch.stack((xy[:,0],xy[:,1],z),dim=1)

            xyz_current_all = torch.vstack((xyz_current_all, xyz_current)).to(self.device)
            xyz_true_all = torch.vstack((xyz_true_all, xyz_true)).to(self.device)

        xy_current_all = self.model(xyz_current_all)
        xy_true_all = self.model(xyz_true_all)
        loss += torch.sum(torch.abs(xy_current_all[:,0] - xy_true_all.detach()[:,0]))
        loss += torch.sum(torch.abs(xy_current_all[:,1] - xy_true_all.detach()[:,1]))

        return loss