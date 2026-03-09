import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import json
import trimesh
from copy import deepcopy

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')   # 'cameras_sphere.npz'
        self.object_cameras_name = conf.get_string('object_cameras_name')
        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)
        self.type = conf.get_string('type')
        
        self.images_lis = []
        self.normal_lis = []
        pose_all = []

        if self.type == 'nerf':
            with open(os.path.join(self.data_dir, 'transforms_train.json'), 'r') as fp:
                data_info = json.load(fp)
            for frame in data_info['frames']:
                img_path = os.path.join(self.data_dir, frame['file_path'][2:] + '.png')
                normal_path = os.path.join(self.data_dir, frame['file_path'][2:] + '_normal' + '.png')
                pose_all.append(torch.from_numpy(np.array(frame['transform_matrix'], dtype=np.float32)))
                self.images_lis.append(img_path)
                self.normal_lis.append(normal_path)

            pose_all = torch.stack(pose_all).cuda()

            # Scale_mat: transform the object to unit sphere for training
            pcd = trimesh.load(os.path.join(self.data_dir, 'points_of_interest.ply'))
            vertices = pcd.vertices
            bbox_max = np.max(vertices, axis=0) 
            bbox_min = np.min(vertices, axis=0) 
            center = (bbox_max + bbox_min) * 0.5
            radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max() 
            scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
            scale_mat[:3, 3] = center

            # Scale_mat: transform the reconstructed mesh in unit sphere to original space with scale 150 for evaluation
            self.scale_mat = deepcopy(scale_mat)
            self.scale_mat[0, 0] *= 150
            self.scale_mat[1, 1] *= 150
            self.scale_mat[2, 2] *= 150
            self.scale_mat[:3, 3] *= 150

            for i in range(pose_all.shape[0]):
                pose_all[i, :, 3:] = torch.from_numpy(np.linalg.inv(scale_mat)).cuda() @ pose_all[i, :, 3:]
            
            # from opencv to opengl
            self.pose_all = torch.matmul(pose_all, torch.diag(torch.tensor([1., -1., -1., 1.])))
            
            self.n_images = len(self.images_lis)
            self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
            self.normal_np = np.stack([cv.imread(im_name) for im_name in self.normal_lis]) 
            self.H, self.W, _ = self.images_np[0].shape
            
            # intrinsic
            camera_angle_x = float(data_info['camera_angle_x'])
            self.focal = .5 * self.W / np.tan(.5 * camera_angle_x)
            self.intrinsics_all = []
            intrinsics = torch.Tensor([
                [self.focal, 0, self.W / 2, 0],
                [0, self.focal, self.H / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]).float()
            for i in range(self.images_np.shape[0]):
                self.intrinsics_all.append(intrinsics)
                
            self.masks_np = np.ones_like(self.images_np) * 255. / 256.
            self.images = torch.from_numpy(self.images_np.astype(np.float32)).cuda()  # [n_images, H, W, 3]
            self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cuda()  # [n_images, H, W, 3]
            self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
            self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
            self.focal = self.intrinsics_all[0][0, 0]
            self.inv_pose_all = torch.inverse(self.pose_all)
            self.H, self.W = self.images.shape[1], self.images.shape[2]
            self.image_pixels = self.H * self.W
            self.all_rays_o = self.pose_all[:,:3,3]

            object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
            object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
            self.object_bbox_min = object_bbox_min[:3]
            self.object_bbox_max = object_bbox_max[:3]

        elif self.type == 'ours':
            import xml.etree.ElementTree as ET
            tree = ET.parse(os.path.join(self.data_dir, 'cameras.xml'))
            root = tree.getroot()
            for child in root:
                for camera in child.iter('camera'):
                    label = camera.get('label')
                    img_path = os.path.join(self.data_dir, 'images/', label + '.JPG')
                    transform_element = camera.find('transform').text
                    transform_values = np.fromstring(transform_element, sep=' ').reshape(4, 4)

                    pose_all.append(torch.from_numpy(transform_values))
                    self.images_lis.append(img_path)
            pose_all = torch.stack(pose_all).cuda()

            # 设置一个scale for mesh
            # Scale_mat: transform the object to unit sphere for training
            pcd = trimesh.load(os.path.join(self.data_dir, 'mesh.ply'))
            vertices = pcd.vertices
            bbox_max = np.max(vertices, axis=0) 
            bbox_min = np.min(vertices, axis=0) 
            center = (bbox_max + bbox_min) * 0.5
            radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max() 
            scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
            scale_mat[:3, 3] = center

            # Scale_mat: transform the reconstructed mesh in unit sphere to original space with scale 150 for evaluation
            self.scale_mat = deepcopy(scale_mat)
            self.scale_mat[0, 0] *= 150
            self.scale_mat[1, 1] *= 150
            self.scale_mat[2, 2] *= 150
            self.scale_mat[:3, 3] *= 150

            for i in range(pose_all.shape[0]):
                pose_all[i, :, 3:] = torch.from_numpy(np.linalg.inv(scale_mat)).cuda() @ pose_all[i, :, 3:].float()

            # 转换坐标系
            # self.pose_all = torch.matmul(pose_all, torch.diag(torch.tensor([1., -1., -1., 1.])).double()).float()
            self.pose_all = pose_all.float()

            self.n_images = len(self.images_lis)
            self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
            self.H, self.W, _ = self.images_np[0].shape
            
            # intrinsic
            
            self.focal = 2552.4247379882272
            self.intrinsics_all = []
            intrinsics = torch.Tensor([
                [self.focal, 0, self.W / 2, 0],
                [0, self.focal, self.H / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]).float()
            for i in range(self.images_np.shape[0]):
                self.intrinsics_all.append(intrinsics)
                
            self.masks_np = np.ones_like(self.images_np) * 255. / 256.
            self.images = torch.from_numpy(self.images_np.astype(np.float32)).cuda()  # [n_images, H, W, 3]
            self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cuda()  # [n_images, H, W, 3]
            self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
            self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
            self.focal = self.intrinsics_all[0][0, 0]
            self.inv_pose_all = torch.inverse(self.pose_all)
            
            self.image_pixels = self.H * self.W
            self.all_rays_o = self.pose_all[:,:3,3]

            object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
            object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
            self.object_bbox_min = object_bbox_min[:3]
            self.object_bbox_max = object_bbox_max[:3]

        elif self.type == 'robustnerf':
            camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
            self.camera_dict = camera_dict
            self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
            self.n_images = len(self.images_lis)

            images = [cv.imread(im_name) for im_name in self.images_lis]
            min_height = min(image.shape[0] for image in images)
            min_width = min(image.shape[1] for image in images)
            cropped_images = []

            # 遍历所有图像，将它们裁剪为最小图像的大小
            for image in images:
                cropped_image = image[:min_height, :min_width]
                cropped_images.append(cropped_image)


            self.images_np = np.stack(cropped_images) / 256.0

            self.masks_np = np.ones_like(self.images_np) * 255. / 256.
            

            # world_mat is a projection matrix from world to image
            self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

            self.scale_mats_np = []

            # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
            self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

            self.intrinsics_all = []
            self.pose_all = []

            for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsics, pose = load_K_Rt_from_P(None, P)
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                self.pose_all.append(torch.from_numpy(pose).float())

            self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
            self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
            self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
            self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
            self.focal = self.intrinsics_all[0][0, 0]
            self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
            # self.pose_all = torch.matmul(self.pose_all, torch.diag(torch.tensor([-1., -1., -1., 1.]))).float()
            
            self.H, self.W = self.images.shape[1], self.images.shape[2]
            self.image_pixels = self.H * self.W

            object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
            object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
            # Object scale mat: region of interest to **extract mesh**
            object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
            object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
            object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
            self.object_bbox_min = object_bbox_min[:3, 0]
            self.object_bbox_max = object_bbox_max[:3, 0]

        else:
            camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
            self.camera_dict = camera_dict
            self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
            self.n_images = len(self.images_lis)
            self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
            self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
            self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

            # world_mat is a projection matrix from world to image
            self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

            self.scale_mats_np = []

            # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
            self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

            self.intrinsics_all = []
            self.pose_all = []

            for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsics, pose = load_K_Rt_from_P(None, P)
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                self.pose_all.append(torch.from_numpy(pose).float())

            self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
            self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
            self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
            self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
            self.focal = self.intrinsics_all[0][0, 0]
            self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
            self.H, self.W = self.images.shape[1], self.images.shape[2]
            self.image_pixels = self.H * self.W

            object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
            object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
            # Object scale mat: region of interest to **extract mesh**
            object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
            object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
            object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
            self.object_bbox_min = object_bbox_min[:3, 0]
            self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)
    

    def gen_rays_visu(self, img_idx, pose, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """

        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(pose[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3    这行不一样
        rays_o = pose[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3   这行不一样
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1), torch.stack([pixels_y, pixels_x], dim=-1).transpose(0, 1)



    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y.to('cpu'), pixels_x.to('cpu'))]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y.to('cpu'), pixels_x.to('cpu'))]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3   转为相机坐标系/d
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o, rays_v, color.cuda(), mask[:, :1].cuda()], dim=-1).cuda()    # batch_size, 10
      
    def gen_random_rays_at_ref(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o, rays_v, color, mask[:, :1]], dim=-1).cuda(), torch.stack([pixels_y, pixels_x], dim=-1)     # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
    
    def mask_at(self, idx, resolution_level):
        mask = cv.imread(self.masks_lis[idx])
        return (cv.resize(mask, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
