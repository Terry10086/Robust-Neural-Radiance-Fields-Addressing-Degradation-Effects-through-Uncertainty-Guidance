import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF, UncertaintyNetwork, FeatExtractor
from models.renderer import NeuSRenderer
torch.autograd.set_detect_anomaly(True)
import copy
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
logging.getLogger('matplotlib.font_manager').disabled = True

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=True):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.ckpt_dir = self.conf['general.ckpt_dir']
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.nll_weight = self.conf.get_float('train.nll_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        self.uncert_network = UncertaintyNetwork(**self.conf['model.uncertainty_network']).to(self.device)
        self.feat_network = FeatExtractor(batchNorm=False, c_in = 3).to(self.device)


        # histogram
        self.bin_scale = self.conf['histogram.bin_scale']
        self.n_bins = self.conf.get_int('histogram.n_bins')
        


        # params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        params_for_uncert = []
        params_for_uncert += list(self.uncert_network.parameters())

        params_for_out = []
        params_for_out += list(self.nerf_outside.parameters())


        # self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.optimizer = torch.optim.Adam([{'params': params_to_train, 'lr': self.learning_rate},
                                           {'params':params_for_out,'lr':self.learning_rate},
                                           {'params': params_for_uncert, 'lr': 0.0001},
                                          ])    #  {'params': params_for_feat, 'lr': 0.0001} 

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.uncert_network,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.ckpt_dir, 'checkpoints'))
            print(self.ckpt_dir)
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        print('Total images:',image_perm.shape[0])
        psnr_list = []
        loss_list = []
        iter_list = []
        term1_list = []
        term2_list = []
        
        
        for iter_i in tqdm(range(res_step)):
            # self.scheduler.step()
            if self.conf['dataset']['type'] == 'nerf':
                data, image_data = self.dataset.gen_random_rays_at_ref(image_perm[self.iter_step % len(image_perm)], self.batch_size)   # 123张图像
            else:
                data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)   # 123张图像
            

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,image_feat=None,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']
            uncert_ests = render_out['uncert_map'] 
            if uncert_ests.shape[-1]==1:
                    uncert_ests = uncert_ests.repeat(1, 3)

            # sdf_loss0 = render_out['sdf_loss'].detach()
            loss_uncert = 1
            
            
            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            nll_term1 = torch.mean((1 / (2*(uncert_ests+1e-9))) * (color_fine_loss)) * 1    #1.5
            nll_term2 = 0.5*torch.mean(torch.log(uncert_ests+1e-9)) * 1     # 0.5

            # torch.mean(color_fine_loss - uncert_ests)
            loss = (nll_term1 + nll_term2) * self.nll_weight +\
                eikonal_loss * self.igr_weight +\
                mask_loss * self.mask_weight + 3
            
            
            self.optimizer.zero_grad()
            loss.backward()

            if iter_i==40000:
                self.optimizer.param_groups[1]['lr'] = 0.0
                self.optimizer.param_groups[2]['lr'] = 0.00001

            self.optimizer.step()

            
            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)



            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                if loss_uncert != None:
                    print('iter:{:8>d} loss = {}, eikonal_loss = {}, lr={} lr_uncert={}, psnr={}'.format(self.iter_step, loss, eikonal_loss, self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[1]['lr'], psnr))
                    
                    
                    loss_list.append(loss.item())
                    iter_list.append(self.iter_step)
                    term1_list.append(nll_term1.item())
                    term2_list.append(nll_term2.item())
                    psnr_list.append(psnr.item())
                    if len(loss_list)>200:
                        lists = [loss_list,iter_list,term1_list,term2_list,psnr_list]
                        lists = [lst[1:] for lst in lists]
                
                # if temp>30:     # 可视化loss_uncert时，去除异常值
                #     temp = 30
                
                

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()
            
            if self.iter_step % 1000 == 0:
                import matplotlib.pyplot as plt
                os.makedirs(os.path.join(self.base_exp_dir, 'fig'), exist_ok=True)
                plt.figure()
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.plot(iter_list,loss_list)
                path = os.path.join(self.base_exp_dir,'fig','{:0>8d}_{}.png'.format(self.iter_step, 'loss_color'))
                plt.savefig(path)

                plt.figure()
                plt.plot(iter_list,psnr_list)
                path = os.path.join(self.base_exp_dir,'fig','{:0>8d}_{}.png'.format(self.iter_step, 'psnr'))
                plt.savefig(path)

                plt.figure()
                plt.plot(iter_list,term1_list, color='g',label='term1')
                plt.plot(iter_list,term2_list, color='r',label='term2')
                # plt.legend()

                path = os.path.join(self.base_exp_dir,'fig','{:0>8d}_{}.png'.format(self.iter_step, 'uncertainty_term'))
                plt.savefig(path)

                # plt.figure()
                # plt.plot(loss_total_list,iter_list)
                # path = os.path.join(self.base_exp_dir,'fig','{:0>8d}_{}.png'.format(self.iter_step, 'loss_total'))
                # plt.savefig(path)


    def get_image_perm(self):
        # # return torch.randperm(8)  
        # filtered_list = list(range(51, 63, 1))  
        # # filtered_list = [0,12,13,14,24,25,26,27,30,31,32]
        # rand_perm = torch.randperm(len(filtered_list))
        # shuffled_elements = [filtered_list[i] for i in rand_perm]
        # shuffled_tensor = torch.tensor(shuffled_elements)
        return torch.randperm(self.dataset.n_images)    

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for i,g in enumerate(self.optimizer.param_groups):
            if i==0:
                g['lr'] = self.learning_rate * learning_factor
            else:
                g['lr'] =  0.0001

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.ckpt_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        if 'uncert_network_fine' in checkpoint:
            self.uncert_network.load_state_dict(checkpoint['uncert_network_fine'])
        if 'feat_network' in checkpoint:
            self.feat_network.load_state_dict(checkpoint['feat_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'uncert_network_fine': self.uncert_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
            'feat_network': self.feat_network.state_dict()
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1, only_normals=False, pose=None):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        if resolution_level < 0:
                resolution_level = self.validate_resolution_level

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'uncerts'), exist_ok=True)

        os.makedirs(os.path.join(self.base_exp_dir, 'images'), exist_ok=True)
    
        for i in range(36,70,1):        # range(44,45,4)
            idx = i
            # idx = np.random.randint(self.dataset.n_images)

            print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

            if pose is None:
                rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
            else:
                rays_o, rays_d, uv = self.dataset.gen_rays_visu(idx, pose, resolution_level=resolution_level)

            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
            rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

            out_rgb_fine = []
            out_normal_fine = []
            out_uncert_fine = []
            out_depth = []
            # feature_uncert = []
            # out_weight = []

            for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
                background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
    
                render_out = self.renderer.render(rays_o_batch,
                                                rays_d_batch,
                                                near,
                                                far,
                                                image_feat=None,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                background_rgb=background_rgb)

                def feasible(key): return (key in render_out) and (render_out[key] is not None)

                if feasible('color_fine'):
                    out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                if feasible('gradients') and feasible('weights'):
                    n_samples = self.renderer.n_samples + self.renderer.n_importance
                    normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                    if feasible('inside_sphere'):
                        normals = normals * render_out['inside_sphere'][..., None]
                    normals = normals.sum(dim=1).detach().cpu().numpy()
                    out_normal_fine.append(normals)

                    weights = render_out['weights'].sum(dim=1).detach().cpu().numpy()
                    out_depth.append(weights)
                    
                
                out_uncert_fine.append(render_out['uncert_map'].detach().cpu().numpy()) # 归一化后的uncertainty
                

                del render_out
            
            
            img_fine = None
            if len(out_rgb_fine) > 0:
                img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

            normal_img = None
            if len(out_normal_fine) > 0:
                normal_img = np.concatenate(out_normal_fine, axis=0)
                rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())      # 旋转矩阵  这里是以第一帧为基础乘上旋转矩阵，实际上不乘旋转矩阵得到的是第0个位姿的normal
                normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
            
            if len(out_uncert_fine) > 0:
                uncert_img = np.concatenate(out_uncert_fine, axis=0).reshape([H, W, -1])     # ！！！！！！！
                np.save(os.path.join(self.base_exp_dir,'uncerts','{:0>8d}_{}_{}_uncert_float.npy'.format(self.iter_step, 1, idx)),uncert_img)
                if uncert_img.shape[-1] == 3:
                    uncert_img = cv.cvtColor(uncert_img, cv.COLOR_RGB2GRAY)
                # uncert_img = (255*np.clip(uncert_img,0,1)).astype(np.uint8)
            

            for i in range(img_fine.shape[-1]):
                if len(out_rgb_fine) > 0:
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'validations_fine',
                                            '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                            np.concatenate([img_fine[..., i],
                                            self.dataset.image_at(idx, resolution_level=resolution_level)]))
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'images',
                                            '{:0>3d}.png'.format(idx)), img_fine[..., i])

                if len(out_normal_fine) > 0:
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'normals',
                                            '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                            normal_img[..., i])
                if len(out_uncert_fine) > 0:
                    # plt.axis('off')
                    plt.imshow(uncert_img, cmap="magma")
                    plt.savefig(os.path.join(self.base_exp_dir,'uncerts','{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)), bbox_inches='tight', pad_inches=0)
                    # np.save(os.path.join(self.base_exp_dir,'uncerts','{:0>8d}_{}_{}_uncert.npy'.format(self.iter_step, i, idx)),uncert_img)

                    # plt.imshow(depth_img, cmap="magma")
                    # plt.savefig(os.path.join(self.base_exp_dir,'uncerts','{:0>8d}_{}_{}_Depth.png'.format(self.iter_step, i, idx)), bbox_inches='tight', pad_inches=0)

                    # mask = self.dataset.mask_at(idx, resolution_level=resolution_level)
                    
                    diff = np.abs(self.dataset.image_at(idx, resolution_level=resolution_level)-img_fine[..., i])
                    diff = cv.cvtColor(diff, cv.COLOR_RGB2GRAY)
                    
                    plt.imshow(diff, cmap="magma")
                    plt.savefig(os.path.join(self.base_exp_dir,'uncerts','{:0>8d}_{}_{}_diff.png'.format(self.iter_step, i, idx)), bbox_inches='tight', pad_inches=0)
                    np.save(os.path.join(self.base_exp_dir,'uncerts','{:0>8d}_{}_{}_diff.npy'.format(self.iter_step, i, idx)), diff)
                # if len(out_depth) > 0:
                #     depth_img = np.concatenate(out_depth, axis=0).reshape([H, W, -1])
                #     # plt.axis('off')
                #     plt.imshow(depth_img, cmap="magma")
                #     plt.savefig(os.path.join(self.base_exp_dir,'uncerts','{:0>8d}_{}_{}_depth.png'.format(self.iter_step, i, idx)), bbox_inches='tight', pad_inches=0)



    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
            

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              image_feat=None,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=512, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/womask.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='zobject')   # bmvs_jade_occ thin_catbus

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)      # 初始化

    if args.mode == 'train':
        start_time = time.time()
        runner.train()
        end_time = time.time()
        print(end_time-start_time)
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
