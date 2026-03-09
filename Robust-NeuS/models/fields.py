import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale        # torch.Size([32768, 3])
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs) # torch.Size([32768, 39])  L=6

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]   # torch.Size([32768, 1])

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)      # pts  torch.Size([65536, 3])
        y = self.sdf(x)     # sdf  torch.Size([65536, 1])
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]        # torch.Size([65536, 3])
        return gradients.unsqueeze(1)


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        if self.mode == 'idr':
            first_dim=289 
        elif self.mode == 'no_view_dir':
            first_dim=262


        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)
        dims[0] = first_dim

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)  # feature_vectors 
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x      # torch.Size([65536, 3]), torch.Size([65536, 256])


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            outputs = self.output_linear(h)
            return outputs[:,3], outputs[:,:3]


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val))) # nn.Parameter函数将 init_val 转为一个可学习参数，并命名为variance

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)


class UncertaintyNetwork(nn.Module):
    def __init__(self, W, output_ch, multires_view=0):
        super().__init__()
        self.embedview_fn = None
        if multires_view == 4:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            self.wfeature = nn.Sequential(nn.Linear(6+input_ch, W), nn.ReLU(inplace=False))
            self.wofeature = nn.Sequential(nn.Linear(3+input_ch, W), nn.ReLU(inplace=False))
        else:
            self.wfeature = nn.Sequential(nn.Linear(9, W), nn.ReLU(inplace=False))
            self.wofeature = nn.Sequential(nn.Linear(6, W), nn.ReLU(inplace=False))

        

        self.transient_encoding = nn.Sequential(
                                        nn.Linear(W, W), nn.ReLU(inplace=False),            # 256，128
                                        nn.Linear(W, W//2), nn.ReLU(inplace=False),
                                        nn.Linear(W//2, W//2), nn.ReLU(inplace=False),
                                        nn.Linear(W//2, W//2), nn.ReLU(inplace=False))
        # transient output layers
        # self.transient_sigma = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())
        # self.transient_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
        # mean = 0.1
        # std = 0.059 
        # bias = 0.01

        # for layer in self.transient_encoding:
        #     if isinstance(layer, nn.Linear):
        #         torch.nn.init.normal_(layer.weight, mean=mean, std=std)
        #         torch.nn.init.constant_(layer.bias, -bias)

        self.transient_beta = nn.Sequential(nn.Linear(W//2, output_ch), nn.Softplus(beta=100))   # nn.Softplus  最后的输出维度为3？
        

    def forward(self, color, view_dirs, image_feat):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
        if image_feat == None:
            uncert_input = torch.cat([color, view_dirs], dim=1)
            x = self.wofeature(uncert_input)
        else:
            image_feat = image_feat.T.unsqueeze(0).unsqueeze(-1)
            image_feat = F.interpolate(image_feat, size=(512,1), mode='bilinear', align_corners=False).squeeze().T
            uncert_input = torch.cat([color, view_dirs, image_feat], dim=1)
            x = self.wfeature(uncert_input)

        x = self.transient_encoding(x)

        # sigma = self.transient_sigma(x)
        # rgb = self.transient_rgb(x)         # render出一个不准确的rgb吗？
        beta = self.transient_beta(x) + 0.001       # torch.Size([65536, 1])  self.beta_min=0.1  0.000001

        # 是否要concat到一起?
        # transient = torch.cat([transient_rgb, transient_sigma, transient_beta], 1) # (B, 5)
        # return torch.cat([static, transient], 1) # (B, 9)


        return beta # sigma, rgb, beta     # beta

class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.conv1 = myconv2d(batchNorm, c_in, 64,  k=3, stride=1, pad=1)
        self.conv2 = myconv2d(batchNorm, 64,   128, k=3, stride=2, pad=1)
        self.conv3 = myconv2d(batchNorm, 128,  128, k=3, stride=2, pad=1)
        self.conv4 = myconv2d(batchNorm, 128,  256, k=3, stride=2, pad=1)
        self.conv5 = myconv2d(batchNorm, 256,  256, k=3, stride=2, pad=1)
        self.conv6 = myconv2d(batchNorm, 256,  128, k=3, stride=2, pad=1)
        self.conv7 = myconv2d(batchNorm, 128, 128, k=3, stride=2, pad=1)
        self.conv8 = myconv2d(batchNorm, 128, 64, k=3, stride=2, pad=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(2, 2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.avg_pool(out)
        out_feat = self.conv8(out)

        n, c, h, w = out_feat.data.shape
        out_feat   = out_feat.reshape(n, -1)
        return out_feat

def myconv2d(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = (k - 1) // 2 if pad < 0 else pad
    # print('Conv pad = %d' % (pad))
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.1, inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
                )