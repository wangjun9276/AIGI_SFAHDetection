from loralib import layers as lora_layers
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
# from main import device
# import resnet_npr
# import math
# import argparse
import kornia
import torch.nn as nn
# from PIL import Image
# import ipdb
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from xception import Xception
# from torchvision.transforms import transforms
# from srm_conv import SRMConv2d_simple
# import torch.nn.init as init
import os
# from torchkeras import summary
from torch import nn
import clip
# import ipdb
# from torch.cuda.amp import autocast, GradScaler
from vit import Transformer
from gmm import On_attention_gaussian_mask
from FacNet_layer import MultiSpectralAttentionLayer

device = torch.device('cuda:1')


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.1):
        '''dim is the length of the input sequences'''

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias) 
        self.kv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape           # [Batchsize (B) x num_patch (N) x embed_size (C)]

        # Q matrix [B x N x C] ----> [B x N x h x (C/h)] ----> [B x h x N x S]; S = C/h
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 

        # We use a reduction technique to reduce the computational complex of 
        # [B x N x C] ----> [B x N/2 x 2 x h x S] ----> [2 x B x h x N/2 x S]
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        k, v = kv[0], kv[1] # [B x h x N/2 x S], [B x h x N/2 x S]

        # Calculate attention weight [B x h x N x S] x [B x h x S x N/2] = [B x h x N x N/2]
        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Calculate attention [B x h x N x N/2] x [B x h x N/2 x S] = [B x h x N x S] 
        # [B x h x N x S] ----> [B x N x h x S] ----> [B x N x (hxS)] = [B x N x C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.1):
        '''dim is the length of the input sequences'''

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias) 
        self.kv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
        # [Batchsize (B) x embed_size (C)]
        B, N, C = q.shape

        # Q matrix [B x N x C] ----> [B x N x h x (C/h)] ----> [B x h x N x S]; S = C/h
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Calculate attention weight [B x h x N x S] x [B x h x S x N/2] = [B x h x N x N/2]
        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 

        # Calculate attention [B x h x N x N/2] x [B x h x N/2 x S] = [B x h x N x S] 
        # [B x h x N x S] ----> [B x N x h x S] ----> [B x N x (hxS)] = [B x N x C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def npr(image: torch.Tensor):
    n, c, w, h = image.shape
    if -1 * w % 2 != 0:
        image = image[:, :, :w % 2 * -1, :]
    if -1 * h % 2 != 0:
        image = image[:, :, :, :h % 2 * -1]

    n, c, w, h = image.shape
    if w % 2 == 1:
        image = image[:, :, :-1, :]
    if h % 2 == 1:
        image = image[:, :, :, :-1]

    factor = 0.5
    NPR = F.interpolate(image, scale_factor=factor,
                        mode='nearest', recompute_scale_factor=True)
    NPR = F.interpolate(NPR, scale_factor=1 / factor,
                        mode='nearest', recompute_scale_factor=True)
    NPR = image - NPR
    image = NPR * 2.0 / 3.0
    return image


clip_model, preprocess = clip.load("/media/sdb/wangjun/tifs_zj/clip/ViT-L-14.pt", device=device)


class xception_sobel_pass_npr(nn.Module):
    def __init__(self):
        super(xception_sobel_pass_npr, self).__init__()
        self.backbone = self.build_backbone()

    def build_backbone(self):
        model_config = {"mode": "original",
                        "num_classes": 1, "inc": 3, "dropout": False}
        backbone = Xception(model_config)

        # 检查本地是否存在预训练模型文件
        model_path = "./xception-b5690688.pth"
        if not os.path.exists(model_path):
            print("正在从Hugging Face下载预训练模型...")
            url = "https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/resolve/main/xception-b5690688.pth?download=true"
            torch.hub.download_url_to_file(url, model_path)
            print("预训练模型下载完成！")

        # 加载预训练模型
        state_dict = torch.load(model_path)
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        return backbone

    def forward(self, x: torch.Tensor, return_feature=False):
        sobel_feature = kornia.filters.Sobel()(x)
        sobel_feature = sobel_feature + x
        # sobel_feature = check(sobel_feature)
        npr_feature = npr(sobel_feature)
        feature = self.backbone.features(npr_feature)
        if return_feature:
            # feature = nn.ReLU(inplace=True)(feature)
            # feature = F.adaptive_avg_pool2d(feature, (1, 1))
            # feature = feature.view(feature.size(0), -1)
            return feature
        pred = self.backbone.classifier(feature)
        return pred, feature


class xception_sobel_pass_npr_student(nn.Module):
    def __init__(self):
        super(xception_sobel_pass_npr_student, self).__init__()
        self.backbone = self.build_backbone()

    def build_backbone(self):
        model_config = {"mode": "original",
                        "num_classes": 1, "inc": 3, "dropout": False}
        backbone = Xception(model_config)

        # 检查本地是否存在预训练模型文件
        model_path = "./xception-b5690688.pth"
        if not os.path.exists(model_path):
            print("正在从Hugging Face下载预训练模型...")
            url = "https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/resolve/main/xception-b5690688.pth?download=true"
            torch.hub.download_url_to_file(url, model_path)
            print("预训练模型下载完成！")

        # 加载预训练模型
        state_dict = torch.load(model_path)
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)

        # # 加载预训练模型
        # state_dict = torch.load('./checkpoints/fdmas_xception_sobel_pass_npr_student_0.7994061907493891.pt')
        # for name, weights in state_dict.items():
        #     if 'pointwise' in name:
        #         state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        # state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        # backbone.load_state_dict(state_dict, False)
        return backbone

    def forward(self, x: torch.Tensor, return_feature=False):
        sobel_feature = kornia.filters.Sobel()(x)
        sobel_feature = sobel_feature + x
        # sobel_feature = check(sobel_feature)
        npr_feature = npr(sobel_feature)
        feature = self.backbone.features(npr_feature)
        if return_feature:
            # feature = nn.ReLU(inplace=True)(feature)
            # feature = F.adaptive_avg_pool2d(feature, (1, 1))
            # feature = feature.view(feature.size(0), -1)
            return feature
        pred = self.backbone.classifier(feature)
        return pred, feature


class xception_dwt(nn.Module):
    def __init__(self):
        super(xception_dwt, self).__init__()
        self.backbone = self.build_backbone()

    def build_backbone(self):
        model_config = {"mode": "original",
                        "num_classes": 1, "inc": 3, "dropout": False}
        backbone = Xception(model_config)

        # 检查本地是否存在预训练模型文件
        model_path = "./xception-b5690688.pth"
        if not os.path.exists(model_path):
            print("正在从Hugging Face下载预训练模型...")
            url = "https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/resolve/main/xception-b5690688.pth?download=true"
            torch.hub.download_url_to_file(url, model_path)
            print("预训练模型下载完成！")

        # 加载预训练模型
        state_dict = torch.load(model_path)
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        return backbone

    def _preprocess_dwt(self, x, mode='symmetric', wave='bior1.3'):
        '''
        pip install pywavelets pytorch_wavelets
        '''
        from pytorch_wavelets import DWTForward, DWTInverse
        DWT_filter = DWTForward(J=1, mode=mode, wave=wave).to(x.device)
        LL, Yh = DWT_filter(x)
        LH, HL, HH = Yh[0][:, :, 0, :], Yh[0][:, :, 1, :], Yh[0][:, :, 2, :]
        return LL, LH, HL, HH

    def forward(self, x: torch.Tensor, return_feature=False):
        _, _, _, x = self._preprocess_dwt(x)
        feature = self.backbone.features(x)
        if return_feature:
            feature = nn.ReLU(inplace=True)(feature)
            feature = F.adaptive_avg_pool2d(feature, (1, 1))
            feature = feature.view(feature.size(0), -1)
            return feature
        pred = self.backbone.classifier(feature)
        return pred


class clip_lora_eeFrozen(nn.Module):
    def __init__(self):
        super(clip_lora_eeFrozen, self).__init__()
        # 只在初始化时加载一次模型
        self.xception_sobel_pass_npr = xception_sobel_pass_npr()
        # 检查本地是否存在预训练模型文件
        model_path = "./checkpoints/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt"
        if not os.path.exists(model_path):
            print("正在从Hugging Face下载预训练模型...")
            url = "https://huggingface.co/jzousz/pt/resolve/main/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt?download=true"
            torch.hub.download_url_to_file(url, model_path)
            print("预训练模型下载完成！")

        self.xception_sobel_pass_npr.load_state_dict(torch.load(model_path,
                                                                map_location=device))
        # self.xception_dwt = xception_dwt()
        # self.xception_dwt.to(device)
        self.xception_sobel_pass_npr.to(device)
        # 确保 xception_sobel_pass_npr 使用 float32 精度
        self.xception_sobel_pass_npr = self.xception_sobel_pass_npr.float()
        # self.xception_dwt = self.xception_dwt.float()
        # 冻结所有参数
        for p in self.xception_sobel_pass_npr.parameters():
            p.requires_grad = False

        self.clip_model, _ = clip.load("/media/sdb/wangjun/tifs_zj/clip/ViT-L-14.pt")
        # 确保 CLIP 模型使用 float32 精度
        self.clip_model = self.clip_model.float()
        self.clip_model.eval()

        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        self.args = Args(
            encoder="vision",
            backbone="ViT-L/14",
            position="half-up",
            params="qkv",
            r=4,
            alpha=0.5,
            dropout_rate=0.0,
        )
        # print('Apply Lora to half top blocks')
        list_lora_layers = apply_lora(self.args, self.clip_model)
        mark_only_lora_as_trainable(self.clip_model)
        self.clip_model = self.clip_model.to(device)
        self.clip_encode_image = self.clip_model.encode_image

        self.project = nn.Linear(768, 512)
        self.map = nn.Linear(2048, 512)
        # self.dwt_map = nn.Linear(2048, 512)

        # 使用concat方式，输入维度为512*3
        # self.head = nn.Linear(512 * 3, 1)
        self.head = nn.Linear(512 * 2, 1)
        init_weights(self.project)
        init_weights(self.head)
        # init_weights(self.dwt_map)

        # 确保所有层使用 float32 精度
        self.project = self.project.float()
        self.head = self.head.float()
        self.map = self.map.float()
        # self.dwt_map = self.dwt_map.float()

    def forward(self, x: torch.Tensor):
        # 确保输入数据是 float32 精度
        x = x.float()

        # 使用 float32 精度的输入进行 xception 特征提取
        ee_features = self.xception_sobel_pass_npr.forward(
            x, return_feature=True)
        ee_features = self.map(ee_features)

        # 提取DWT特征
        # dwt_features = self.xception_dwt.forward(x, return_feature=True)
        # dwt_features = self.dwt_map(dwt_features)

        # 使用相同的 float32 精度输入进行 CLIP 特征提取
        clip_feature = self.clip_encode_image(x)
        clip_feature = self.project(clip_feature)

        # 使用concat方式融合特征
        # combined_feature = torch.cat(
        #     [clip_feature, ee_features, dwt_features], dim=-1)
        combined_feature = torch.cat(
            [clip_feature, ee_features], dim=-1)
        pred = self.head(combined_feature)

        return pred, torch.tensor(0.0, dtype=torch.float32, device=device)


class Xception_GMM(nn.Module):
    def __init__(self):
        super(Xception_GMM, self).__init__()
        # 只在初始化时加载一次模型
        self.xception_sobel_pass_npr = xception_sobel_pass_npr()
        # 检查本地是否存在预训练模型文件
        model_path = "./checkpoints/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt"
        if not os.path.exists(model_path):
            print("正在从Hugging Face下载预训练模型...")
            url = "https://huggingface.co/jzousz/pt/resolve/main/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt?download=true"
            torch.hub.download_url_to_file(url, model_path)
            print("预训练模型下载完成！")

        self.xception_sobel_pass_npr.load_state_dict(torch.load(model_path,
                                                                map_location=device))
                                                                
        self.xception_sobel_pass_npr.to(device)
        # 确保 xception_sobel_pass_npr 使用 float32 精度
        self.xception_sobel_pass_npr = self.xception_sobel_pass_npr.float()
        
        # 冻结所有参数
        for p in self.xception_sobel_pass_npr.parameters():
            p.requires_grad = False
        
        self.mask = nn.Parameter(On_attention_gaussian_mask(49), requires_grad=False)
        self.transformer = Transformer(dim=2048, num_patches=49, depth=2, heads=4, dim_head=256//4, mlp_dim_ratio=2, num_kernals=5, dropout=0.25, 
                                       stochastic_depth=0.1, is_GMM=True, is_SLM=False, mask=self.mask)
                                       
        init_weights(self.transformer)

        # 确保所有层使用 float32 精度
        self.transformer = self.transformer.float()

    def forward(self, x: torch.Tensor):
        # 确保输入数据是 float32 精度
        x = x.float()

        # 使用 float32 精度的输入进行 xception 特征提取
        ee_features = self.xception_sobel_pass_npr.forward(
            x, return_feature=True)
        
        # # original code for xception sobel npr classification
        ee_features0 = nn.ReLU(inplace=True)(ee_features)
        ee_features0 = F.adaptive_avg_pool2d(ee_features0, (1, 1))
        ee_features0 = ee_features0.view(ee_features0.size(0), -1)
        # ee_features = self.map(ee_features)

        # two strategies for feature fusion
        # 1. concat
        # 2. the output of B x 2048 x 7 x 7, --> B x 2048 x 49, --> B x 49 x 2048, 
        # --> B x 49 x 768, --> transformer with gaussian mixture mask, --> B x 49 x 768, 
        # --> B x 1 x 768, concat with 2048, --> classifier
        # 3. GMM adapter in clip half-up (half-bottom may be better) + LoRA in clip half-up, 
        # --> concat with 2048, --> classifier
        ee_features = ee_features.view(ee_features.size(0), 2048, -1).permute(0,2,1)
        # ee_features = self.dropout(ee_features)
        ee_features = self.transformer(ee_features).mean(dim=1)
        # ee_features = nn.ReLU(inplace=True)(ee_features)
        pred = self.xception_sobel_pass_npr.backbone.classifier(ee_features + ee_features0)

        return pred, torch.tensor(0.0, dtype=torch.float32, device=device)


class Xception_GMM2(nn.Module):
    def __init__(self):
        super(Xception_GMM2, self).__init__()
        # 只在初始化时加载一次模型
        self.xception_sobel_pass_npr = xception_sobel_pass_npr()
        # 检查本地是否存在预训练模型文件
        model_path = "./checkpoints/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt"
        if not os.path.exists(model_path):
            print("正在从Hugging Face下载预训练模型...")
            url = "https://huggingface.co/jzousz/pt/resolve/main/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt?download=true"
            torch.hub.download_url_to_file(url, model_path)
            print("预训练模型下载完成！")

        self.xception_sobel_pass_npr.load_state_dict(torch.load(model_path,
                                                                map_location=device))
                                                                
        self.xception_sobel_pass_npr.to(device)
        # 确保 xception_sobel_pass_npr 使用 float32 精度
        self.xception_sobel_pass_npr = self.xception_sobel_pass_npr.float()
        
        # 冻结所有参数
        for p in self.xception_sobel_pass_npr.parameters():
            p.requires_grad = False
        
        self.mask = nn.Parameter(On_attention_gaussian_mask(49), requires_grad=False)
        self.transformer = Transformer(dim=2048, num_patches=49, depth=2, heads=8, dim_head=256//4, mlp_dim_ratio=2, num_kernals=5, dropout=0.25, 
                                       stochastic_depth=0.1, is_GMM=True, is_SLM=False, mask=self.mask)
        self.projector = nn.Linear(2048, 1024)
        self.projector_gmm = nn.Linear(2048, 1024)
                                       
        init_weights(self.transformer)
        init_weights(self.projector)
        init_weights(self.projector_gmm)

        # 确保所有层使用 float32 精度
        self.transformer = self.transformer.float()
        self.projector = self.projector.float()
        self.projector_gmm = self.projector_gmm.float()

    def forward(self, x: torch.Tensor):
        # 确保输入数据是 float32 精度
        x = x.float()

        # 使用 float32 精度的输入进行 xception 特征提取
        ee_features = self.xception_sobel_pass_npr.forward(
            x, return_feature=True)
        
        # # original code for xception sobel npr classification
        ee_features0 = nn.ReLU(inplace=True)(ee_features)
        ee_features0 = F.adaptive_avg_pool2d(ee_features0, (1, 1))
        ee_features0 = ee_features0.view(ee_features0.size(0), -1)
        ee_features0 = self.projector(ee_features0)

        # two strategies for feature fusion
        # 1. concat
        # 2. the output of B x 2048 x 7 x 7, --> B x 2048 x 49, --> B x 49 x 2048, 
        # --> B x 49 x 768, --> transformer with gaussian mixture mask, --> B x 49 x 768, 
        # --> B x 1 x 768, concat with 2048, --> classifier
        # 3. GMM adapter in clip half-up (half-bottom may be better) + LoRA in clip half-up, 
        # --> concat with 2048, --> classifier
        ee_features = ee_features.view(ee_features.size(0), 2048, -1).permute(0,2,1)
        ee_features = self.transformer(ee_features).mean(dim=1)
        ee_features = self.projector_gmm(ee_features)
        pred = self.xception_sobel_pass_npr.backbone.classifier(torch.cat([ee_features, ee_features0], dim=1))

        return pred, torch.tensor(0.0, dtype=torch.float32, device=device)


class Xception_DCT(nn.Module):
    def __init__(self):
        super(Xception_DCT, self).__init__()
        # 只在初始化时加载一次模型
        self.xception_sobel_pass_npr = xception_sobel_pass_npr()
        # 检查本地是否存在预训练模型文件
        model_path = "./checkpoints/xception_sobel_pass_npr/fdmas_xception_sobel_pass_npr_class4_0.9288116083373297.pt"
        if not os.path.exists(model_path):
            print("正在从Hugging Face下载预训练模型...")
            url = "https://huggingface.co/jzousz/pt/resolve/main/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt?download=true"
            torch.hub.download_url_to_file(url, model_path)
            print("预训练模型下载完成！")

        self.xception_sobel_pass_npr.load_state_dict(torch.load(model_path,
                                                                map_location=device))
                                                                
        self.xception_sobel_pass_npr.to(device)
        # 确保 xception_sobel_pass_npr 使用 float32 精度
        self.xception_sobel_pass_npr = self.xception_sobel_pass_npr.float()
        
        # 冻结所有参数
        for p in self.xception_sobel_pass_npr.parameters():
            p.requires_grad = False
        
        self.att = MultiSpectralAttentionLayer(2048, 7, 7,  reduction=4, freq_sel_method = 'bot16')

        self.projector = nn.Linear(2048, 1024)
        self.projector_dct = nn.Linear(2048, 1024)
        
        init_weights(self.projector)
        init_weights(self.projector_dct)

        # 确保所有层使用 float32 精度
        # self.transformer = self.transformer.float()
        self.projector = self.projector.float()
        self.projector_dct = self.projector_dct.float()

    def forward(self, x: torch.Tensor):
        # 确保输入数据是 float32 精度
        x = x.float()

        # 使用 float32 精度的输入进行 xception 特征提取
        ee_features = self.xception_sobel_pass_npr.forward(
            x, return_feature=True)
        
        # # original code for xception sobel npr classification
        ee_features0 = nn.ReLU(inplace=True)(ee_features)
        ee_features0 = F.adaptive_avg_pool2d(ee_features0, (1, 1))
        ee_features0 = ee_features0.view(ee_features0.size(0), -1)
        ee_features0 = self.projector(ee_features0)

        # two strategies for feature fusion
        # 1. concat
        # 2. the output of B x 2048 x 7 x 7, --> B x 2048 x 49, --> B x 49 x 2048, 
        # --> B x 49 x 768, --> transformer with gaussian mixture mask, --> B x 49 x 768, 
        # --> B x 1 x 768, concat with 2048, --> classifier
        # 3. GMM adapter in clip half-up (half-bottom may be better) + LoRA in clip half-up, 
        # --> concat with 2048, --> classifier
        ee_features = self.att(ee_features)
        ee_features = nn.ReLU(inplace=True)(ee_features)
        ee_features = F.adaptive_avg_pool2d(ee_features, (1, 1))
        ee_features = ee_features.view(ee_features.size(0), -1)
        ee_features = self.projector_dct(ee_features)
        pred = self.xception_sobel_pass_npr.backbone.classifier(torch.cat([ee_features, ee_features0], dim=1))

        return pred, torch.cat([ee_features, ee_features0], dim=1)


class Xception_DCTMoe(nn.Module):
    def __init__(self):
        super(Xception_DCTMoe, self).__init__()
        # 只在初始化时加载一次模型
        self.xception_sobel_pass_npr = xception_sobel_pass_npr()
        # 检查本地是否存在预训练模型文件
        model_path = "./checkpoints/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt"
        if not os.path.exists(model_path):
            print("正在从Hugging Face下载预训练模型...")
            url = "https://huggingface.co/jzousz/pt/resolve/main/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt?download=true"
            torch.hub.download_url_to_file(url, model_path)
            print("预训练模型下载完成！")

        self.xception_sobel_pass_npr.load_state_dict(torch.load(model_path,
                                                                map_location=device))
                                                                
        self.xception_sobel_pass_npr.to(device)
        # 确保 xception_sobel_pass_npr 使用 float32 精度
        self.xception_sobel_pass_npr = self.xception_sobel_pass_npr.float()
        
        # 冻结所有参数
        for p in self.xception_sobel_pass_npr.parameters():
            p.requires_grad = False
        
        self.att = MultiSpectralAttentionLayer(2048, 7, 7,  reduction=4, freq_sel_method = 'bot16')

        self.projector = nn.Linear(2048, 1024)
        self.projector_dct = nn.Linear(2048, 1024)
        
        init_weights(self.projector)
        init_weights(self.projector_dct)

        # 确保所有层使用 float32 精度
        # self.transformer = self.transformer.float()
        self.projector = self.projector.float()
        self.projector_dct = self.projector_dct.float()

    def forward(self, x: torch.Tensor):
        # 确保输入数据是 float32 精度
        x = x.float()

        # 使用 float32 精度的输入进行 xception 特征提取
        ee_features = self.xception_sobel_pass_npr.forward(
            x, return_feature=True)
        
        # # original code for xception sobel npr classification
        ee_features0 = nn.ReLU(inplace=True)(ee_features)
        ee_features0 = F.adaptive_avg_pool2d(ee_features0, (1, 1))
        ee_features0 = ee_features0.view(ee_features0.size(0), -1)
        ee_features0 = self.projector(ee_features0)

        # two strategies for feature fusion
        # 1. concat
        # 2. the output of B x 2048 x 7 x 7, --> B x 2048 x 49, --> B x 49 x 2048, 
        # --> B x 49 x 768, --> transformer with gaussian mixture mask, --> B x 49 x 768, 
        # --> B x 1 x 768, concat with 2048, --> classifier
        # 3. GMM adapter in clip half-up (half-bottom may be better) + LoRA in clip half-up, 
        # --> concat with 2048, --> classifier
        ee_features = self.att(ee_features)
        ee_features = nn.ReLU(inplace=True)(ee_features)
        ee_features = F.adaptive_avg_pool2d(ee_features, (1, 1))
        ee_features = ee_features.view(ee_features.size(0), -1)
        ee_features = self.projector_dct(ee_features)
        pred = self.xception_sobel_pass_npr.backbone.classifier(torch.cat([ee_features, ee_features0], dim=1))

        return pred, torch.tensor(0.0, dtype=torch.float32, device=device)


class clip_lora_eeFrozen_GMM(nn.Module):
    def __init__(self):
        super(clip_lora_eeFrozen_GMM, self).__init__()
        # 只在初始化时加载一次模型
        self.xception_sobel_pass_npr = xception_sobel_pass_npr()
        # 检查本地是否存在预训练模型文件
        model_path = "./checkpoints/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt"
        if not os.path.exists(model_path):
            print("正在从Hugging Face下载预训练模型...")
            url = "https://huggingface.co/jzousz/pt/resolve/main/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt?download=true"
            torch.hub.download_url_to_file(url, model_path)
            print("预训练模型下载完成！")

        self.xception_sobel_pass_npr.load_state_dict(torch.load(model_path,
                                                                map_location=device))
        # self.xception_dwt = xception_dwt()
        # self.xception_dwt.to(device)
        self.xception_sobel_pass_npr.to(device)
        # 确保 xception_sobel_pass_npr 使用 float32 精度
        self.xception_sobel_pass_npr = self.xception_sobel_pass_npr.float()
        # self.xception_dwt = self.xception_dwt.float()
        # 冻结所有参数
        for p in self.xception_sobel_pass_npr.parameters():
            p.requires_grad = False

        self.clip_model, _ = clip.load("/media/sdb/wangjun/tifs_zj/clip/ViT-L-14.pt")
        # 确保 CLIP 模型使用 float32 精度
        self.clip_model = self.clip_model.float()
        self.clip_model.eval()
        
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        self.args = Args(
            encoder="vision",
            backbone="ViT-L/14",
            position="half-up",
            params="qkv",
            r=4,
            alpha=0.5,
            dropout_rate=0.0,
        )
        # print('Apply Lora to half top blocks')
        list_lora_layers = apply_lora(self.args, self.clip_model)
        mark_only_lora_as_trainable(self.clip_model)
        self.clip_model = self.clip_model.to(device)
        self.clip_encode_image = self.clip_model.encode_image
        
        # zou's projector
        self.project = nn.Linear(768, 512)
        self.map = nn.Linear(2048, 512)
        # self.dwt_map = nn.Linear(2048, 512)

        # jun improvement
        self.dropout = nn.Dropout(0.25)
        # self.projector = nn.Linear(2048, 768)
        
        self.mask = nn.Parameter(On_attention_gaussian_mask(49), requires_grad=False)
        self.transformer = Transformer(dim=2048, num_patches=49, depth=2, heads=4, dim_head=256//4, mlp_dim_ratio=2, num_kernals=5, dropout=0.25, 
                                       stochastic_depth=0.1, is_GMM=True, is_SLM=False, mask=self.mask)

        # 使用concat方式，输入维度为512*3
        # self.head = nn.Linear(512 * 3, 1)
        self.head = nn.Linear(512 * 2, 1)
        init_weights(self.project)
        init_weights(self.head)
        # init_weights(self.dwt_map)

        # 确保所有层使用 float32 精度
        self.project = self.project.float()
        self.head = self.head.float()
        self.map = self.map.float()
        self.transformer = self.transformer.float()

    def forward(self, x: torch.Tensor):
        # 确保输入数据是 float32 精度
        x = x.float()

        # 使用 float32 精度的输入进行 xception 特征提取
        ee_features = self.xception_sobel_pass_npr.forward(
            x, return_feature=True)
        
        # # original code for xception sobel npr classification
        ee_features0 = nn.ReLU(inplace=True)(ee_features)
        ee_features0 = F.adaptive_avg_pool2d(ee_features0, (1, 1))
        ee_features0 = ee_features0.view(ee_features0.size(0), -1)
        # ee_features = self.map(ee_features)

        # two strategies for feature fusion
        # 1. concat
        # 2. the output of B x 2048 x 7 x 7, --> B x 2048 x 49, --> B x 49 x 2048, 
        # --> B x 49 x 768, --> transformer with gaussian mixture mask, --> B x 49 x 768, 
        # --> B x 1 x 768, concat with 2048, --> classifier
        # 3. GMM adapter in clip half-up (half-bottom may be better) + LoRA in clip half-up, 
        # --> concat with 2048, --> classifier
        ee_features = ee_features.view(ee_features.size(0), 2048, -1).permute(0,2,1)
        ee_features = self.dropout(ee_features)
        ee_features = self.transformer(ee_features).mean(dim=1)
        # ee_features = self.projector(ee_features)
        # print(ee_features.shape, ee_features0.shape)
        ee_features = self.map(ee_features + ee_features0)
        # ee_features = self.map(ee_features)

        # 使用相同的 float32 精度输入进行 CLIP 特征提取
        clip_feature = self.clip_encode_image(x)
        clip_feature = self.project(clip_feature)

        # 使用concat方式融合特征
        combined_feature = torch.cat(
            [clip_feature, ee_features], dim=-1)
        pred = self.head(combined_feature)

        return pred, torch.tensor(0.0, dtype=torch.float32, device=device)


class clip_lora_GMM(nn.Module):
    def __init__(self):
        super(clip_lora_GMM, self).__init__()
        # 只在初始化时加载一次模型
        self.xception_sobel_pass_npr = xception_sobel_pass_npr()
        # 检查本地是否存在预训练模型文件
        model_path = "./checkpoints/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt"
        if not os.path.exists(model_path):
            print("正在从Hugging Face下载预训练模型...")
            url = "https://huggingface.co/jzousz/pt/resolve/main/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt?download=true"
            torch.hub.download_url_to_file(url, model_path)
            print("预训练模型下载完成！")

        self.xception_sobel_pass_npr.load_state_dict(torch.load(model_path,
                                                                map_location=device))
        # self.xception_dwt = xception_dwt()
        # self.xception_dwt.to(device)
        self.xception_sobel_pass_npr.to(device)
        # 确保 xception_sobel_pass_npr 使用 float32 精度
        self.xception_sobel_pass_npr = self.xception_sobel_pass_npr.float()
        # self.xception_dwt = self.xception_dwt.float()
        # 冻结所有参数
        for p in self.xception_sobel_pass_npr.parameters():
            p.requires_grad = False

        self.clip_model, _ = clip.load("/media/sdb/wangjun/tifs_zj/clip/ViT-L-14.pt")
        # 确保 CLIP 模型使用 float32 精度
        self.clip_model = self.clip_model.float()
        self.clip_model.eval()
        
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        self.args = Args(
            encoder="vision",
            backbone="ViT-L/14",
            position="half-up",
            params="qkv",
            r=4,
            alpha=0.5,
            dropout_rate=0.0,
        )
        # print('Apply Lora to half top blocks')
        list_lora_layers = apply_lora(self.args, self.clip_model)
        mark_only_lora_as_trainable(self.clip_model)
        self.clip_model = self.clip_model.to(device)
        self.clip_encode_image = self.clip_model.encode_image
        
        # zou's projector
        self.project = nn.Linear(768, 512)
        self.map = nn.Linear(2048, 512)
        # self.dwt_map = nn.Linear(2048, 512)

        # jun improvement
        self.dropout = nn.Dropout(0.25)
        # self.projector = nn.Linear(2048, 768)
        
        self.mask = nn.Parameter(On_attention_gaussian_mask(256), requires_grad=False)
        self.transformer = Transformer(dim=768, num_patches=256, depth=2, heads=4, dim_head=256//4, mlp_dim_ratio=2, num_kernals=5, dropout=0.25, 
                                       stochastic_depth=0.1, is_GMM=True, is_SLM=False, mask=self.mask)

        # 使用concat方式，输入维度为512*3
        # self.head = nn.Linear(512 * 3, 1)
        self.head = nn.Linear(512 * 2, 1)
        init_weights(self.project)
        init_weights(self.head)
        # init_weights(self.dwt_map)

        # 确保所有层使用 float32 精度
        self.project = self.project.float()
        self.head = self.head.float()
        self.map = self.map.float()
        self.transformer = self.transformer.float()

    def forward(self, x: torch.Tensor):
        # 确保输入数据是 float32 精度
        x = x.float()

        # 使用 float32 精度的输入进行 xception 特征提取
        ee_features = self.xception_sobel_pass_npr.forward(
            x, return_feature=True)
        
        # # original code for xception sobel npr classification
        ee_features = nn.ReLU(inplace=True)(ee_features)
        ee_features = F.adaptive_avg_pool2d(ee_features, (1, 1))
        ee_features = ee_features.view(ee_features.size(0), -1)
        ee_features = self.map(ee_features)

        # 使用相同的 float32 精度输入进行 CLIP 特征提取
        clip_feature, all_patches = self.clip_encode_image(x)
        all_patches = self.transformer(all_patches).mean(dim=1)
        clip_feature = self.project(clip_feature + all_patches)

        # 使用concat方式融合特征
        combined_feature = torch.cat(
            [clip_feature, ee_features], dim=-1)
        pred = self.head(combined_feature)

        return pred, torch.tensor(0.0, dtype=torch.float32, device=device)


class clip_lora_eeFrozen_DCT(nn.Module):
    def __init__(self):
        super(clip_lora_eeFrozen_DCT, self).__init__()
        # 只在初始化时加载一次模型
        self.xception_sobel_pass_npr = xception_sobel_pass_npr()
        # 检查本地是否存在预训练模型文件
        model_path = "./checkpoints/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt"
        if not os.path.exists(model_path):
            print("正在从Hugging Face下载预训练模型...")
            url = "https://huggingface.co/jzousz/pt/resolve/main/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt?download=true"
            torch.hub.download_url_to_file(url, model_path)
            print("预训练模型下载完成！")

        self.xception_sobel_pass_npr.load_state_dict(torch.load(model_path,
                                                                map_location=device))
        # self.xception_dwt = xception_dwt()
        # self.xception_dwt.to(device)
        self.xception_sobel_pass_npr.to(device)
        # 确保 xception_sobel_pass_npr 使用 float32 精度
        self.xception_sobel_pass_npr = self.xception_sobel_pass_npr.float()
        # self.xception_dwt = self.xception_dwt.float()
        # 冻结所有参数
        for p in self.xception_sobel_pass_npr.parameters():
            p.requires_grad = False

        self.att = MultiSpectralAttentionLayer(2048, 7, 7,  reduction=4, freq_sel_method = 'bot16')

        # init_weights(self.projector)
        self.clip_model, _ = clip.load("/media/sdb/wangjun/tifs_zj/clip/ViT-L-14.pt")
        # 确保 CLIP 模型使用 float32 精度
        self.clip_model = self.clip_model.float()
        self.clip_model.eval()
        
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        self.args = Args(
            encoder="vision",
            backbone="ViT-L/14",
            position="half-up",
            params="qkv",
            # params="qkv",
            r=4,
            alpha=0.5,
            dropout_rate=0.0,
        )
        # print('Apply Lora to half top blocks')
        list_lora_layers = apply_lora(self.args, self.clip_model)
        mark_only_lora_as_trainable(self.clip_model)
        self.clip_model = self.clip_model.to(device)
        self.clip_encode_image = self.clip_model.encode_image
        
        # zou's projector
        self.project = nn.Linear(768, 512)
        self.map = nn.Linear(2048, 512)
        # self.dwt_map = nn.Linear(2048, 512)

        # jun improvement
        self.dropout = nn.Dropout(0.25)
        # self.projector = nn.Linear(2048, 768)
        
        self.mask = nn.Parameter(On_attention_gaussian_mask(256), requires_grad=False)
        self.transformer = Transformer(dim=1024, num_patches=256, depth=2, heads=4, dim_head=256//4, mlp_dim_ratio=2, num_kernals=5, dropout=0.25, 
                                       stochastic_depth=0.1, is_GMM=True, is_SLM=False, mask=self.mask)

        # 使用concat方式，输入维度为512*3
        self.head = nn.Linear(512 * 2, 1)
        init_weights(self.project)
        init_weights(self.head)
        init_weights(self.att)
        init_weights(self.transformer)

        # 确保所有层使用 float32 精度
        self.project = self.project.float()
        self.head = self.head.float()
        self.map = self.map.float()
        self.att = self.att.float()
        self.transformer = self.transformer.float()

    def forward(self, x: torch.Tensor):
        # 确保输入数据是 float32 精度
        x = x.float()

        # 使用 float32 精度的输入进行 xception 特征提取
        ee_features = self.xception_sobel_pass_npr.forward(
            x, return_feature=True)
        
        # # original code for xception sobel npr classification
        ee_features0 = nn.ReLU(inplace=True)(ee_features)
        ee_features0 = F.adaptive_avg_pool2d(ee_features0, (1, 1))
        ee_features0 = ee_features0.view(ee_features0.size(0), -1)
        # ee_features = self.map(ee_features)

        # two strategies for feature fusion
        # 1. concat
        # 2. the output of B x 2048 x 7 x 7, --> B x 2048 x 49, --> B x 49 x 2048, 
        # --> B x 49 x 768, --> transformer with gaussian mixture mask, --> B x 49 x 768, 
        # --> B x 1 x 768, concat with 2048, --> classifier
        # 3. GMM adapter in clip half-up (half-bottom may be better) + LoRA in clip half-up, 
        # --> concat with 2048, --> classifier
        ee_features = self.att(ee_features)
        ee_features = nn.ReLU(inplace=True)(ee_features)
        ee_features = F.adaptive_avg_pool2d(ee_features, (1, 1))
        ee_features = ee_features.view(ee_features.size(0), -1)
        ee_features = self.map(ee_features + ee_features0)
        # ee_features = self.map(ee_features)

        # 使用相同的 float32 精度输入进行 CLIP 特征提取
        clip_feature, all_patches = self.clip_encode_image(x)
        all_patches = self.transformer(all_patches).mean(dim=1)
        all_patches = all_patches @ self.clip_model.visual.proj

        clip_feature = clip_feature + all_patches
        clip_feature = self.project(clip_feature)

        # 使用concat方式融合特征
        combined_feature = torch.cat(
            [clip_feature, ee_features], dim=-1)
        pred = self.head(combined_feature)

        return pred, 0


class clip_lora_eeFrozen_DCT2(nn.Module):
    def __init__(self):
        super(clip_lora_eeFrozen_DCT2, self).__init__()
        # 只在初始化时加载一次模型
        self.xception_sobel_pass_npr = xception_sobel_pass_npr()
        # 检查本地是否存在预训练模型文件
        model_path = "./checkpoints/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt"
        if not os.path.exists(model_path):
            print("正在从Hugging Face下载预训练模型...")
            url = "https://huggingface.co/jzousz/pt/resolve/main/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt?download=true"
            torch.hub.download_url_to_file(url, model_path)
            print("预训练模型下载完成！")

        self.xception_sobel_pass_npr.load_state_dict(torch.load(model_path,
                                                                map_location=device))
        # self.xception_dwt = xception_dwt()
        # self.xception_dwt.to(device)
        self.xception_sobel_pass_npr.to(device)
        # 确保 xception_sobel_pass_npr 使用 float32 精度
        self.xception_sobel_pass_npr = self.xception_sobel_pass_npr.float()
        # self.xception_dwt = self.xception_dwt.float()
        # 冻结所有参数
        for p in self.xception_sobel_pass_npr.parameters():
            p.requires_grad = False

        self.att = MultiSpectralAttentionLayer(2048, 7, 7,  reduction=4, freq_sel_method = 'bot16')

        # init_weights(self.projector)
        self.clip_model, _ = clip.load("/media/sdb/wangjun/tifs_zj/clip/ViT-L-14.pt")
        # 确保 CLIP 模型使用 float32 精度
        self.clip_model = self.clip_model.float()
        self.clip_model.eval()
        
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        self.args = Args(
            encoder="vision",
            backbone="ViT-L/14",
            position="half-up",
            params="qkv",
            # params="qkv",
            r=4,
            alpha=0.5,
            dropout_rate=0.0,
        )
        # print('Apply Lora to half top blocks')
        list_lora_layers = apply_lora(self.args, self.clip_model)
        mark_only_lora_as_trainable(self.clip_model)
        self.clip_model = self.clip_model.to(device)
        self.clip_encode_image = self.clip_model.encode_image
        
        # zou's projector
        self.project = nn.Linear(768, 512)
        self.map = nn.Linear(2048, 512)
        # self.dwt_map = nn.Linear(2048, 512)

        # jun improvement
        self.dropout = nn.Dropout(0.25)
        # self.projector = nn.Linear(2048, 768)
        
        self.mask = nn.Parameter(On_attention_gaussian_mask(256), requires_grad=False)
        self.transformer = Transformer(dim=1024, num_patches=256, depth=2, heads=4, dim_head=256//4, mlp_dim_ratio=2, num_kernals=5, dropout=0.25, 
                                       stochastic_depth=0.1, is_GMM=True, is_SLM=False, mask=self.mask)

        # 使用concat方式，输入维度为512*3
        self.head_aux = nn.Linear(512, 1)
        self.head = nn.Linear(512 * 2, 1)
        init_weights(self.project)
        init_weights(self.head)
        init_weights(self.att)
        init_weights(self.transformer)
        init_weights(self.head_aux)

        # 确保所有层使用 float32 精度
        self.project = self.project.float()
        self.head = self.head.float()
        self.map = self.map.float()
        self.att = self.att.float()
        self.transformer = self.transformer.float()
        self.head_aux = self.head_aux.float()
        
        # freq-aware attn
        self.freq_scale = nn.Parameter(torch.zeros(1))
        self.gmm_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        # 确保输入数据是 float32 精度
        x = x.float()

        # 使用 float32 精度的输入进行 xception 特征提取
        ee_features = self.xception_sobel_pass_npr.forward(
            x, return_feature=True)
        
        # # original code for xception sobel npr classification
        ee_features0 = nn.ReLU(inplace=True)(ee_features)
        ee_features0 = F.adaptive_avg_pool2d(ee_features0, (1, 1))
        ee_features0 = ee_features0.view(ee_features0.size(0), -1)
        # ee_features = self.map(ee_features)

        # two strategies for feature fusion
        # 1. concat
        # 2. the output of B x 2048 x 7 x 7, --> B x 2048 x 49, --> B x 49 x 2048, 
        # --> B x 49 x 768, --> transformer with gaussian mixture mask, --> B x 49 x 768, 
        # --> B x 1 x 768, concat with 2048, --> classifier
        # 3. GMM adapter in clip half-up (half-bottom may be better) + LoRA in clip half-up, 
        # --> concat with 2048, --> classifier
        ee_features = self.att(ee_features)
        ee_features = nn.ReLU(inplace=True)(ee_features)
        ee_features = F.adaptive_avg_pool2d(ee_features, (1, 1))
        ee_features = ee_features.view(ee_features.size(0), -1)

        pred_aux0 = self.head_aux(self.map(ee_features0))
        pred_aux1 = self.head_aux(self.map(ee_features))

        ee_features = self.map(self.freq_scale * ee_features + ee_features0)
        # ee_features = self.map(ee_features)

        # 使用相同的 float32 精度输入进行 CLIP 特征提取
        clip_feature, all_patches = self.clip_encode_image(x)
        all_patches = self.transformer(all_patches).mean(dim=1)
        all_patches = all_patches @ self.clip_model.visual.proj
        pred_aux2 = self.head_aux(self.project(clip_feature))
        pred_aux3 = self.head_aux(self.project(all_patches))

        clip_feature = - clip_feature + self.gmm_scale * all_patches
        clip_feature = self.project(clip_feature)

        # 使用concat方式融合特征
        combined_feature = torch.cat(
            [clip_feature, ee_features], dim=-1)
        # combined_feature = clip_feature + ee_features
        pred = self.head(combined_feature)


        return pred, pred_aux0, pred_aux1, pred_aux2, pred_aux3


class clip_lora_eeFrozen_DCT3(nn.Module):
    def __init__(self):
        super(clip_lora_eeFrozen_DCT3, self).__init__()
        # 只在初始化时加载一次模型
        self.xception_sobel_pass_npr = xception_sobel_pass_npr()
        # 检查本地是否存在预训练模型文件
        model_path = "./checkpoints/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt"
        if not os.path.exists(model_path):
            print("正在从Hugging Face下载预训练模型...")
            url = "https://huggingface.co/jzousz/pt/resolve/main/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt?download=true"
            torch.hub.download_url_to_file(url, model_path)
            print("预训练模型下载完成！")

        self.xception_sobel_pass_npr.load_state_dict(torch.load(model_path,
                                                                map_location=device))
        # self.xception_dwt = xception_dwt()
        # self.xception_dwt.to(device)
        self.xception_sobel_pass_npr.to(device)
        # 确保 xception_sobel_pass_npr 使用 float32 精度
        self.xception_sobel_pass_npr = self.xception_sobel_pass_npr.float()
        # self.xception_dwt = self.xception_dwt.float()
        # 冻结所有参数
        for p in self.xception_sobel_pass_npr.parameters():
            p.requires_grad = False

        self.att = MultiSpectralAttentionLayer(2048, 7, 7,  reduction=4, freq_sel_method = 'bot16')

        # init_weights(self.projector)
        self.clip_model, _ = clip.load("/media/sdb/wangjun/tifs_zj/clip/ViT-L-14.pt")
        # 确保 CLIP 模型使用 float32 精度
        self.clip_model = self.clip_model.float()
        self.clip_model.eval()
        
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        self.args = Args(
            encoder="vision",
            backbone="ViT-L/14",
            position="half-up",
            params="qkv",
            # params="qkv",
            r=4,
            alpha=0.5,
            dropout_rate=0.0,
        )
        # print('Apply Lora to half top blocks')
        list_lora_layers = apply_lora(self.args, self.clip_model)
        mark_only_lora_as_trainable(self.clip_model)
        self.clip_model = self.clip_model.to(device)
        self.clip_encode_image = self.clip_model.encode_image
        
        # zou's projector
        self.project = nn.Linear(768, 512)
        self.map = nn.Linear(2048, 512)
        # self.dwt_map = nn.Linear(2048, 512)

        # jun improvement
        self.dropout = nn.Dropout(0.25)
        # self.projector = nn.Linear(2048, 768)
        
        self.mask = nn.Parameter(On_attention_gaussian_mask(256), requires_grad=False)
        self.transformer = Transformer(dim=1024, num_patches=256, depth=2, heads=4, dim_head=256//4, mlp_dim_ratio=2, num_kernals=5, dropout=0.25, 
                                       stochastic_depth=0.1, is_GMM=True, is_SLM=False, mask=self.mask)

        # self.cross_att1 = CrossAttention(512, num_heads=4)
        # self.cross_att2 = CrossAttention(512, num_heads=4)
        # 使用concat方式，输入维度为512*3
        self.head_aux = nn.Linear(512, 1)
        self.head = nn.Linear(512, 1)
        init_weights(self.project)
        init_weights(self.head)
        init_weights(self.att)
        init_weights(self.transformer)
        init_weights(self.head_aux)

        # 确保所有层使用 float32 精度
        self.project = self.project.float()
        self.head = self.head.float()
        self.map = self.map.float()
        self.att = self.att.float()
        self.transformer = self.transformer.float()
        self.head_aux = self.head_aux.float()
        
        # freq-aware attn
        self.freq_scale = nn.Parameter(torch.zeros(1))
        self.gmm_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        # 确保输入数据是 float32 精度
        x = x.float()

        # 使用 float32 精度的输入进行 xception 特征提取
        ee_features = self.xception_sobel_pass_npr.forward(
            x, return_feature=True)
        
        # # original code for xception sobel npr classification
        ee_features0 = nn.ReLU(inplace=True)(ee_features)
        ee_features0 = F.adaptive_avg_pool2d(ee_features0, (1, 1))
        ee_features0 = ee_features0.view(ee_features0.size(0), -1)
        # ee_features = self.map(ee_features)

        # two strategies for feature fusion
        # 1. concat
        # 2. the output of B x 2048 x 7 x 7, --> B x 2048 x 49, --> B x 49 x 2048, 
        # --> B x 49 x 768, --> transformer with gaussian mixture mask, --> B x 49 x 768, 
        # --> B x 1 x 768, concat with 2048, --> classifier
        # 3. GMM adapter in clip half-up (half-bottom may be better) + LoRA in clip half-up, 
        # --> concat with 2048, --> classifier
        ee_features = self.att(ee_features)
        ee_features = nn.ReLU(inplace=True)(ee_features)
        ee_features = F.adaptive_avg_pool2d(ee_features, (1, 1))
        ee_features = ee_features.view(ee_features.size(0), -1)

        pred_aux0 = self.head_aux(self.map(ee_features0))
        pred_aux1 = self.head_aux(self.map(ee_features))

        ee_features = self.map(self.freq_scale * ee_features + ee_features0)
        # ee_features = self.map(ee_features)

        # 使用相同的 float32 精度输入进行 CLIP 特征提取
        clip_feature, all_patches = self.clip_encode_image(x)
        all_patches = self.transformer(all_patches).mean(dim=1)
        all_patches = all_patches @ self.clip_model.visual.proj
        pred_aux2 = self.head_aux(self.project(clip_feature))
        pred_aux3 = self.head_aux(self.project(all_patches))

        clip_feature = - clip_feature + self.gmm_scale * all_patches
        clip_feature = self.project(clip_feature)

        # 使用concat方式融合特征
        # combined_feature = torch.cat(
        #     [clip_feature, ee_features], dim=-1)
        combined_feature = clip_feature + ee_features
        # combined_feature = self.cross_att1(ee_features, clip_feature, clip_feature)
        pred = self.head(combined_feature)

        return pred, pred_aux0, pred_aux1, pred_aux2, pred_aux3


class clip_lora_eeFrozen_DCT4(nn.Module):
    def __init__(self):
        super(clip_lora_eeFrozen_DCT4, self).__init__()
        # 只在初始化时加载一次模型
        self.xception_sobel_pass_npr = xception_sobel_pass_npr()
        # 检查本地是否存在预训练模型文件
        
        model_path = "./checkpoints/xception_sobel_pass_npr/fdmas_xception_sobel_pass_npr_class4_0.9288116083373297.pt"
        # model_path = "./checkpoints/xception_sobel_pass_npr/fdmas_xception_sobel_pass_npr_class2_0.9098411343228443.pt"
        # model_path = "./checkpoints/xception_sobel_pass_npr/genimage_xception_sobel_pass_npr_0.8833229166666666.pt"
        if not os.path.exists(model_path):
            print("正在从Hugging Face下载预训练模型...")
            url = "https://huggingface.co/jzousz/pt/resolve/main/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt?download=true"
            torch.hub.download_url_to_file(url, model_path)
            print("预训练模型下载完成！")

        self.xception_sobel_pass_npr.load_state_dict(torch.load(model_path,
                                                                map_location=device))
        # self.xception_dwt = xception_dwt()
        # self.xception_dwt.to(device)
        self.xception_sobel_pass_npr.to(device)
        # 确保 xception_sobel_pass_npr 使用 float32 精度
        self.xception_sobel_pass_npr = self.xception_sobel_pass_npr.float()
        # self.xception_dwt = self.xception_dwt.float()
        # 冻结所有参数
        for p in self.xception_sobel_pass_npr.parameters():
            p.requires_grad = False

        self.att = MultiSpectralAttentionLayer(2048, 7, 7,  reduction=4, freq_sel_method = 'bot16')

        # init_weights(self.projector)
        self.clip_model, _ = clip.load("/media/sdb/wangjun/tifs_zj/clip/ViT-L-14.pt")
        # 确保 CLIP 模型使用 float32 精度
        self.clip_model = self.clip_model.float()
        self.clip_model.eval()
        
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        self.args = Args(
            encoder="vision",
            backbone="ViT-L/14",
            position="half-up",
            params="qkv",
            # params="qkv",
            r=4,
            alpha=0.5,
            dropout_rate=0.0,
        )
        # print('Apply Lora to half top blocks')
        list_lora_layers = apply_lora(self.args, self.clip_model)
        mark_only_lora_as_trainable(self.clip_model)
        self.clip_model = self.clip_model.to(device)
        self.clip_encode_image = self.clip_model.encode_image
        
        # zou's projector
        self.project = nn.Linear(768, 512)
        self.map = nn.Linear(2048, 512)
        # self.dwt_map = nn.Linear(2048, 512)

        # jun improvement
        self.dropout = nn.Dropout(0.25)
        # self.projector = nn.Linear(2048, 768)
        
        self.mask = nn.Parameter(On_attention_gaussian_mask(256), requires_grad=False)
        self.transformer = Transformer(dim=1024, num_patches=256, depth=2, heads=4, dim_head=256//4, mlp_dim_ratio=2, num_kernals=5, dropout=0.25, 
                                       stochastic_depth=0.1, is_GMM=True, is_SLM=False, mask=self.mask)

        # self.cross_att1 = CrossAttention(512, num_heads=4)
        # self.cross_att2 = CrossAttention(512, num_heads=4)
        # 使用concat方式，输入维度为512*3
        self.head_aux = nn.Linear(512, 1)
        self.head = nn.Linear(512, 1)
        init_weights(self.project)
        init_weights(self.head)
        init_weights(self.att)
        init_weights(self.transformer)
        init_weights(self.head_aux)

        # 确保所有层使用 float32 精度
        self.project = self.project.float()
        self.head = self.head.float()
        self.map = self.map.float()
        self.att = self.att.float()
        self.transformer = self.transformer.float()
        self.head_aux = self.head_aux.float()
        
        # freq-aware attn
        self.freq_scale = nn.Parameter(torch.zeros(1))
        self.gmm_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        # 确保输入数据是 float32 精度
        x = x.float()

        # 使用 float32 精度的输入进行 xception 特征提取
        ee_features = self.xception_sobel_pass_npr.forward(
            x, return_feature=True)
        
        # # original code for xception sobel npr classification
        ee_features0 = nn.ReLU(inplace=True)(ee_features)
        ee_features0 = F.adaptive_avg_pool2d(ee_features0, (1, 1))
        ee_features0 = ee_features0.view(ee_features0.size(0), -1)
        # ee_features = self.map(ee_features)

        # two strategies for feature fusion
        # 1. concat
        # 2. the output of B x 2048 x 7 x 7, --> B x 2048 x 49, --> B x 49 x 2048, 
        # --> B x 49 x 768, --> transformer with gaussian mixture mask, --> B x 49 x 768, 
        # --> B x 1 x 768, concat with 2048, --> classifier
        # 3. GMM adapter in clip half-up (half-bottom may be better) + LoRA in clip half-up, 
        # --> concat with 2048, --> classifier
        ee_features = self.att(ee_features)
        ee_features = nn.ReLU(inplace=True)(ee_features)
        ee_features = F.adaptive_avg_pool2d(ee_features, (1, 1))
        ee_features = ee_features.view(ee_features.size(0), -1)

        pred_aux0 = self.head_aux(self.map(ee_features0))
        pred_aux1 = self.head_aux(self.map(ee_features))

        ee_features = self.map(self.freq_scale * ee_features + ee_features0)
        # ee_features = self.map(ee_features)

        # 使用相同的 float32 精度输入进行 CLIP 特征提取
        clip_feature, all_patches = self.clip_encode_image(x)
        all_patches = self.transformer(all_patches).mean(dim=1)
        all_patches = all_patches @ self.clip_model.visual.proj
        pred_aux2 = self.head_aux(self.project(clip_feature))
        pred_aux3 = self.head_aux(self.project(all_patches))

        clip_feature = clip_feature - self.gmm_scale * all_patches
        clip_feature = self.project(clip_feature)

        # 使用concat方式融合特征
        # combined_feature = torch.cat(
        #     [clip_feature, ee_features], dim=-1)
        combined_feature = clip_feature + ee_features
        # combined_feature = self.cross_att1(ee_features, clip_feature, clip_feature)
        pred = self.head(combined_feature)

        return pred
        # return pred, pred_aux0, pred_aux1, pred_aux2, pred_aux3, combined_feature


class clip_lora_eeFrozen_GMM2(nn.Module):
    def __init__(self):
        super(clip_lora_eeFrozen_GMM2, self).__init__()
        # 只在初始化时加载一次模型
        self.xception_sobel_pass_npr = xception_sobel_pass_npr()
        # 检查本地是否存在预训练模型文件
        model_path = "./checkpoints/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt"
        if not os.path.exists(model_path):
            print("正在从Hugging Face下载预训练模型...")
            url = "https://huggingface.co/jzousz/pt/resolve/main/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt?download=true"
            torch.hub.download_url_to_file(url, model_path)
            print("预训练模型下载完成！")

        self.xception_sobel_pass_npr.load_state_dict(torch.load(model_path,
                                                                map_location=device))
        # self.xception_dwt = xception_dwt()
        # self.xception_dwt.to(device)
        self.xception_sobel_pass_npr.to(device)
        # 确保 xception_sobel_pass_npr 使用 float32 精度
        self.xception_sobel_pass_npr = self.xception_sobel_pass_npr.float()
        # self.xception_dwt = self.xception_dwt.float()
        # 冻结所有参数
        for p in self.xception_sobel_pass_npr.parameters():
            p.requires_grad = False

        self.clip_model, _ = clip.load("/media/sdb/wangjun/tifs_zj/clip/ViT-L-14.pt")
        # 确保 CLIP 模型使用 float32 精度
        self.clip_model = self.clip_model.float()
        self.clip_model.eval()
        
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        self.args = Args(
            encoder="vision",
            backbone="ViT-L/14",
            position="half-up",
            params="qkv",
            r=4,
            alpha=0.5,
            dropout_rate=0.0,
        )
        # print('Apply Lora to half top blocks')
        list_lora_layers = apply_lora(self.args, self.clip_model)
        mark_only_lora_as_trainable(self.clip_model)
        self.clip_model = self.clip_model.to(device)
        self.clip_encode_image = self.clip_model.encode_image
        
        # zou's projector
        self.project = nn.Linear(768, 512)
        self.map = nn.Linear(2048, 512)
        # self.dwt_map = nn.Linear(2048, 512)

        # jun improvement
        self.dropout = nn.Dropout(0.25)
        self.mask = nn.Parameter(On_attention_gaussian_mask(49), requires_grad=False)
        self.transformer = Transformer(dim=768, num_patches=49, depth=1, heads=8, dim_head=256//4, mlp_dim_ratio=2, num_kernals=5, dropout=0.25, 
                                       stochastic_depth=0.1, is_GMM=True, is_SLM=False, mask=True)

        # 使用concat方式，输入维度为512*3
        # self.head = nn.Linear(512 * 3, 1)
        self.head = nn.Linear(512 * 2, 1)
        init_weights(self.project)
        init_weights(self.head)
        # init_weights(self.dwt_map)

        # 确保所有层使用 float32 精度
        self.project = self.project.float()
        self.head = self.head.float()
        self.map = self.map.float()
        self.transformer = self.transformer.float()

    def forward(self, x: torch.Tensor):
        # 确保输入数据是 float32 精度
        x = x.float()

        # 使用 float32 精度的输入进行 xception 特征提取
        ee_features = self.xception_sobel_pass_npr.forward(
            x, return_feature=True)
        
        # # # original code for xception sobel npr classification
        # ee_features0 = nn.ReLU(inplace=True)(ee_features)
        # ee_features0 = F.adaptive_avg_pool2d(ee_features0, (1, 1))
        # ee_features0 = ee_features0.view(ee_features0.size(0), -1)
        # # ee_features = self.map(ee_features)

        # two strategies for feature fusion
        # 1. concat
        # 2. the output of B x 2048 x 7 x 7, --> B x 2048 x 49, --> B x 49 x 2048, 
        # --> B x 49 x 768, --> transformer with gaussian mixture mask, --> B x 49 x 768, 
        # --> B x 1 x 768, concat with 2048, --> classifier
        # 3. GMM adapter in clip half-up (half-bottom may be better) + LoRA in clip half-up, 
        # --> concat with 2048, --> classifier
        ee_features = ee_features.view(ee_features.size(0), 2048, -1).permute(0,2,1)
        ee_features = self.dropout(ee_features)
        ee_features = self.transformer(ee_features).mean(dim=1)
        # ee_features = self.projector(ee_features)
        # print(ee_features.shape, ee_features0.shape)
        # ee_features = self.map(ee_features + ee_features0)
        ee_features = self.map(ee_features)
        # pred = self.xception_sobel_pass_npr.backbone.classifier(ee_features + ee_features0)


        # # 提取DWT特征
        # # dwt_features = self.xception_dwt.forward(x, return_feature=True)
        # # dwt_features = self.dwt_map(dwt_features)

        # 使用相同的 float32 精度输入进行 CLIP 特征提取
        clip_feature = self.clip_encode_image(x)
        clip_feature = self.project(clip_feature)

        # 使用concat方式融合特征
        # combined_feature = torch.cat(
        #     [clip_feature, ee_features, dwt_features], dim=-1)
        combined_feature = torch.cat(
            [clip_feature, ee_features], dim=-1)
        pred = self.head(combined_feature)

        return pred, torch.tensor(0.0, dtype=torch.float32, device=device)
