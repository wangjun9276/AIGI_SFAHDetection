import importlib
import os
import argparse
from loguru import logger
import csv
from sklearn.metrics import accuracy_score, average_precision_score
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageFile
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import random
torch.cuda.empty_cache()
ImageFile.LOAD_TRUNCATED_IMAGES = True
import skimage
# from augment import pil_jpg
from aug_utils import *
# from model_engine import *
import umap

from pytorch_grad_cam import GradCAM, \
                            ScoreCAM, \
                            GradCAMPlusPlus, \
                            AblationCAM, \
                            XGradCAM, \
                            EigenCAM, \
                            EigenGradCAM, \
                            LayerCAM, \
                            FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
# ===============================================================================================
# ===============================================================================================
class BaseOptions():
    def __init__(self):
        self.initialized = False
        self.parser = None

    def initialize(self, parser):
        parser.add_argument('--method', type=str,
                            default='clip_adapter_distillation', help='method name')
        parser.add_argument('--num_workers', type=int,
                            default=5, help='method name')
        parser.add_argument('--normalize', type=str, default="clip", )
        parser.add_argument('--loadSize', type=int, default=256, )
        parser.add_argument('--epochs', type=int, default=100, )
        parser.add_argument('--ckpt_path', type=str, default=".pt")
        parser.add_argument('--dataset', type=str, default="fdmas", help='try')
        parser.add_argument(
            '--gpu', type=str, default='0',
            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU'
        )
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--optim_type", type=str, default='SGD')
        parser.add_argument("--lr", type=float, default=0.04)
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            self.parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            self.parser = self.initialize(self.parser)
        # get the basic options
        if self.parser is None:
            raise ValueError("Parser not initialized")
        opt, unknown = self.parser.parse_known_args()
        return opt

    # def print_options(self, opt):
    #     message = ''
    #     message += '----------------- Options ---------------\n'
    #     for k, v in sorted(vars(opt).items()):
    #         comment = ''
    #         default = self.parser.get_default(k)
    #         if v != default:
    #             comment = '\t[default: %s]' % str(default)
    #         message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    #     message += '----------------- End -------------------'
    #     logger.info(message)

    def parse(self, print_options=True):
        os.makedirs("./results", exist_ok=True)
        os.makedirs("./checkpoints", exist_ok=True)

        opt = self.gather_options()
        self.opt = opt
        # if print_options:
        #     self.print_options(opt)
        return self.opt


args = BaseOptions().parse()
# device = torch.device('cuda:{}'.format(
#     args.gpu[0])) if args.gpu else torch.device('cpu')
device = torch.device('cuda') if args.gpu else torch.device('cpu')
# ===============================================================================================
BASE_DIR = "/home/ubuntu/lipingkang/lixuyao/zj/dataset/four"
datasets = {
    "fdmas": {
        "test_root": os.path.join('/media/sda/junwang/datasets/inthewild/', "test"),
        "vals": [
            # "progan", "stylegan", "biggan", "cyclegan", "stargan", "gaugan",
            # "stylegan2", "whichfaceisreal", "ADM", "Glide", "Midjourney",
            # "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "VQDM",
            # "wukong", "DALLE2"
            "progan", "whichfaceisreal", "ADM", "Glide", "Midjourney",
            "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "VQDM",
            "wukong", "DALLE2"
        ],

    },
    "assess": {
        # "train_root": os.path.join(BASE_DIR, "train"),
        "train_root": os.path.join(BASE_DIR, "val"),
        "val_root": os.path.join(BASE_DIR, "val"),
        "test_root": os.path.join(BASE_DIR, "test"),
        "vals": [
            "progan", "stylegan", "biggan", "cyclegan", "stargan", "gaugan",
            "stylegan2", "whichfaceisreal", "ADM", "Glide", "Midjourney",
            "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "VQDM",
            "wukong", "DALLE2"
        ],

    },
    "cnnspot": {
        "test_root": os.path.join('/media/sda/junwang/datasets/CNNspot', "test"),
        "vals": [
            # "biggan", "cyclegan", "gaugan", "progan", "seeingdark", "stylegan", "whichfaceisreal",
            # "crn", "deepfake", "imle", "san", "stargan", "stylegan2"
            "seeingdark", "crn", "deepfake", "imle", "san"
        ]
    },
    "ojha": {
        "test_root": os.path.join('/media/sda/junwang/datasets/UnivBench/testset/images', "UnivFD"),
        "vals": ['dalle', 'glide_100_10', 'glide_100_27', 'glide_50_27', 'guided',
                 'ldm_100', 'ldm_200', 'ldm_200_cfg']
    },
    "tan": {
        "test_root": os.path.join('/media/sda/junwang/datasets/UnivBench/testset/images', "Tan"),
        "vals": ['AttGAN', 'BEGAN', 'CramerGAN', 'InfoMaxGAN',
                 'MMDGAN', 'RelGAN', 'S3GAN', 'SNGAN', 'STGAN']
    },
    "Chameleon": {
        "test_root": os.path.join('/media/sda/junwang/datasets/', "Chameleon"),
        "vals": ['test']
    },

    "VARsC": {
        "test_root": os.path.join('/media/sda/junwang/datasets/UnivBench/testset/images', "commercial"),
        "vals": ['VARsC']
    },
    # "genimage": {
    #     "test_root": os.path.join(BASE_DIR, "test"),
    #     "vals": [
    #         'adm_imagenet',     'glide_imagenet',       'sdv4_imagenet', 'vqdm_imagenet',
    #         'biggan_imagenet',  'midjourney_imagenet',  'sdv5_imagenet',  'wukong_imagenet']

    # },
}
test_root = datasets[args.dataset]["test_root"]
vals = datasets[args.dataset]["vals"]
y_features, gan_label, class_y = [], [], []

class ForenSynths(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['0_real', '1_fake']
        self.data = []

        # 使用多进程预加载数据
        from concurrent.futures import ThreadPoolExecutor

        def process_path(root, filename):
            file_path = os.path.join(root, filename)
            if '0_real' in file_path:
                return (file_path, 0)
            if '1_fake' in file_path:
                return (file_path, 1)
            return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for root, _, files in os.walk(self.root_dir):
                for filename in files:
                    futures.append(executor.submit(
                        process_path, root, filename))

            for future in futures:
                result = future.result()
                if result is not None:
                    self.data.append(result)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        try:
            image = Image.open(img_path).convert("RGB")
            # image = jpeg_compression_in_buffer(image, 90)
            # image = pil_jpg(image, 50)
            # image = skimage.util.random_noise(np.array(image), mode='gaussian', var=0.0001, clip=True)
            # image = Image.fromarray(np.uint8(image*255.0))
            # image = Image.fromarray(image)
            image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            # 返回一个默认图像
            return torch.zeros(3, 224, 224), label


# ===============================================================================================
MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}
rz_dict = {'bilinear': InterpolationMode.BILINEAR,
           'bicubic': InterpolationMode.BICUBIC,
           'lanczos': InterpolationMode.LANCZOS,
           'nearest': InterpolationMode.NEAREST}


def judge_img(img):
    img_width, img_height = img.size
    if (img_width < args.loadSize or img_height < args.loadSize):
        img = torchvision.transforms.Resize((args.loadSize, args.loadSize), interpolation=InterpolationMode.BILINEAR)(
            img)
    return img


def test_augment():
    transform_list = [
        transforms.Lambda(judge_img),
        # transforms.Resize([224, 224]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN[args.normalize], std=STD[args.normalize]),
    ]
    return transforms.Compose(transform_list)


# ===============================================================================================
def set_random_seed(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def validate(model):
    # folder = '/media/sda/junwang/datasets/CNNspot/test/biggan/1_fake'
    # folder = '/media/sda/junwang/datasets/CNNspot/test/progan/airplane/1_fake'
    # folder = '/media/sda/junwang/datasets/UnivBench/testset/images/DRCT/stable-diffusion-xl-base-1.0/val2017'
    folder = '/media/sda/junwang/datasets/UnivBench/testset/images/UnivFD/ldm_100/1_fake'
    images = ['jqnuoffbtb.png', 'jroisduzzw.png', 'lvtmqaadzz.png', 'odyhbcgchi.png', 'pvrfveosdy.png',
              'rprdsmygod.png', 'whqsatsqum.png']
    # folder = '/media/sda/junwang/datasets/UnivBench/testset/images/UnivFD/imagenet/0_real'
    # images = ['bsoomuiluf.JPEG', 'fgizsruyxb.JPEG', 'glmtywqxhj.JPEG', 'hnzpgeledt.JPEG', 'uprjbmydux.JPEG', 'uxbgzfafqm.JPEG',
    #           'wgmfkffpbs.JPEG']
    transforms_augs = transforms.Compose([
                                            # transforms.ToPILImage(),
                                            # cnnspot_func,
                                            # transforms.GaussianBlur(kernel_size=(3), sigma=(0.1, 5.)),
                                            transforms.CenterCrop([224, 224]),
                                            transforms.ToTensor(),
                                            # transforms.v2.GaussianNoise(0.0, 0.001, clip=True),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ])
    image = os.path.join(folder, images[0])
    img = Image.open(image).convert('RGB')
    # img.save(f'results/cammap/{os.path.basename(image)}')
    img = transforms_augs(img).unsqueeze(0)

    def reshape_transform(tensor, height=16, width=16):
        # print(tensor.shape)
        if tensor.shape[0]==257:
            tensor = tensor.permute(1,0,2)
        # 去掉cls token
        # print(tensor.shape)
        # result = tensor[:, 1:, :].reshape(tensor.size(0),
        result = tensor.reshape(tensor.size(0),
        height, width, tensor.size(2))

        # 将通道维度放到第一个位置
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    
    for (name, param) in model.named_parameters():
        param.requires_grad = True
        # print(name)

    # print(model.transformer.layers[-1][0].fn.mask.sigma.shape)
    # 创建 GradCAM 对象
    cam = GradCAM(model=model,
    # cam = EigenCAM(model=model,
                target_layers = 
                                [model.transformer.layers[-1][1].fn.net[3]],
                                # [model.transformer.layers[-1][1].fn.net[0]],
                                # [model.transformer.layers[-1][0].fn.to_qkv],
                                # [model.transformer.layers[-1][0].fn.to_out[0]],
                                # [model.transformer.layers[-1][0].fn.mask.sigma],
                                # [model.clip_model.visual.transformer.resblocks[-1].mlp.c_proj],
                            #    [model.clip_model.visual.ln_post],
                            #    [model.att.fc[2]],
                            #    [model.xception_sobel_pass_npr.backbone.conv3.pointwise],
                            #    [model.xception_sobel_pass_npr.backbone.conv4.conv1],
                # target_layers=[model.clip_model.visual.transformer.resblocks[-1].mlp.c_proj],
                # target_layers=[basemodel.visual.ln_post],
                # target_layers=[basemodel.blocks[-1].ffn.norm],
                # 这里的target_layer要看模型情况，
                # 比如还有可能是：target_layers = [model.blocks[-1].ffn.norm]
                use_cuda=True,
                reshape_transform=reshape_transform)
                # reshape_transform=None)
    # 计算 grad-cam
    grayscale_cam = cam(img)
    grayscale_cam = grayscale_cam[0, :]
    # print(grayscale_cam.shape)

    # 将 grad-cam 的输出叠加到原始图像上
    visualization = show_cam_on_image(np.array(Image.open(image).convert('RGB').resize((224, 224)))/255.0, grayscale_cam)

    # 保存可视化结果
    cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
    cv2.imwrite(f'results/cammap/cam_{os.path.basename(image)}', visualization)


# ===============================================================================================

def create_dataloaders():
    dl_test = {}
    for v_id, val in enumerate(vals):
        test_dir = os.path.join(test_root, val)
        dl_test[val] = torch.utils.data.DataLoader(
            ForenSynths(test_dir, test_augment()),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
    return dl_test


def create_net():
    module = importlib.import_module('model_engine')
    model = getattr(module, args.method)()
    # model_path = '/media/sdb/wangjun/tifs_zj/checkpoints/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt'
    # model_path = './checkpoints/clip_lora_eeFrozen_DCT2/fdmas_clip_lora_eeFrozen_DCT2_0.9354950228032031.pt'
    model_path = './checkpoints/clip_lora_eeFrozen_DCT4/fdmas_clip_lora_eeFrozen_DCT4_class4_0.9511984589323377.pt'
    # model_path = './checkpoints/clip_lora_eeFrozen_DCT4/fdmas_clip_lora_eeFrozen_DCT4_class2_0.9330424333640768.pt'
    # model_path = './checkpoints/Xception_DCT_bot16/fdmas_Xception_DCT_0.9318243310916867.pt'
    # model_path = './checkpoints/genimage_clip_lora_eeFrozen_DCT4_0.9447630208333334.pt'
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model = model.to(device)
    return model


def test_epoch(model):
    validate(model)


def test():
    # ======================================================================
    set_random_seed()

    # ======================================================================
    model = create_net()
    # 验证阶段
    model.eval()
    
    test_acc = test_epoch(model)


def visualize(data, label, classes):
    colors_16 = [
    "#1f77b4",  # 蓝
    "#ff7f0e",  # 橙
    "#2ca02c",  # 绿
    "#d62728",  # 红
    "#9467bd",  # 紫
    "#8c564b",  # 棕
    "#e377c2",  # 粉
    "#7f7f7f",  # 灰
    "#bcbd22",  # 黄绿
    "#17becf",  # 青
    "#393b79",  # 深蓝
    "#637939",  # 橄榄绿
    "#8c6d31",  # 金棕
    "#843c39",  # 暗红
    "#7b4173",  # 暗紫
    "#3182bd"   # 天蓝
]

    # print(np.array(data).shape)
    plt.figure(figsize=(8, 6))
    # print('Running visualization')
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(np.array(data))

    for idxx in range(len(datasets['fdmas']['vals'])):
        datas, labels, fake_embeddings, real_embeddings = [], [], [], []
        for idxxx, item in enumerate(classes):
            # idxxx = int(idxxx)
            if item == idxx:
                datas.append(data[idxxx])
                labels.append(label[idxxx]+idxx)
                if label[idxxx] == 0:
                    real_embeddings.append(embedding[idxxx, :])
                else:
                    fake_embeddings.append(embedding[idxxx, :])
                # embeddings.append(embedding[idxxx, :])

        real_embeddings = np.array(real_embeddings)
        # print(real_embeddings.shape)
        fake_embeddings = np.array(fake_embeddings)
        # print(fake_embeddings.shape)
        # plt.scatter(embeddings[:, 0], embeddings[:, 1], c=np.array(labels), cmap='viridis')
        plt.scatter(real_embeddings[:, 0], real_embeddings[:, 1], c=colors_16[0], label='Real' if idxx == 0 else '')
        plt.scatter(fake_embeddings[:, 0], fake_embeddings[:, 1], c=colors_16[idxx+1], label=datasets['fdmas']['vals'][idxx])
        # plt.title('UMAP Projection of Moon Data')
        # plt.show()
    plt.legend()
    plt.savefig('visual.pdf')


if __name__ == "__main__":
    test()
