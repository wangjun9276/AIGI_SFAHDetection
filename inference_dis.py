import importlib
import os
import argparse
from loguru import logger
import sys
import csv
from sklearn.metrics import accuracy_score, average_precision_score
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageFile
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import random
import torch
import os
import torch
import torch
import os
import importlib
torch.cuda.empty_cache()
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
device = torch.device('cuda:{}'.format(
    args.gpu[0])) if args.gpu else torch.device('cpu')
# ===============================================================================================
BASE_DIR = "/home/ubuntu/lipingkang/lixuyao/zj/dataset/four"
datasets = {
    "fdmas": {
        "test_root": os.path.join('/media/sda/junwang/datasets/inthewild/', "test"),
        "vals": [
            "progan", "stylegan", "biggan", "cyclegan", "stargan", "gaugan",
            "stylegan2", "whichfaceisreal", "ADM", "Glide", "Midjourney",
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


def validate(model, student_model, data_loader):
    model.eval()
    y_true, y_pred = [], []
    y_pred0, y_pred1, y_pred2, y_pred3, pred_sum = [], [], [],  [], []
    with torch.no_grad():
        for img, label in data_loader:
            in_tens = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            
            teacher_outputs = model(in_tens, False)
            student_outputs = student_model(in_tens, False)
            
            errors = teacher_outputs
            # print(errors.shape)
            # errors = errors.mean(3).mean(2).mean(1)
            # errors = errors.mean(1)
            # errors = errors.sigmoid()
            
            # outputs = errors.squeeze(1)
            y_pred.extend(errors.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    # print(f"overall acc: {acc}, ap: {ap}, r_acc: {r_acc}, f_acc: {f_acc}")

    # y_true, y_pred0 = np.array(y_true), np.array(y_pred0)
    # r_acc0 = accuracy_score(y_true[y_true == 0], y_pred0[y_true == 0] > 0.5)
    # f_acc0 = accuracy_score(y_true[y_true == 1], y_pred0[y_true == 1] > 0.5)
    # acc0 = accuracy_score(y_true, y_pred0 > 0.5)
    # ap0 = average_precision_score(y_true, y_pred0)
    # print(f"Xception acc: {acc0}, ap: {ap0}, r_acc: {r_acc0}, f_acc: {f_acc0}")

    # y_true, y_pred1 = np.array(y_true), np.array(y_pred1)
    # r_acc1 = accuracy_score(y_true[y_true == 0], y_pred1[y_true == 0] <= 0.5)
    # f_acc1 = accuracy_score(y_true[y_true == 1], y_pred1[y_true == 1] <= 0.5)
    # acc1 = accuracy_score(y_true, y_pred1 <= 0.5)
    # ap1 = average_precision_score(y_true, y_pred1)
    # print(f"DCT acc: {acc1}, ap: {ap1}, r_acc: {r_acc1}, f_acc: {f_acc1}")

    # y_true, y_pred2 = np.array(y_true), np.array(y_pred2)
    # r_acc2 = accuracy_score(y_true[y_true == 0], y_pred2[y_true == 0] > 0.5)
    # f_acc2 = accuracy_score(y_true[y_true == 1], y_pred2[y_true == 1] > 0.5)
    # acc2 = accuracy_score(y_true, y_pred2 > 0.5)
    # ap2 = average_precision_score(y_true, y_pred2)
    # print(f"CLIP acc: {acc2}, ap: {ap2}, r_acc: {r_acc2}, f_acc: {f_acc2}")

    # y_true, y_pred3 = np.array(y_true), np.array(y_pred3)
    # r_acc3 = accuracy_score(y_true[y_true == 0], y_pred3[y_true == 0] <= 0.5)
    # f_acc3 = accuracy_score(y_true[y_true == 1], y_pred3[y_true == 1] <= 0.5)
    # acc3 = accuracy_score(y_true, y_pred3 <= 0.5)
    # ap3 = average_precision_score(y_true, y_pred3)
    # print(f"GMM acc: {acc3}, ap: {ap3}, r_acc: {r_acc3}, f_acc: {f_acc3}")

    # y_true, pred_sum = np.array(y_true), np.array(pred_sum)
    # r_accs = accuracy_score(y_true[y_true == 0], pred_sum[y_true == 0] > 0.5)
    # f_accs = accuracy_score(y_true[y_true == 1], pred_sum[y_true == 1] > 0.5)
    # accs = accuracy_score(y_true, pred_sum > 0.5)
    # aps = average_precision_score(y_true, pred_sum)
    # print(f"SUM acc: {accs}, ap: {aps}, r_acc: {r_accs}, f_acc: {f_accs}")
    return acc, ap, r_acc, f_acc, y_true, y_pred


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
    # model_path = './checkpoints/fdmas_clip_lora_eeFrozen_DCT4_0.9511984589323377.pt'
    # model_path = './checkpoints/genimage_clip_lora_eeFrozen_DCT4_0.9447630208333334.pt'
    # model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model = model.to(device)
    return model

def create_net2():
    module = importlib.import_module('model_engine')
    args.method = 'xception_sobel_pass_npr_student'
    model = getattr(module, args.method)()
    # model_path = '/media/sdb/wangjun/tifs_zj/checkpoints/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt'
    # model_path = './checkpoints/clip_lora_eeFrozen_DCT2/fdmas_clip_lora_eeFrozen_DCT2_0.9354950228032031.pt'
    # model_path = './checkpoints/fdmas_clip_lora_eeFrozen_DCT4_0.9511984589323377.pt'
    model_path = './checkpoints/fdmas_xception_sobel_pass_npr_student_0.7994061907493891.pt'
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model = model.to(device)
    return model


def test_epoch(model, dl_test):
    student_model = create_net2()
    student_model.to(device)
    student_model.eval()

    datasets = []
    accs = []
    aps = []
    r_accs = []
    f_accs = []
    data_csv = []
    # 添加参数信息
    data_csv.append(['Parameters'])
    for k, v in sorted(vars(args).items()):
        data_csv.append([k, v])
    data_csv.append([])  # 添加空行分隔
    # 添加性能指标
    data_csv.append(['Dataset', 'Accuracy', 'AP', 'r_acc', 'f_acc'])

    with torch.no_grad():
        for _, (test_type, loader) in enumerate(dl_test.items()):
            acc, ap, r_acc, f_acc = validate(model, student_model, loader)[:4]
            data_csv.append([test_type, acc * 100, ap *
                            100, r_acc * 100, f_acc * 100])
            datasets.append(test_type)
            accs.append(acc)
            aps.append(ap)
            r_accs.append(r_acc)
            f_accs.append(f_acc)

            logger.info("( {:12}) acc: {:.4f}; ap: {:.4f};  r_acc: {:.4f}, f_acc: {:.4f}".format(
                test_type, acc * 100, ap * 100, r_acc * 100, f_acc * 100))

    mean_acc = np.array(accs).mean() * 100
    mean_ap = np.array(aps).mean() * 100
    logger.info("({:10}) acc: {:.1f}; ap: {:.1f}".format(
        'Mean', mean_acc, mean_ap))
    data_csv.append(['MEAN', mean_acc, mean_ap])

    # 使用异步IO写入CSV
    import asyncio

    async def write_csv():
        with open(f'{os.path.join("./results", str(args.dataset)+"_"+str(args.method) + "_" + str(mean_acc))}.csv', 'a',
                  newline='') as file_:
            writer = csv.writer(file_, delimiter=',')
            writer.writerows(data_csv)

    asyncio.run(write_csv())
    return mean_acc / 100  # 返回归一化的准确率


def test():
    # ======================================================================
    set_random_seed()

    # ======================================================================
    model = create_net()
    model = model.to(device)
    # ======================================================================
    dl_test = create_dataloaders()
    # ======================================================================
    # 验证阶段
    model.eval()
    
    test_acc = test_epoch(model, dl_test)


if __name__ == "__main__":
    test()
