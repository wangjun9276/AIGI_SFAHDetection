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
import os
import importlib
torch.cuda.empty_cache()
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.nn.functional as F
from typing import Optional



class EarlyStopping:
    def __init__(
        self,
        patience=3,
        delta=0
    ):
        self.patience = patience
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.delta = delta

    def __call__(self, score, model, args):

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, os.path.join(
                "./checkpoints", str(args.dataset)+"_"+str(args.method) + "_" + str(score) + args.ckpt_path))
        elif score < self.best_score - self.delta:
            self.counter += 1
            logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping.")
        else:
            self.best_score = score
            self.save_checkpoint(model, os.path.join(
                "./checkpoints",  str(args.dataset)+"_"+str(args.method) + "_" + str(score) + args.ckpt_path))
            self.counter = 0

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)


def _compute_white_box_distillation_loss(self, student_logits: torch.Tensor, teacher_probs: torch.Tensor, labels: Optional[torch.Tensor]):
    # student_logits = student_logits[:, :self.max_seq_length, :]
    # teacher_probs = teacher_logits[:, :student_logits.size(1), :student_logits.size(-1)]
    mask = (labels != -100).float() if labels is not None else torch.ones_like(student_logits[:, :, 0])
    
    if self.distillation_type == "forward_kld":
        # Forward KLD: student learns from teacher (original implementation)
        loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            teacher_probs,
            reduction='none',
            log_target=False
        ).sum(dim=-1)/torch.sum(mask.view(-1), dim=0) 
    elif self.distillation_type == "reverse_kld":
        # Reverse KLD: teacher provides certainty to student
        loss = F.kl_div(
            torch.log(teacher_probs.clamp(min=1e-10)),  # avoid log(0)
            F.softmax(student_logits, dim=-1),
            reduction='none',
            log_target=False
        ).sum(dim=-1)/torch.sum(mask.view(-1), dim=0) 
    else:
        raise ValueError(f"Unsupported distillation type: {self.distillation_type}. Use 'forward_kld' or 'reverse_kld'")
        
    return (loss * mask).sum() / mask.sum()


def init_logger(console_log_level: str = "INFO") -> None:
    logger.remove()
    # Console output
    logger.add(
        sys.stderr,
        level=console_log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}",
        colorize=True,
    )

    # File output
    logger.add(
        "logs/debug_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message} | {extra}",
        backtrace=True,
        diagnose=True,
    )


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

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        logger.info(message)

    def parse(self, print_options=True):
        os.makedirs("./results", exist_ok=True)
        os.makedirs("./checkpoints", exist_ok=True)

        opt = self.gather_options()
        self.opt = opt
        if print_options:
            self.print_options(opt)
        return self.opt


args = BaseOptions().parse()
device = torch.device('cuda:{}'.format(
    args.gpu[0])) if args.gpu else torch.device('cpu')
# ===============================================================================================
BASE_DIR = "/home/ubuntu/lipingkang/lixuyao/zj/dataset/four"

datasets = {
    "fdmas": {
        "train_root": os.path.join('/media/sda/junwang/datasets/CNNspot/', "progan_train"),
        # "train_root": os.path.join(BASE_DIR, "val"),
        "val_root": os.path.join('/media/sda/junwang/datasets/CNNspot/', "progan_val"),
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
        "train_root": os.path.join('BASE_DIR', "val"),
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
        "train_root": os.path.join(BASE_DIR, "train"),
        "val_root": os.path.join(BASE_DIR, "val"),
        "test_root": os.path.join(BASE_DIR, "cnnspot"),
        "vals": [
            "biggan", "cyclegan", "gaugan", "progan", "seeingdark", "stylegan", "whichfaceisreal",
            "crn", "deepfake", "imle", "san", "stargan", "stylegan"
        ]
    },
    "ojha": {
        "train_root": os.path.join(BASE_DIR, "train"),
        "val_root": os.path.join(BASE_DIR, "val"),
        "test_root": os.path.join(BASE_DIR, "ojha"),
        "vals": ['dalle', 'glide_100_10', 'glide_100_27', 'glide_50_27', 'guided',
                 'ldm_100', 'ldm_200', 'ldm_200_cfg', 'pndm', 'vqdiffusion']
    },
    "tan": {
        "train_root": os.path.join(BASE_DIR, "train"),
        "val_root": os.path.join(BASE_DIR, "val"),
        "test_root": os.path.join(BASE_DIR, "tan"),
        "vals": ['AttGAN', 'BEGAN', 'CramerGAN', 'InfoMaxGAN',
                 'MMDGAN', 'RelGAN', 'S3GAN', 'SNGAN', 'STGAN']
    },

    # "genimage": {
    #     "train_root": os.path.join(BASE_DIR, "train"),
    #     "val_root": os.path.join(BASE_DIR, "val"),
    #     "test_root": os.path.join(BASE_DIR, "test"),
    #     "vals": [
    #         'adm_imagenet',     'glide_imagenet',       'sdv4_imagenet', 'vqdm_imagenet',
    #         'biggan_imagenet',  'midjourney_imagenet',  'sdv5_imagenet',  'wukong_imagenet']

    # },
}
# print([args.dataset])
train_root = datasets[args.dataset]["train_root"]
val_root = datasets[args.dataset]["val_root"]
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
                # for filename in ['chair', 'cat', 'horse', 'car']:
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


def train_augment():
    transform_list = [
        transforms.Lambda(judge_img),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN[args.normalize], std=STD[args.normalize]),
    ]
    return transforms.Compose(transform_list)


def val_augment():
    transform_list = [
        transforms.Lambda(judge_img),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN[args.normalize], std=STD[args.normalize]),
    ]
    return transforms.Compose(transform_list)


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


def validate(model, data_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for img, label in data_loader:
            in_tens = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            outputs, _ = model(in_tens)
            # outputs, _, _, _, _ = model(in_tens)
            outputs = outputs.squeeze(1)
            y_pred.extend(outputs.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred


# ===============================================================================================

def create_dataloaders():
    dl_train = torch.utils.data.DataLoader(
        ForenSynths(train_root, train_augment()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,  # 使用pin_memory加速数据传输
        persistent_workers=True,  # 保持worker进程存活
        prefetch_factor=2  # 预加载因子
    )
    dl_val = torch.utils.data.DataLoader(
        ForenSynths(val_root, train_augment()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
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
    return dl_train, dl_val, dl_test


def create_net():
    module = importlib.import_module('model_engine')
    model = getattr(module, args.method)()
    model_path = './checkpoints/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

def create_net2():
    module = importlib.import_module('model_engine')
    args.method = 'xception_sobel_pass_npr_student'
    model = getattr(module, args.method)()
    # './checkpoints/fdmas_xception_sobel_pass_npr_0.9288116083373297.pt'
    model_path = './checkpoints/fdmas_xception_sobel_pass_npr_student_0.701232776338285.pt'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model


def get_optimizer(model, optim_type, lr):
    if optim_type == 'Adam':
        optimizer = torch.optim.Adam(filter(
            lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999))
    elif optim_type == 'SGD':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)
    elif optim_type == 'AdamW':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999),
                                      weight_decay=0.0)
    else:
        raise ValueError(f"Invalid optimizer: {optim_type}")
    return optimizer


def test_epoch(model, dl_test):
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
            acc, ap, r_acc, f_acc = validate(model, loader)[:4]
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


def train():
    # ======================================================================
    set_random_seed()
    init_logger()
    T = 0.7
    lamb = 0.9
    # ======================================================================
    teacher_model = create_net()
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)

    args.method = 'xception_sobel_pass_npr_student'
    student_model = create_net2()
    student_model = student_model.to(device)

    # 使用梯度累积
    accumulation_steps = 2  # 累积2个批次的梯度

    logger.info("learnable:")
    count = 1
    for name, param in student_model.named_parameters():
        if param.requires_grad:
            logger.info(f'{count}:{name}')
            count = count + 1

    # ======================================================================
    dl_train, dl_val, dl_test = create_dataloaders()
    optimizer = get_optimizer(student_model, args.optim_type, args.lr)

    # 使用学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose='True'
    )

    # ======================================================================
    early_stopping = EarlyStopping()
    # ======================================================================
    for epoch in range(1, args.epochs + 1):
        # 训练阶段
        student_model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, (inputs, labels) in enumerate(dl_train):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            # 前向传播
            # outputs, loss_distillation = model(inputs)
            # # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits, teacher_feature = teacher_model(inputs)

            student_logits, student_feature = student_model(inputs)

            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)
            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)
            # soft_targets_loss = torch.nn.MSELoss()(student_feature, teacher_feature)

            # Calculate the true label loss
            student_logits = student_logits.squeeze(1)
            label_loss = nn.BCEWithLogitsLoss()(student_logits, labels)

            # Weighted sum of the two losses
            loss = (1 - lamb) * soft_targets_loss + lamb * label_loss
            loss = loss / accumulation_steps

            torch.autograd.set_detect_anomaly(True)
            # 反向传播
            # print(loss)
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * inputs.size(0) * accumulation_steps

        train_loss = running_loss / len(dl_train.dataset)
        logger.info(f"epoch【{epoch}】--> epoch_loss= {train_loss}")

        # 验证阶段
        student_model.eval()
        val_acc = validate(student_model, dl_val)[0]
        logger.info(f"epoch【{epoch}】--> val_acc = {100 * val_acc:.2f}%")

        # 更新学习率
        scheduler.step(val_acc)

        if val_acc >= 0.978:
        # if val_acc >= 0.94:
            test_acc = test_epoch(student_model, dl_test)
            logger.info(f"epoch【{epoch}】--> test_acc = {100 * test_acc:.2f}%")
            early_stopping(test_acc, student_model, args)
            # if early_stopping.early_stop:
            #     return
    return


if __name__ == "__main__":
    train()
