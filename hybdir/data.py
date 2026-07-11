from pathlib import Path
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class ResizeIfSmall:
    def __init__(self, minimum=256):
        self.minimum = minimum

    def __call__(self, image):
        width, height = image.size
        if width >= self.minimum and height >= self.minimum:
            return image
        scale = self.minimum / min(width, height)
        size = (round(width * scale), round(height * scale))
        return image.resize(size, Image.Resampling.BILINEAR)


def build_transform(train=False, image_size=224, load_size=256):
    operations = [ResizeIfSmall(load_size)]
    if train:
        operations += [transforms.RandomCrop(image_size), transforms.RandomHorizontalFlip()]
    else:
        operations += [transforms.CenterCrop(image_size)]
    operations += [transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)]
    return transforms.Compose(operations)


class BinaryImageDataset(Dataset):
    def __init__(self, root, transform, real_names=("0_real", "nature", "real"), fake_names=("1_fake", "ai", "fake")):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        self.transform = transform
        self.real_names = {name.lower() for name in real_names}
        self.fake_names = {name.lower() for name in fake_names}
        self.samples = self._scan()
        if not self.samples:
            raise RuntimeError(f"No labeled images found under {self.root}. Expected path components such as 0_real/1_fake or nature/ai.")

    def _label_from_path(self, path):
        parts = {part.lower() for part in path.relative_to(self.root).parts[:-1]}
        real = bool(parts & self.real_names)
        fake = bool(parts & self.fake_names)
        if real == fake:
            return None
        return 0 if real else 1

    def _scan(self):
        samples = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                label = self._label_from_path(path)
                if label is not None:
                    samples.append((path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                image = self.transform(image.convert("RGB"))
            return image, torch.tensor(label, dtype=torch.float32), str(path)
        except Exception:
            return None


def safe_collate(batch):
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    images, labels, paths = zip(*valid)
    return torch.stack(images), torch.stack(labels), list(paths)


def create_loader(root, batch_size, train=False, num_workers=4, image_size=224, load_size=256):
    dataset = BinaryImageDataset(root, build_transform(train, image_size, load_size))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers > 0 else None, collate_fn=safe_collate, drop_last=False)
