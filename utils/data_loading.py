import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
import os

from torchvision import transforms
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, datasetpath):
        # 下文 txt文件中的每行
        self.datasetpath = datasetpath
        self.imglist = os.listdir(os.path.join(self.datasetpath, 'image'))
        self.transform_img = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalization
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        self.transform_label = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.imglist)

    @staticmethod
    def preprosess(img):
        img_train = Image.open(img)
        transform_img = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalization
        ])
        img_train = transform_img(img_train)
        return img_train

    @staticmethod
    def getmask(img):
        transform_mask = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        img = cv2.imread(img, 0)
        ret, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        threshold = 1 - threshold / 255 + 0.1
        threshold = Image.fromarray(threshold)
        mask = transform_mask(threshold)
        return mask

    def __getitem__(self, index):
        # get train img
        # print(self.imglist[index])
        img_train = Image.open(os.path.join(self.datasetpath, 'image', self.imglist[index]))

        img_train = self.transform_img(img_train)

        # get mask img
        img = cv2.imread(os.path.join(self.datasetpath, 'image', self.imglist[index]), 0)
        ret, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        threshold = 1 - threshold / 255 + 0.1
        threshold = Image.fromarray(threshold)
        mask = self.transform_mask(threshold)

        # get label img
        lablelist = os.listdir(os.path.join(self.datasetpath, 'label'))
        t = []
        imgname = self.imglist[index].split(".")[0]
        for each in lablelist:
            if imgname in each:
                t.append(each)

        img_label = 255 - cv2.resize(cv2.imread(os.path.join(self.datasetpath, 'label', t[0]), 0), (512, 512))

        for i in range(1, len(t)):
            img_t = 255 - cv2.resize(cv2.imread(os.path.join(self.datasetpath, 'label', t[i]), 0), (512, 512))
            img_label = cv2.add(img_label, img_t)
        img_label = np.where(img_label > 10, 1.0, 0).astype(np.float32)

        # ret4, img_label = cv2.threshold(img_label, 20, 255, cv2.THRESH_TOZERO_INV)
        # img_label=img_label/255
        # img_label=Image.fromarray(img_label)
        # img_label=self.transform_label(img_label)

        # return img_train, mask, img_label
        return {
            'image': img_train,
            'mask': img_label,
            'condition': mask,
        }


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):

        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')


if __name__ == '__main__':
    dataset = MyDataset('/shared_dir/dataset/crack-segment')
    val_percent = 0.1
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(1))
    loader_args = dict(batch_size=4, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    num_val_batches = len(val_loader)

    for batch in tqdm(val_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        print(1)