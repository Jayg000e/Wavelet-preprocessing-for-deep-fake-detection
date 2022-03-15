# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)

    dataset=FaceDataset(real_dir=config.REAL_PATH,
                        fake_dir=config.FAKE_PATH,
                        real_wavelet_dir=config.REAL_WAVELET_PATH,
                        fake_wavelet_dir=config.FAKE_WAVELET_PATH,
                        is_train=is_train,
                        require_image=not config.WAVELET_FEATURE,
                        transform=transform)
    # if config.DATA.DATASET == 'imagenet':
    #     prefix = 'train' if is_train else 'val'
    #     if config.DATA.ZIP_MODE:
    #         ann_file = prefix + "_map.txt"
    #         prefix = prefix + ".zip@/"
    #         dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
    #                                     cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
    #     else:
    #         root = os.path.join(config.DATA.DATA_PATH, prefix)
    #         dataset = datasets.ImageFolder(root, transform=transform)
    #     nb_classes = 1000
    # else:
    #     raise NotImplementedError("We only support ImageNet Now.")
    nb_classes=2
    return dataset, nb_classes


def build_transform(is_train, config):
    t=[]
    t.append(
        transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                          interpolation=_pil_interp(config.DATA.INTERPOLATION))
    )
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


class FaceDataset(Dataset):

    def __init__(self,real_dir,fake_dir,real_wavelet_dir,fake_wavelet_dir,require_image=True,is_train=True,transform=None):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.real_wavelet_dir=real_wavelet_dir
        self.fake_wavelet_dir=fake_wavelet_dir

        self.is_train=is_train
        self.train_class_samples=10000
        self.test_class_samples=60000

        self.transform = transform

        self.require_image=require_image

        # import pdb
        # pdb.set_trace()
    def __len__(self):
        if self.is_train:
            return 2*self.train_class_samples
        else:
            return 2*self.test_class_samples

    def __getitem__(self, idx):

        if self.is_train:
            real= idx<self.train_class_samples
        else:
            real= idx<self.test_class_samples

        if real:
            target=torch.tensor(1,dtype=torch.int64)
        else:
            target=torch.tensor(0,dtype=torch.int64)

        #get_image path and read image
        if self.require_image:
            if self.is_train:
                if real:
                    img_path=os.path.join(self.real_dir,"%05d"%idx+'.png')
                else:
                    if idx-self.train_class_samples<10000:
                        img_path=os.path.join(self.fake_dir,'seed'+"%04d"%(idx-self.train_class_samples)+'.png')
                    else:
                        img_path=os.path.join(self.fake_dir,'seed'+"%05d"%(idx-self.train_class_samples)+'.png')
            else:
                if real:
                    img_path=os.path.join(self.real_dir,"%05d"%(idx+self.train_class_samples)+'.png')
                else:
                    if idx-self.test_class_samples+self.train_class_samples<10000:
                        img_path=os.path.join(self.fake_dir,'seed'+"%04d"%(idx-self.test_class_samples+self.train_class_samples)+'.png')
                    else:
                        img_path=os.path.join(self.fake_dir,'seed'+"%05d"%(idx-self.test_class_samples+self.train_class_samples)+'.png')

            sample=Image.open(img_path)

            if self.transform:
                sample = self.transform(sample)

            return sample,target

        #get wavelet path and read wavelet
        else:
            if self.is_train:
                if real:
                    wavelet_path=os.path.join(self.real_wavelet_dir,"%05d"%idx+'wavelet.pkl')
                else:
                    if idx-self.train_class_samples<10000:
                        wavelet_path=os.path.join(self.fake_wavelet_dir,'seed'+"%04d"%(idx-self.train_class_samples)+'wavelet.pkl')
                    else:
                        wavelet_path=os.path.join(self.fake_wavelet_dir,'seed'+"%05d"%(idx-self.train_class_samples)+'wavelet.pkl')
            else:
                if real:
                    wavelet_path=os.path.join(self.real_wavelet_dir,"%05d"%(idx+self.train_class_samples)+'wavelet.pkl')
                else:
                    if idx-self.test_class_samples+self.train_class_samples<10000:
                        wavelet_path=os.path.join(self.fake_wavelet_dir,'seed'+"%04d"%(idx-self.test_class_samples+self.train_class_samples)+'wavelet.pkl')
                    else:
                        wavelet_path=os.path.join(self.fake_wavelet_dir,'seed'+"%05d"%(idx-self.test_class_samples+self.train_class_samples)+'wavelet.pkl')

            with open(wavelet_path,'rb') as f:
                wavelets=pickle.load(f)

                #resize wavelet to shape of [224,224]
                cA2=torch.tensor([chan[0] for chan in wavelets])
                cH2=torch.tensor([chan[1][0] for chan in wavelets])
                cV2=torch.tensor([chan[1][1] for chan in wavelets])
                cD2=torch.tensor([chan[1][2] for chan in wavelets])
                cH1=torch.tensor([chan[2][0] for chan in wavelets])
                cV1=torch.tensor([chan[2][1] for chan in wavelets])
                cD1=torch.tensor([chan[2][2] for chan in wavelets])

                normalized_wavelet=[normalize(wavelet) for wavelet in [cA2,cH2, cV2, cD2,cH1, cV1, cD1]]

                resized_wavelet=[F.resize(wavelet,224) for wavelet in normalized_wavelet]

                return torch.cat(resized_wavelet),target

def normalize(t):
    return (t-t.mean())/t.std()




