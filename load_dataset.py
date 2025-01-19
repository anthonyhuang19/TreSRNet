import glob
import os
from PIL import Image
import random
import numpy as np

from torch import nn
from torchvision import transforms
from torch.utils import data as data
import torch.nn.functional as F

class PairedCaptionDataset(data.Dataset):
    def __init__(
            self,
            root_folders=None,
            tokenizer=None,
            null_text_ratio=0.5,
    ):
        super(PairedCaptionDataset, self).__init__()

        self.null_text_ratio = null_text_ratio
        self.lr_list = []
        self.gt_list = []

        root_folders = root_folders.split(',')
        for root_folder in root_folders:
            lr_path = root_folder + '/lr'
            gt_path = root_folder + '/gt'

            self.lr_list += glob.glob(os.path.join(lr_path, '*.png'))
            self.gt_list += glob.glob(os.path.join(gt_path, '*.png'))

        assert len(self.lr_list) == len(self.gt_list)

        self.img_preproc = transforms.Compose([       
            transforms.ToTensor(),
        ])

        ram_mean = [0.485, 0.456, 0.406]
        ram_std = [0.229, 0.224, 0.225]
        self.ram_normalize = transforms.Normalize(mean=ram_mean, std=ram_std)

        self.tokenizer = tokenizer

    def tokenize_caption(self, caption=""):
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    def __getitem__(self, index):
        # Load ground truth (HR) image
        gt_path = self.gt_list[index]
        gt_img = Image.open(gt_path).convert('RGB')
        gt_img = self.img_preproc(gt_img)
        
        # Load low resolution (LR) image
        lq_path = self.lr_list[index]
        lq_img = Image.open(lq_path).convert('RGB')
        lq_img = self.img_preproc(lq_img)

        # Use empty string for tag if null_text_ratio is greater than random value
        if random.random() < self.null_text_ratio:
            tag = ''
        else:
            tag = ''  # You can define any default tag value if necessary.

        example = dict()
        example["lr_image"] = lq_img.squeeze(0)
        example["hr_image"] = gt_img.squeeze(0)

        # Resize low-resolution image for RAM model
        lq_img = lq_img.squeeze()
        ram_values = F.interpolate(lq_img.unsqueeze(0), size=(384, 384), mode='bicubic')
        ram_values = ram_values.clamp(0.0, 1.0)
        example["ram_values"] = self.ram_normalize(ram_values.squeeze(0))

        return example

    def __len__(self):
        return len(self.gt_list)
