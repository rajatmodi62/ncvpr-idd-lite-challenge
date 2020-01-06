import os

import jpeg4py as jpeg
from PIL import Image
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils import make_mask

CLASSES = {'road,parking,drivable fallback': 0,
           'sidewalk,rail track,non-drivable fallback': 1,
           'person,animal,rider': 2,
           'motorcycle,bicycle,autorickshaw,car,bus,truck,caravan,trailer,train,vehicle': 3,
           'curb,wall,fence,guard rail, traffic light, sign, pole': 4,
           'building,bridge,tunnel,vegetation': 5,
           'sky,fallback background': 6,
           'unlabeled, ego vehicle, rectification border, out of roi, license plate': 7,
           }
INV_CLASSES = {0: 'road,parking,drivable fallback',
               1: 'sidewalk,rail track,non-drivable fallback',
               2: 'person,animal,rider',
               3: 'motorcycle,bicycle,autorickshaw,car,bus,truck,caravan,trailer,train,vehicle',
               4: 'curb,wall,fence,guard rail, traffic light, sign, pole',
               5: 'building,bridge,tunnel,vegetation',
               6: 'sky,fallback background',
               7: 'unlabeled, ego vehicle, rectification border, out of roi, license plate',

               }

#
class TrainDataset(Dataset):
    def __init__(self, df, data_folder, phase, transforms, img_size, num_classes=4, return_fnames=False):
        self.df = df
        self.root = data_folder
        self.phase = phase
        self.transforms = transforms
        self.img_size = img_size
        self.fnames = self.df.index.tolist()
        self.num_classes = num_classes
        self.return_fnames = return_fnames
#
    def __getitem__(self, idx):

        image_id, mask = make_mask(
            idx, self.df, height=self.img_size[0], width=self.img_size[1])
        image_path = os.path.join(self.root, image_id)
        image_path=image_path+'.jpg'
        img = jpeg.JPEG(str(image_path)).decode()
        augmented = self.transforms(image=img,mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        mask = mask[0].permute(2, 0, 1)
        #print("size of the output tensor is ",mask.size())
        #return torch.randn(3,128,256),torch.randn(8,128,256)
        mask=mask/255

        if self.return_fnames:
            return img, mask, image_id
        else:
            return img, mask

    def __len__(self):
        return len(self.fnames)
#
#
# class ClsTrainDataset(Dataset):
#     def __init__(self, df, data_folder, phase, transforms, num_classes=4, return_fnames=False):
#         self.df = df
#         self.root = data_folder
#         self.phase = phase
#         self.transforms = transforms
#         self.fnames = self.df.index.tolist()
#         self.num_classes = num_classes
#         self.return_fnames = return_fnames
#
#     def __getitem__(self, idx):
#         image_id = self.df.iloc[idx].name
#         if self.num_classes == 4:
#             label = self.df.iloc[idx, :4].notnull().values.astype('f')
#         else:
#             label = np.zeros(5)
#             label[1:5] = self.df.iloc[idx, :4].notnull()
#             label[0] = label[1:5].sum() <= 0
#             label = label.astype('f')
#
#         image_path = os.path.join(self.root, image_id)
#         img = jpeg.JPEG(image_path).decode()
#         augmented = self.transforms(image=img)
#         img = augmented['image']
#         if self.return_fnames:
#             return img, label, image_id
#         else:
#             return img, label
#
#     def __len__(self):
#         return len(self.fnames)
#
#
class TestDataset(Dataset):
    def __init__(self, root, df, transforms):
        self.root = root
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transforms = transforms

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        image_path = os.path.join(self.root, fname)
        image_path=image_path+'.jpg'

        img = jpeg.JPEG(image_path).decode()
        images = self.transforms(image=img)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples
#
#
# class FilteredTestDataset(Dataset):
#     def __init__(self, root, df, transform):
#         self.root = root
#         df = df[(df > 0.5).sum(axis=1) > 0]  # screen no defect images
#         self.fnames = df.index.tolist()
#         self.num_samples = len(self.fnames)
#         self.transform = transform
#
#     def __getitem__(self, idx):
#         fname = self.fnames[idx]
#         image_path = os.path.join(self.root, fname)
#         img = jpeg.JPEG(image_path).decode()
#         images = self.transform(image=img)["image"]
#         return fname, images
#
#     def __len__(self):
#         return self.num_samples
#
#
# def expand_path(p):
#     train_dir = Path('../input/understanding_cloud_organization/train_images')
#     test_dir = Path('../input/understanding_cloud_organization/test_images')
#     if (train_dir / p).exists():
#         return train_dir / p
#     elif (test_dir / p).exists():
#         return test_dir / p


def make_loader(
        data_folder,
        df_path,
        phase,
        img_size=(1400, 2100),
        batch_size=8,
        num_workers=2,
        idx_fold=None,
        transforms=None,
        num_classes=4,
        pseudo_label_path=None,
        task='seg',  # choice of ['cls', 'seg'],
        return_fnames=False,
        debug=False,
):
    if debug:
        num_rows = 100
    else:
        num_rows = None

    print("godzilla",img_size)
    df = pd.read_csv(df_path, nrows=num_rows)
    if phase == 'test':
        print("test")
        image_dataset = TestDataset(data_folder, df, transforms)
        is_shuffle = False
        #adding pass ,remove later
        pass

    elif phase == 'filtered_test':

        #code will not go here
        # df = pd.read_csv(df_path, nrows=num_rows, index_col=0)
        # image_dataset = FilteredTestDataset(data_folder, df, transforms)
        # is_shuffle = False
        #adding pass ,remove later
        pass

    else:  # train or valid
        if os.path.exists('dataset/train_folds.csv'):
            folds = pd.read_csv(
                'dataset/train_folds.csv', index_col='ImageId', nrows=num_rows)
        else:
            raise Exception('You need to run split_folds.py beforehand.')

        if phase == "train":

            folds = folds[folds['fold'] != idx_fold]

    # skip the pseudolabelling code for now
    #         if os.path.exists(pseudo_label_path):
    #             pseudo_df = pd.read_csv(pseudo_label_path)
    #             #pseudo_df['ImageId'], pseudo_df['ClassId'] = zip(*pseudo_df['Image_Label'].str.split('_'))
    #             #pseudo_df['ClassId'] = pseudo_df['ClassId'].astype(int)
    #             pseudo_df['ImageId'] = pseudo_df['Image_Label'].map(
    #                 lambda x: x.split('_')[0])
    #             pseudo_df['ClassId'] = pseudo_df['Image_Label'].map(
    #                 lambda x: x.split('_')[1]).map(CLASSES).astype(int)
    #             pseudo_df['exists'] = pseudo_df['EncodedPixels'].notnull().astype(
    #                 int)
    #             pseudo_df['ClassId0'] = [
    #                 row.ClassId if row.exists else 0 for row in pseudo_df.itertuples()]
    #             pv_df = pseudo_df.pivot(
    #                 index='ImageId', columns='ClassId', values='EncodedPixels')
    #             folds = pd.concat([folds, pv_df], axis=0)
    #
            is_shuffle = True

        else:

            folds = folds[folds['fold'] == idx_fold]
            is_shuffle = False

        if task == 'seg':
            image_dataset = TrainDataset(
                folds, data_folder, phase, transforms, img_size, num_classes, return_fnames)
    #     else:
    #         image_dataset = ClsTrainDataset(
    #             folds, data_folder, phase, transforms, num_classes, return_fnames)
    #
    image_dataset[1]

    return DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=is_shuffle,
    )
