from transforms import get_transforms
from dataset_loader import make_loader, INV_CLASSES
from utils.config import load_config
from utils.functions import resize_batch_images
from utils.utils import load_model,post_process
from utils import predict_batch
from models import MultiSegModels
from PIL import Image

import argparse
import json
import os
import warnings
import cv2

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

''' set up submission height and width'''
SUB_HEIGHT, SUB_WIDTH = 128, 256

def ensemble():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # ------------------------------------------------------------------------------------------------------------
    # parmeters and configs
    # ------------------------------------------------------------------------------------------------------------
    config_paths320 = [
        'config/seg/017_efnet_b3_Unet_img320_cutout5_aug_fold0.yml',
        'config/seg/017_efnet_b3_Unet_img320_cutout5_aug_fold1.yml',
        'config/seg/017_efnet_b3_Unet_img320_cutout5_aug_fold2.yml',
        'config/seg/017_efnet_b3_Unet_img320_cutout5_aug_fold3.yml',
        'config/seg/017_efnet_b3_Unet_img320_cutout5_aug_fold4.yml',
        'config/seg/030_efnet_b0_Unet_bs16_half_cosine_fold0.yml',
        'config/seg/030_efnet_b0_Unet_bs16_half_cosine_fold1.yml',
        'config/seg/030_efnet_b0_Unet_bs16_half_cosine_fold2.yml',
        'config/seg/030_efnet_b0_Unet_bs16_half_cosine_fold3.yml',
        'config/seg/030_efnet_b0_Unet_bs16_half_cosine_fold4.yml',
    ]
    #see there use later on
    # LABEL_THRESHOLDS = [0.68, 0.69, 0.69, 0.67]
    # MASK_THRESHOLDS = [0.31, 0.36, 0.31, 0.34]
    LABEL_THRESHOLDS = [0.67, 0.67, 0.67, 0.67,0.67,0.67,0.67,0.50]
    MASK_THRESHOLDS = [0.31, 0.31, 0.31, 0.31,0.31,0.31,0.31,0.31]
    # MIN_SIZES = [7500, 7500, 7500, 7500,7500,7500,7500,7500]
    MIN_SIZES = [0,0,0,0,0,0,0,0]
    WEIGHTS = [0.5, 0.5]
    # ------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------
    config = load_config('config/base_config.yml')

    ''' load the models for evaluation'''
    def get_model_and_loader(config_paths):
        config = load_config(config_paths[0])
        models = []
        for c in config_paths:
            models.append(load_model(c))

        model = MultiSegModels(models)

        print(config.data.test_dir)

        testloader = make_loader(
            data_folder=config.data.test_dir,
            df_path=config.data.sample_submission_path,
            phase='test',
            img_size=(config.data.height, config.data.width),
            batch_size=config.test.batch_size,
            num_workers=config.num_workers,
            transforms=get_transforms(config.transforms.test)
        )
        return model, testloader

    model320, loader320=get_model_and_loader(config_paths320)

    predictions = []

    with torch.no_grad():
        for (batch_fnames320, batch_images320) in tqdm(loader320):
            batch_images320 = batch_images320.to(config.device)
            print(batch_images320.size())
            batch_preds320 = predict_batch(
                model320, batch_images320, tta=config.test.tta)

            #resize the images from multi resolution models
            batch_preds320 = resize_batch_images(
                batch_preds320, SUB_HEIGHT, SUB_WIDTH)
            batch_preds=batch_preds320



            batch_labels320 = torch.nn.functional.adaptive_max_pool2d(torch.sigmoid(
                torch.Tensor(batch_preds320)), 1).view(batch_preds320.shape[0], -1)
            #print(batch_labels320)

            #change batch_labels by weighing factor later on
            batch_labels =batch_labels320

            print("batch_preds",batch_preds.shape)
            print("batch_labels",batch_labels.size())


            for fname, preds, labels in zip(batch_fnames320, batch_preds, batch_labels):
                print("ad",labels.size())
                for cls in range(8):
                    if labels[cls] <= LABEL_THRESHOLDS[cls]:
                        pred = np.zeros(preds[cls, :, :].shape)
                        print("setting 0",cls)
                    else:
                        if cls==7:
                            print("ok")
                        #print("probability",preds[cls, :, :])

                        pred, _ = post_process(
                            preds[cls, :, :], MASK_THRESHOLDS[cls], MIN_SIZES[cls], height=SUB_HEIGHT, width=SUB_WIDTH)
                        cls_name = INV_CLASSES[cls]
                        print(fname)
                    dump_name='results/masks/experiment1/'+fname+'class_'+str(cls)+'.jpg'
                    print(dump_name)
                    cv2.imwrite(dump_name,  pred* 255)

if __name__ == '__main__':
    ensemble()
