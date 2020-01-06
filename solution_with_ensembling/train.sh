#!/usr/bin/env bash
#
# python train.py --config=config/seg/017_efnet_b3_Unet_img320_cutout5_aug_fold0.yml
# python train.py --config=config/seg/017_efnet_b3_Unet_img320_cutout5_aug_fold1.yml
# python train.py --config=config/seg/017_efnet_b3_Unet_img320_cutout5_aug_fold2.yml
# python train.py --config=config/seg/017_efnet_b3_Unet_img320_cutout5_aug_fold3.yml
# python train.py --config=config/seg/017_efnet_b3_Unet_img320_cutout5_aug_fold4.yml

python train.py --config=config/seg/030_efnet_b0_Unet_bs16_half_cosine_fold0.yml
python train.py --config=config/seg/030_efnet_b0_Unet_bs16_half_cosine_fold1.yml
python train.py --config=config/seg/030_efnet_b0_Unet_bs16_half_cosine_fold2.yml
python train.py --config=config/seg/030_efnet_b0_Unet_bs16_half_cosine_fold3.yml
python train.py --config=config/seg/030_efnet_b0_Unet_bs16_half_cosine_fold4.yml
