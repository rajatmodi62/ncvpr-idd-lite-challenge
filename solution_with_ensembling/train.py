from transforms import get_transforms
from schedulers import get_scheduler
from losses import get_loss
from optimizers import get_optimizer
from dataset_loader import make_loader
import argparse
import os
import warnings
from utils.config import load_config, save_config
from utils.callbacks import CutMixCallback

#model imports
import segmentation_models_pytorch as smp

#using catalyst dl wrapper for the training loop
from catalyst.dl import SupervisedRunner
from catalyst.utils import get_device
from catalyst.dl.callbacks import DiceCallback, IouCallback, CheckpointCallback, MixupCallback, EarlyStoppingCallback, OptimizerCallback, CriterionCallback

def run(config_file):
    config = load_config(config_file)
    #set up the environment flags for working with the KAGGLE GPU OR COLAB_GPU
    if 'COLAB_GPU' in os.environ:
        config.work_dir = '/content/drive/My Drive/kaggle_cloud/' + config.work_dir
    elif 'KAGGLE_WORKING_DIR' in os.environ:
        config.work_dir = '/kaggle/working/' + config.work_dir
    print('working directory:', config.work_dir)

    #save the configuration to the working dir
    if not os.path.exists(config.work_dir):
        os.makedirs(config.work_dir, exist_ok=True)
    save_config(config, config.work_dir + '/config.yml')

    #Enter the GPUS you have,
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    all_transforms = {}
    all_transforms['train'] = get_transforms(config.transforms.train)
    #our dataset has an explicit validation folder, use that later.
    all_transforms['valid'] = get_transforms(config.transforms.test)

    print("before rajat config",config.data.height,config.data.width)
    #fetch the dataloaders we need
    dataloaders = {
        phase: make_loader(
            data_folder=config.data.train_dir,
            df_path=config.data.train_df_path,
            phase=phase,
            img_size=(config.data.height, config.data.width),
            batch_size=config.train.batch_size,
            num_workers=config.num_workers,
            idx_fold=config.data.params.idx_fold,
            transforms=all_transforms[phase],
            num_classes=config.data.num_classes,
            pseudo_label_path=config.train.pseudo_label_path,
            debug=config.debug
        )
        for phase in ['train', 'valid']
    }

    #creating the segmentation model with pre-trained encoder
    '''
    dumping the parameters for smp library
    encoder_name: str = "resnet34",
    encoder_depth: int = 5,
    encoder_weights: str = "imagenet",
    decoder_use_batchnorm: bool = True,
    decoder_channels: List[int] = (256, 128, 64, 32, 16),
    decoder_attention_type: Optional[str] = None,
    in_channels: int = 3,
    classes: int = 1,
    activation: Optional[Union[str, callable]] = None,
    aux_params: Optional[dict] = None,
    '''
    model = getattr(smp, config.model.arch)(
        encoder_name=config.model.encoder,
        encoder_weights=config.model.pretrained,
        classes=config.data.num_classes,
        activation=None,
    )

    #fetch the loss
    criterion = get_loss(config)
    params = [
        {'params': model.decoder.parameters(), 'lr': config.optimizer.params.decoder_lr},
        {'params': model.encoder.parameters(), 'lr': config.optimizer.params.encoder_lr},
    ]
    optimizer = get_optimizer(params, config)
    scheduler = get_scheduler(optimizer, config)

    '''
    dumping the catalyst supervised runner
    https://github.com/catalyst-team/catalyst/blob/master/catalyst/dl/runner/supervised.py

    model (Model): Torch model object
    device (Device): Torch device
    input_key (str): Key in batch dict mapping for model input
    output_key (str): Key in output dict model output
        will be stored under
    input_target_key (str): Key in batch dict mapping for target
    '''

    runner = SupervisedRunner(model=model, device=get_device())

    #@pavel,srk,rajat,vladimir,pudae check the IOU and the Dice Callbacks

    callbacks = [DiceCallback(), IouCallback()]

    #adding patience
    if config.train.early_stop_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            patience=config.train.early_stop_patience))

    #thanks for handling the distributed training
    '''
    we are gonna take zero_grad after accumulation accumulation_steps
    '''
    if config.train.accumulation_size > 0:
        accumulation_steps = config.train.accumulation_size // config.train.batch_size
        callbacks.extend(
            [CriterionCallback(),
             OptimizerCallback(accumulation_steps=accumulation_steps)]
        )

    # to resume from check points if exists
    if os.path.exists(config.work_dir + '/checkpoints/best.pth'):
        callbacks.append(CheckpointCallback(
            resume=config.work_dir + '/checkpoints/last_full.pth'))


    '''
    pudae добавь пожалуйста обратный вызов
    https://arxiv.org/pdf/1710.09412.pdf
    **srk adding the mixup callback
    '''
    if config.train.mixup:
        callbacks.append(MixupCallback())
    if config.train.cutmix:
        callbacks.append(CutMixCallback())

    '''@rajat implemented cutmix, a wieghed combination of cutout and mixup '''
    callbacks.append(MixupCallback())
    callbacks.append(CutMixCallback())

    '''
    rajat introducing training loop
    https://github.com/catalyst-team/catalyst/blob/master/catalyst/dl/runner/supervised.py
    take care of the nvidias fp16 precision
    '''
    print(config.work_dir)
    print(config.train.minimize_metric)
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=dataloaders,
        logdir=config.work_dir,
        num_epochs=config.train.num_epochs,
        main_metric=config.train.main_metric,
        minimize_metric=config.train.minimize_metric,
        callbacks=callbacks,
        verbose=True,
        fp16=False,
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_file',
                        help='configuration file path',
                        default=None, type=str)
    return parser.parse_args()


def main():
    print('train Segmentation model.')
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')
    print('load config from {}'.format(args.config_file))
    run(args.config_file)


if __name__ == '__main__':
    main()
