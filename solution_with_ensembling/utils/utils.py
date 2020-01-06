import json
import os
from PIL import Image
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from .config import load_config
from catalyst.dl.utils import load_checkpoint

#rle encoding mask

def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)


def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    if rle == ' -1' or rle == '-1':
        return mask.reshape(width,height)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)

def make_mask(row_id, df, height=1400, width=2100):

    #the mask has to be first read and then reshaped
    fname = df.iloc[row_id].name
    #update the class number here in the future
    labels = df.iloc[row_id][:8].astype('str')
    masks = np.zeros((height, width, 8), dtype=np.float32)
    for idx, label in enumerate(labels.values):
        if label=='nan':
            #do nothing
            pass
        else:
            masks[:,:,idx]=rle2mask(label,height,width)

    #need to resize the mask
    #
    # new_mask=np.zeros((height,width,8),dtype=np.float32)
    # for j in range(8):
    #     #convert the mask into image
    #     mask_image=Image.fromarray(masks[:,:,j])
    #     mask_image=mask_image.resize((width,height),Image.BICUBIC)
    #     w,h=mask_image.size
    #     print(w,h)

    return fname,masks



def post_process(probability, threshold, min_size, height=1400, width=2100):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''

    print("entered post processing")
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((height, width), np.float32)
    num = 0

    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num



''' function to load the model for ensembling '''
def load_model(config_path):
    config = load_config(config_path)
    if 'COLAB_GPU' in os.environ:
        config.work_dir = '/content/drive/My Drive/kaggle_cloud/' + config.work_dir
    elif 'KAGGLE_WORKING_DIR' in os.environ:
        config.work_dir = '/kaggle/working/' + config.work_dir

    if config.checkpoint_path == None:
        config.checkpoint_path = config.work_dir + '/checkpoints/best.pth'
    print(config.checkpoint_path)

    # create segmentation model with pre-trained encoder
    model = getattr(smp, config.model.arch)(
        encoder_name=config.model.encoder,
        encoder_weights=None,
        classes=config.data.num_classes,
        activation=None,
    )
    model.to(config.device)
    model.eval()
    checkpoint = load_checkpoint(config.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
