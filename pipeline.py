import sys
sys.path.append('/home/mariapap/CODE/CLv2_eval/mamba_train/')

import torch
from PIL import Image
import numpy as np
import cv2
import os
import rasterio
import shutil
from tools import *
from shapely import geometry
#from buildings import *
from tqdm import tqdm
from mamba_class import Trainer
import argparse


parser = argparse.ArgumentParser(description="Training on xBD dataset")
parser.add_argument(
        '--cfg', type=str, default='./mamba_train/vssm_small_224.yaml')
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )


args = parser.parse_args()



#/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/NatBasemaps/basemaps_natfuel_test/images/A
#################################################################################################################################

mamba_format = 'vssm_small_224.yaml'
mamba_weights = './model_weights/frp_7868_7814_7922_model.pth'

#################################################################################################################################
dir_before = '/home/mariapap/CODE/Class_Location_v2/May_images/'
dir_after = '/home/mariapap/CODE/Class_Location_v2/July_images/'

ids = os.listdir(dir_before)
#ids_after = os.listdir(dir_after)

#ids = list(set(ids_before) & set(ids_after))

####folders
f_pipeline_result = './PIPELINE_RESULTS'
f_before_registered = f_pipeline_result + '/BEFORE_REGISTERED'
f_pred_before = f_pipeline_result + '/PREDS_BEFORE'
f_pred_after = f_pipeline_result + '/PREDS_AFTER'
f_output = f_pipeline_result + '/OUTPUT'
#f_output_disappear = f_pipeline_result + '/OUTPUT_DISAPPEAR'
##############


if os.path.exists(f_pipeline_result):
    shutil.rmtree(f_pipeline_result)
os.mkdir(f_pipeline_result)

#os.mkdir(f_before)
#os.mkdir(f_after)
os.mkdir(f_before_registered)
os.mkdir(f_pred_before)
os.mkdir(f_pred_after)
os.mkdir(f_output)

trainer = Trainer(mamba_format, mamba_weights, args)


def save_clouds(im1, im2, id):
    im1=im1*255
    im2 = im2*255

    im1 = Image.fromarray(im1)
    im2 = Image.fromarray(im2)

    im1.save('./clouds_before/{}_{}'.format('before', id))
    im2.save('./clouds_after/{}_{}'.format('after', id))



for _, id in enumerate(tqdm(ids)):

    raster_before = rasterio.open(dir_before + id)
    raster_after = rasterio.open(dir_after +  id)


    im_before, im_after, bounds = check_shape_and_resize(raster_before, raster_after)


    buildings_before = trainer.validation(im_before)
    buildings_before = filter_small_contours(buildings_before)


    buildings_after = trainer.validation(im_after)
    buildings_after = filter_small_contours(buildings_after)


    clouds_before = predict_clouds_for_model(np.transpose(im_before, (2,0,1))/255., 3, 'https://highrescloudv2-ml.orbitaleye.nl/api/process/rgb')
    clouds_after = predict_clouds_for_model(np.transpose(im_after, (2,0,1))/255., 3, 'https://highrescloudv2-ml.orbitaleye.nl/api/process/rgb')
    clouds_before, clouds_after = np.array(clouds_before), np.array(clouds_after)

    save_clouds(clouds_before, clouds_after, id)



    clouds_total = clouds_before + clouds_after
    cloud_idx = np.where(clouds_total>0)

##############################################################################################################################################################
#################################################################################################################################################################

    save_tif_coregistered('{}/{}'.format(f_pred_before,'before_' + id), buildings_before*255, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 1)
    save_tif_coregistered('{}/{}'.format(f_pred_after,'after_' +id), buildings_after*255, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 1)


    output_appears = apply_watershed(buildings_before, buildings_after)
    output_disappears = apply_watershed(buildings_after, buildings_before)

    idx2 = np.where(output_disappears==1)
    output_appears[idx2]=2

    output = visualize(output_appears)
    output = np.array(output)

    for ch in range(0, 3):
        output[:,:,ch][cloud_idx] = 0


    save_tif_coregistered('{}/output_{}'.format(f_output,id), output, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 3)















    
