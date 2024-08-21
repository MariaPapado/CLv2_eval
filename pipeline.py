import sys
sys.path.append('/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/')

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



for _, id in enumerate(tqdm(ids)):

    raster_before = rasterio.open(dir_before + id)
    raster_after = rasterio.open(dir_after +  id)


    im_before, im_after, bounds = check_shape_and_resize(raster_before, raster_after)

    #try:
    #    im_before_transformed, flag = register_image_pair(im_after, im_before)
    #except:
    #    im_before_transformed, flag = im_after
    #im_before_transformed = im_before
    flag=True
#    im_before = histogram_match(im_before, im_after_transformed)
#    im_after_transformed = histogram_match(im_after_transformed, im_before)

    if flag==True:

    #    save_tif_coregistered('{}/before_transformed_{}'.format(f_before_registered,id), im_before_transformed, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 3)
    #    save_tif_coregistered('{}/before_{}'.format(f_after_registered,id), im_before, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 3)




        buildings_before = trainer.validation(im_before)
        buildings_before = filter_small_contours(buildings_before)
    

        buildings_after = trainer.validation(im_after)
        buildings_after = filter_small_contours(buildings_after)
    ##############################################################################################################################################################
    #################################################################################################################################################################

        save_tif_coregistered('{}/{}'.format(f_pred_before,'before_' + id), buildings_before*255, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 1)
        save_tif_coregistered('{}/{}'.format(f_pred_after,'after_' +id), buildings_after*255, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 1)


        output_appears = apply_watershed(buildings_before, buildings_after)
        output_disappears = apply_watershed(buildings_after, buildings_before)

        idx2 = np.where(output_disappears==1)
        output_appears[idx2]=2

        output = visualize(output_appears)

        save_tif_coregistered('{}/output_{}'.format(f_output,id), output, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 3)















    
