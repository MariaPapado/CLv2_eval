from datetime import datetime, date
from utils import *
import cv2
import psycopg2 as psycopg
from tqdm import tqdm
import os
from PIL import Image
import rasterio.windows
import rasterio.features
import base64
import json
import requests
import pickle
import pyproj
from tools import *
import shutil
import clip
import imagecodecs

from pimsys.regions.RegionsDb import RegionsDb
import orbital_vault as ov
import cosmic_eye_client
from CustomerDatabase import CustomerDatabase
from shapely.geometry import Polygon, MultiPolygon, mapping
import shapely.geometry as geometry


def apply_watershed(label_before, thresh):

#    thresh = np.array(thresh, dtype=np.uint8)
#    D = ndi.distance_transform_edt(thresh)
#    coords = peak_local_max(D, footprint=np.ones((20,20)), labels=thresh)
#    mask = np.zeros(D.shape, dtype=bool)
#    mask[tuple(coords.T)] = True
#    markers, _ = ndi.label(mask)
#    labels = watershed(-D, markers, mask=thresh)

    contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labels = np.zeros_like(thresh)

    for i, contour in enumerate(contours, start=1):  # Start from 1 to avoid background (0)
        cv2.drawContours(labels, [contour], -1, color=i, thickness=-1)  # Fill the contour



    appears = np.zeros((labels.shape[0], labels.shape[1]))

    tp, fn = 0, 0
    for label in np.unique(labels)[1:]:
        # if the label is zero, we are examining the 'background'
        # so simply ignore it  
        idx = np.where(labels==label)
        #print(idx[0].shape)
        before_builds = label_before[idx]
        builds = np.count_nonzero(before_builds==1)
        #print(builds/len(before_builds))

        #print('aaaaaa', builds, len(before_builds), builds/len(before_builds))

        if (builds/len(before_builds))>0.000001:
            tp +=1
        elif (builds/len(before_builds))<0.00001:
            fn +=1
        

    return tp, fn, len(np.unique(labels))-1


ids = os.listdir('./images_test/')
test_url_mamba_cl = 'http://10.10.100.8:8060/api/process'
ids = ['NF_5997911_159.png']
#ids = ['NF_5998272_200.png']

sum_tp, sum_fp, sum_fn, sum_total = 0, 0, 0, 0

for id in ids:
    print(id)
    target_img = Image.open('./images_test/{}'.format(id))
    mask = Image.open('./labels_test/{}'.format(id))
    target_img, mask = np.array(target_img), np.array(mask)/255.
    builds_pred1 = predict_buildings(target_img, test_url_mamba_cl)
    tp, fn, total = apply_watershed(builds_pred1, mask)

    sum_tp += tp
    sum_fn +=  fn
    sum_total += total

    _, fp, _ = apply_watershed(mask, builds_pred1)

    sum_fp += fp

    builds_pred1 = builds_pred1*255
    builds_pred1 = np.array(builds_pred1, dtype=np.uint8)
    builds_pred1 = Image.fromarray(builds_pred1)
    builds_pred1.save('./PREDS/{}'.format(id))

print('tp', sum_tp)
print('fp', sum_fp)
print('fn', sum_fn)
print('total', sum_total)

#tp 675
#fp 82
#fn 93
#total 768
#builds_pred2 = predict_buildings(target_img_2, test_url_mamba_cl)

#builds_pred1 = filter_small_contours(builds_pred1)
#builds_pred2 = filter_small_contours(builds_pred2)


#output_appears = apply_watershed(builds_pred1, builds_pred2)
#output_disappears = apply_watershed(builds_pred2, builds_pred1)

#idx2 = np.where(output_disappears==1)
#output_appears[idx2]=2

#output = visualize(output_appears)
#output = np.array(output)



#save_tif_coregistered_with_params('{}/{}.tif'.format(save_folder,region['id']), output, xparams, region['bounds'], channels = 3)

