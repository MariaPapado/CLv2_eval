import rasterio
import cv2
import numpy as np
import torch
from skimage.morphology import erosion
from skimage.morphology import disk
from skimage.morphology import area_opening
from PIL import Image

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

import base64
import json
import requests



def filter_small_contours(im):
    im = np.array(im, dtype=np.uint8)
    threshold_area = 50     #threshold area 
    contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)   
    k_contours=[]

    for cnt in contours:    
#        print(cnt)    
        area = cv2.contourArea(cnt)         
#        print(area)
        if area > threshold_area:
            #Put your code in here
            k_contours.append(cnt)

    fim = np.zeros((im.shape[0], im.shape[1]))
    cv2.fillPoly(fim, pts =k_contours, color=(1,1,1))
#    print(fim.shape)
#    cv2.imwrite(id[:-4] + '__filter.png', fim)
    return fim


def check_shape_and_resize(raster_before, raster_after):
    h_before, w_before = raster_before.height, raster_before.width
    h_after, w_after = raster_after.height, raster_after.width
    img_before = raster_before.read()
    img_after = raster_after.read()
    img_before, img_after = np.transpose(img_before, (1,2,0)), np.transpose(img_after, (1,2,0))
    if (h_before + w_before) < (h_after + w_after):
        img_after = cv2.resize(img_after, (w_before, h_before), cv2.INTER_NEAREST)
        return img_before, img_after, raster_before.bounds  
    elif (h_before + w_before) > (h_after + w_after):    
        img_before = cv2.resize(img_before, (w_after, h_after), cv2.INTER_NEAREST)
        return img_before, img_after, raster_after.bounds 
    else:
        return img_before, img_after, raster_before.bounds




def save_tif_coregistered(filename, image, poly, channels=1, factor=1):
    height, width = image.shape[0], image.shape[1]
    geotiff_transform = rasterio.transform.from_bounds(poly.bounds[0], poly.bounds[1],
                                                       poly.bounds[2], poly.bounds[3],
                                                       width/factor, height/factor)

    new_dataset = rasterio.open(filename, 'w', driver='GTiff',
                                height=height/factor, width=width/factor,
                                count=channels, dtype='uint8',
                                crs='+proj=latlong',
                                transform=geotiff_transform)

    # Write bands
    if channels>1:
     for ch in range(0, image.shape[2]):
       new_dataset.write(image[:,:,ch], ch+1)
    else:
       new_dataset.write(image, 1)
    new_dataset.close()

    return True

 
def visualize(dam):
    dam = np.array(dam, dtype=np.uint8)
    dam = Image.fromarray(dam)
    dam.putpalette([0, 0, 0,
                    0, 255, 0,
                    255, 0, 0])
    dam = dam.convert('RGB')
    dam = np.asarray(dam)

    return dam
 

def apply_watershed(label_before, thresh):

    thresh = np.array(thresh, dtype=np.uint8)
    D = ndi.distance_transform_edt(thresh)
    coords = peak_local_max(D, footprint=np.ones((20, 20)), labels=thresh)
    mask = np.zeros(D.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-D, markers, mask=thresh)



#    D = ndi.distance_transform_edt(thresh)
#    localMax = peak_local_max(D,  min_distance=20, labels=thresh)

#    markers = ndi.label(localMax, structure=np.ones((3, 3)))[0]
#    labels = watershed(-D, markers, mask=thresh)


    #print(np.unique(labels))

    #cv2.imwrite('wat.png', labels*4)

#    print('nppppp', labels.shape, np.unique(labels))
    appears = np.zeros((labels.shape[0], labels.shape[1]))

    for label in np.unique(labels)[1:]:
        # if the label is zero, we are examining the 'background'
        # so simply ignore it  
        idx = np.where(labels==label)
        #print(idx[0].shape)
        before_builds = label_before[idx]
        builds = np.count_nonzero(before_builds==1)

        if (builds/len(before_builds))<0.02:
            appears[idx] = 1

    return appears
    #cv2.imwrite('appears.png', appears)



def predict_clouds_for_model(image, bands, test_url_cloud):
    image = np.moveaxis(image[:bands], 0, -1)
    image_d = base64.b64encode(np.ascontiguousarray(image.astype(np.float32)))

    response = requests.post(test_url_cloud,json={"image": image_d.decode(), "shape": json.dumps(list(image.shape))})

    if response.ok:
        response_result = json.loads(response.text)
        response_result_data = base64.b64decode(response_result["result"])
        result = np.frombuffer(response_result_data, dtype=np.uint8)
        cloud_mask = result.reshape(image.shape[:2])

    else:
        print("error", response)
        raise ValueError("error wrong response from cloud server")
    return cloud_mask

    

def get_wms_image_by_id(image_id, creds_mapserver, settings_db):
    image_broker_url = 'https://maps.orbitaleye.nl/image-broker/products?id={}&_skip=0'.format(image_id)
    response = requests.get(image_broker_url, auth=(creds_mapserver['username'], creds_mapserver['password']))
    wms_image = json.loads(response.text)[0]

    with RegionsDb(settings_db) as database:
        wms_image = database.get_optical_image_by_wms_layer_name(wms_image['wms_layer_name'])
        
    return wms_image
