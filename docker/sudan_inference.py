import glob
import os
import itertools
import numpy as np
from PIL import Image
import cv2
import preprocess
import prediction
import time
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# create annotations & img folder if it does not exist
os.makedirs('/workspace/data/SudanData/predictions/cropped', exist_ok=True)
os.makedirs('/workspace/data/SudanData/images/cropped', exist_ok=True)
os.makedirs('/workspace/data/SudanData/metadata/cropped', exist_ok=True)

# get list of all images in folder
img_paths = glob.glob('/workspace/data/SudanData/**/*Thick.png', recursive=True)
img_paths = glob.glob('/workspace/data/SudanData/GS/thick/images/*/0011a_1/*Thick.png', recursive=True)
img_paths = glob.glob('/workspace/data/SudanData/SOR/thick/images/*/0138a_1/*Thick.png', recursive=True)
# img_paths = glob.glob('/workspace/malariadetection_webapp/docker/images_thick/*3.jpg', recursive=True)

# devide the img_paths list into chunks defined by patient id (e.g. 0011a_1)
img_paths_patients = [[path for path in patient_paths] for patient_id, patient_paths in itertools.groupby(img_paths, lambda x: x.split('/')[-2])]
patient_ids = [patient_id for patient_id, patient_paths in itertools.groupby(img_paths, lambda x: x.split('/')[-2])]
class_names = ['parasitized', 'uninfected']

for patient_paths, patient_id in zip(img_paths_patients, patient_ids):
    imgs_padded_cropped = []
    imgs_1024_padded_cropped = []
    imgs_small_cropped = []
    calc_radiuses = []
    files = []
    for img_path in patient_paths:
        img = cv2.imread(img_path, flags=cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_COLOR)   
        height, width, c = img.shape
        reduction_prct = 600/ min([height, width]) 
        img_small = cv2.resize(img, None, fx=reduction_prct, fy=reduction_prct)
        img_small_cropped, xmin, xmax, ymin, ymax  = preprocess.circle_crop(img_small)
        img_cropped =     img[int(xmin * width):int(xmax * width),
                                int(ymin * height) :int(ymax * height)]
        # pad high resolution image
        height_cropped, width_cropped, c = img_cropped.shape
        img_padded_cropped = preprocess.pad(img_cropped, height_cropped, width_cropped)
        img_padded_cropped_pil = Image.fromarray(img_padded_cropped)
        img_1024_padded_cropped= np.array(img_padded_cropped_pil.resize((1024, 1024), Image.Resampling.LANCZOS))
        # img_1024_padded_cropped = img_padded_cropped.resize(img_padded_cropped, (1024, 1024), interpolation = cv2.INTER_AREA)
        imgs_padded_cropped.append(img_padded_cropped)
        imgs_1024_padded_cropped.append(img_1024_padded_cropped)
        imgs_small_cropped.append(img_small_cropped)
        files.append(img_path)
        
    # prediction
    time_start = time.time()
    result, selected_patches, selected_predictions = prediction.patient_eval(imgs_1024_padded_cropped, imgs_padded_cropped,patient_n = patient_id, model_score_thr = 0.5, return_predictions = True, verbose=True)
    prediction_time = time.time() - time_start
    
    # save the cropped predictions annotations as txt files
    for selected_prediction in selected_predictions:
        # get the filename of the image
        filename = files[selected_prediction[0]].split('/')[-1].split('.')[0]
        # get the coordinates of the bounding box
        xmin = selected_prediction[1]
        xmax = selected_prediction[2]
        ymin = selected_prediction[3]
        ymax = selected_prediction[4]
        # save the annotation as txt file
        patient_annotations_path = '/workspace/data/SudanData/predictions/cropped/' + patient_id
        os.makedirs(patient_annotations_path, exist_ok=True)
        with open(patient_annotations_path + filename + '.txt', 'a') as f:
            f.write( '0' + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + '\n')

    # save metadata to text file
    metadata = str(patient_id) + ' ' + str(result) + ' ' + str(len(selected_predictions)) 
    # append patient metadata to metadata text file
    with open('/workspace/data/SudanData/metadata/cropped/metadata.txt', 'a') as f:
        f.write(str(metadata) + '\n')
    # save additional metadata to text file
    metadata_debug = str(patient_id) + ' ' + str(result) + ' ' + str(len(selected_predictions)) + ' ' + str(prediction_time) + ' ' + str(files)
    with open('/workspace/data/SudanData/metadata/cropped/metadata_debug.txt', 'a') as f:
        f.write(str(metadata_debug) + '\n')