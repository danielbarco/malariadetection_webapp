import numpy as np
from PIL import Image
import cv2
import preprocess
import prediction
import time



imgs_padded_cropped = []
imgs_1024_padded_cropped = []
imgs_small_cropped = []
calc_radiuses = []
files = []
img = cv2.imread('docker/images_thick/20170714_170303.jpg', flags=cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_COLOR)   
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


class_names = ['parasitized', 'uninfected']

# prediction
time_start = time.time()
result, selected_patches = prediction.patient_eval(imgs_1024_padded_cropped, imgs_padded_cropped, model_score_thr = 0.5, save_predictions = True, verbose=True)
prediction_time = time.time() - time_start


