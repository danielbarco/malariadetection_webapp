import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from tensorflow import expand_dims
from tensorflow.nn import softmax

from object_detection.utils import config_util
from object_detection.builders import model_builder

import time
from PIL import Image
from six import BytesIO
import cv2
import random
import os
from matplotlib import pyplot as plt



def initiate_detection_model(path_cfg, path_ckpt):
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(path_cfg)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(path_ckpt).expect_partial()
    return detection_model

def initiate_classification_model(model_path):
    classification_model = load_model(model_path)
    return classification_model

crop_size = 1024

pf_path_cfg = 'malariadetection_webapp/docker/models/fasterrcnn_inception_v2_1024_PF_train150000/thick_PF_wh64.config'
pf_path_ckpt = 'malariadetection_webapp/docker/models/fasterrcnn_inception_v2_1024_PF_train150000/ckpt-151'
pf_model_path = 'malariadetection_webapp/docker/models/PFU_256_resnet50_custom.h5'
pf_classification_model = initiate_classification_model(pf_model_path)
pf_detection_model = initiate_detection_model(pf_path_cfg, pf_path_ckpt)

@tf.function
def pf_detection_model_fn(image):
    """Detect objects in image."""

    image, shapes = pf_detection_model.preprocess(image)
    prediction_dict = pf_detection_model.predict(image, shapes)
    detections = pf_detection_model.postprocess(prediction_dict, shapes)

    return detections

pv_path_cfg = 'malariadetection_webapp/docker/models/fasterrcnn_inception_v2_1024_PV_train150000/thick_PV_wh64.config'
pv_path_ckpt = 'malariadetection_webapp/docker/models/fasterrcnn_inception_v2_1024_PV_train150000/ckpt-151'
pv_model_path = 'malariadetection_webapp/docker/models/PV_256_resnet50_custom.h5'
pv_classification_model = initiate_classification_model(pv_model_path)
pv_detection_model = initiate_detection_model(pv_path_cfg, pv_path_ckpt)

@tf.function
def pv_detection_model_fn(image):
    """Detect objects in image."""

    image, shapes = pv_detection_model.preprocess(image)
    prediction_dict = pv_detection_model.predict(image, shapes)
    detections = pv_detection_model.postprocess(prediction_dict, shapes)

    return detections


pvf_model_path = 'malariadetection_webapp/docker/models/PVF_256_resnet50_custom.h5'
pvf_classification_model = initiate_classification_model(pvf_model_path)

# logging
log_filename = "/malariadetection_webapp/logging/docker/logs.csv"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
if not os.path.isfile(log_filename):
    df_logs = pd.DataFrame(columns = ['patient_n', 'img', 'model_score_thr', \
                'PV_detected', 'PV_selected', 'PV_prob' ,\
                'PF_detected', 'PF_selected', 'PF_prob', 'dataset', 'result'])
    df_logs.to_csv(log_filename, index=False)


def return_selected_predictions(patches_selected, detections, patches_resized):
    """
    return the selected patches 
    :param patches_selected: list of selected patches
    :type patches_selected: list
    :param detections: dictionary of detections
    :type detections: dict
    :param patches_resized: list of resized patches
    :type patches_resized: list
    :return: None
    """
    
    # select the same patches from detections and save to a csv in the logging folder
    patch_idx = [np.where(np.all(patch == patches_resized, axis=(1, 2, 3)))[0][0] for patch in patches_selected]
    detections_selected = {key: value[patch_idx].tolist() for key, value in detections.items()}
    return detections_selected
    

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: the file path to the image

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections#, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn



def object_detection(img_np, detection_model_fn): 
    '''Predicts object detection on image
    Args:
        img_np (np.array): image 1024 x 1024 with white blood cell diameter of 
        path_cfg (str): path to tf object detection model config file
        path_ckpt (str): path to tf checkpoint
    Returns:
        detections (dictionary): tf obect detections output'''



    input_tensor = tf.convert_to_tensor(np.expand_dims(img_np, 0), dtype=tf.float32)

    detections_raw = detection_model_fn(input_tensor)
    
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections_raw.pop('num_detections'))


    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections_raw.items()}

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    return detections

def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box
    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
    Returns:
        float: value of the IoU for the two boxes.
    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou

def cut_patches(detections, img_full, model_score_thr = 0.5, input_img_size = 1024, out_img_size = 256):
    '''Takes tf objecdetection scores and boxes and cuts the patches from the full image.
    This allows for higher resolution patches.
    Args:
        detections (dictionary): tf obect detections output
        img_full (numpy array): image object as numpy array
    Returns:
        patches_resized (list): list of potentally infected patches'''

    # sort detections_scores & detection_boxes
    arg_sort = np.argsort(detections['detection_scores'])
    detections['detection_scores'] = np.array(detections['detection_scores'])[arg_sort]
    detections['detection_boxes'] = np.array(detections['detection_boxes'])[arg_sort]
    
    
    # keep detections with detection_socres above the model_score_thr
    thr_detections = detections.copy()
    thr_detections['detection_boxes'] = detections['detection_boxes'][detections['detection_scores'] > model_score_thr]
    
    # change bbx measures to pixel
    tf_detections = np.array([np.array([int(bbx[3]* input_img_size), int(bbx[2]* input_img_size), int(bbx[1]* input_img_size), int(bbx[0]* crop_size)]) 
                              for bbx in thr_detections['detection_boxes']])
      
    # check the full image
    height_scale = img_full.shape[0] / input_img_size
    width_scale = img_full.shape[1] / input_img_size

#     # only non-overlapping detection_boxes
#     product_boxes = list(itertools.product(detection_boxes, detection_boxes))
#     detection_boxes_cleaned = [product[0] for product in product_boxes if calc_iou_individual(product[0], product[1]) < 0.1]
#     print(len(detection_boxes_cleaned))
    
    patches_resized = []
    
    # image[start_row:end_row, start_column:end_column] e.g. image[30:250, 100:230] or [x1:x2, y1:y2]
    for bbx in tf_detections:
        patch = img_full[int(bbx[3] * width_scale):int(bbx[1] * width_scale),
                    int(bbx[2] * height_scale):int(bbx[0] * height_scale)]
        if len(patch) != 0 and len(patch.shape) == 3 and patch.shape[0] > 2 and patch.shape[1] > 2:
            patch_resized = cv2.resize(patch, (out_img_size, out_img_size), interpolation = cv2.INTER_CUBIC)
            # patch_resized = tf.image.resize(patch, [out_img_size, out_img_size])
            patches_resized.append(patch_resized)

    return patches_resized


def tf_classification(patches, model):
    '''function for tf model; resizes and pads an image array and then returns the prediction i.e. the most likely class
    Args:
        patches (list): list of image object as numpy array
    Returns:
        y_pred (list): list of most probable class'''
    
    return model.predict(np.array(patches))


def patient_eval(imgs_np, imgs_full, patient_n = -1, model_score_thr = 0.5, dataset = None, return_predictions= False, verbose = False):
    total_pf = []
    total_pv = []
    total_u = []
    pf_all_selected_patches = []
    pv_all_selected_patches = []
    selected_patches = []
    df_logs = pd.read_csv(log_filename)
    total_start_time = time.time()

    for img_np, img_full in zip(imgs_np, imgs_full):
        
        #load images
        # img_np_bgr = cv2.imread(img_path, flags=cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_COLOR)   
        # img_full_bgr = cv2.imread(img_full_path, flags=cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_COLOR) 
        # img_np = img_np_bgr[...,::-1]
        # img_full = img_full_bgr[...,::-1]

        pil_img_np = Image.fromarray(img_np,"RGB")
        # if folder does not exist create it
        os.makedirs(os.path.dirname("malariadetection_webapp/logging/docker/imgs/img_np.jpg"), exist_ok=True)
        pil_img_np.save("malariadetection_webapp/logging/docker/imgs/img_np.jpg")
        pil_img_full = Image.fromarray(img_full,"RGB")
        pil_img_full.save("malariadetection_webapp/logging/docker/imgs/img_full.jpg")

        # Falciparum object detection & filter with ResNet50
        start_time = time.time()
        detections = object_detection(img_np, pf_detection_model_fn)
        pf_patches_resized = cut_patches(detections, img_full, model_score_thr = 0.5, input_img_size = 1024, out_img_size = 256)

        if verbose:
            duration = time.time() - start_time
            print(f'PF detection took {round(duration, 2)} seconds, {len(pf_patches_resized)} detections')
            for n, patch in enumerate(pf_patches_resized[0:5]):
                pil_patch = Image.fromarray(patch,"RGB")
                pil_patch.save(f"malariadetection_webapp/logging/docker/imgs/pf_patch_{n}.jpg")
        start_time = time.time()
        if len(pf_patches_resized) > 0:
            y_pred = tf_classification(pf_patches_resized, pf_classification_model)
            pf_patches_selected = [patch for patch, label in zip(pf_patches_resized, y_pred) if np.argmax(softmax(label)) == 1]
            
        else:
            pf_patches_selected = []    
        
        if verbose:
            duration = time.time() - start_time
            print(f'PF ResNet50 took {round(duration, 2)} seconds, {len(pf_patches_selected)} detections')

        # Vivax object detection & filter with ResNet50
        start_time = time.time()
        detections = object_detection(img_np, pv_detection_model_fn)
        pv_patches_resized = cut_patches(detections, img_full, model_score_thr = model_score_thr, input_img_size = 1024, out_img_size = 256)
        
        if verbose:
            duration= time.time() - start_time   
            print(f'PV detection took {round(duration, 2)} seconds, {len(pv_patches_resized)} detections')
        
        start_time = time.time()
        if len(pv_patches_resized) > 0:
            y_pred = tf_classification(pv_patches_resized, pv_classification_model)
            pv_patches_selected = [patch for patch, label in zip(pv_patches_resized, y_pred) if np.argmax(softmax(label)) == 1]

        else:
            pv_patches_selected = []
        
        if verbose: 
            duration= time.time() - start_time   
            print(f'PV ResNet50 took {round(duration, 2)} seconds, {len(pv_patches_selected)} detections')
        
        # for logging purposes
        avg_pf, avg_pv = -1, -1

        if len(pf_patches_selected) > 1 and len(pv_patches_selected) > 1:
            start_time = time.time()

            # vivax vs. falciparum
            pvf_patches_selected = pf_patches_selected + pv_patches_selected
            # y_prob = tf_classification(pvf_patches_selected, pvf_classification_model)
            # avg_pf = np.mean(y_prob, axis=0)[0]     
            # avg_pv = np.mean(y_prob, axis=0)[1]       

            y_prob_pf = tf_classification(pf_patches_selected, pvf_classification_model)
            avg_pf = np.sum(y_prob_pf, axis=0)[0] / len(pvf_patches_selected)
            y_prob_pv = tf_classification(pv_patches_selected, pvf_classification_model)
            avg_pv = np.sum(y_prob_pv, axis=0)[1]/ len(pvf_patches_selected)
            if verbose:
                print('PVF len pv', len(y_prob_pv) , 'len pf', len(y_prob_pf))

            if avg_pv > avg_pf:
                img_result = 'pv'
                total_pv.append(len(pv_patches_selected))
                total_pf.append(0)
                total_u.append(0)
                pv_all_selected_patches = pv_all_selected_patches + pv_patches_selected
                if return_predictions:
                    selected_predictions = return_selected_predictions( pv_patches_selected, detections, pv_patches_resized)

            elif avg_pf > avg_pv:
                img_result = 'pf'
                total_pf.append(len(pf_patches_selected))
                total_pv.append(0)
                total_u.append(0)
                pf_all_selected_patches = pf_all_selected_patches + pf_patches_selected
                if return_predictions:
                    selected_predictions = return_selected_predictions( pf_patches_selected, detections, pf_patches_resized)

            if verbose:
                duration = time.time() - start_time
                print(f'PVF ResNet50 took {round(duration, 2)} seconds')
                decision = 'pv' if avg_pv > avg_pf else 'pf' if avg_pf > avg_pv else '??'
                print(f'PVF ResNet50 PF {len(pf_patches_selected)} ({round(float(avg_pf), 2)}) vs PV {len(pv_patches_selected)} ({round(float(avg_pv), 2)}) | decision {decision}')

        elif len(pf_patches_selected) > 1 and len(pv_patches_selected) <= 1:
            img_result = 'pf'
            total_pf.append(len(pf_patches_selected))
            total_pv.append(0)
            total_u.append(0)
            pf_all_selected_patches = pf_all_selected_patches + pf_patches_selected
            if return_predictions:
                selected_predictions = return_selected_predictions(pf_patches_selected, detections, pf_patches_resized)

        elif len(pv_patches_selected) > 1 and len(pf_patches_selected) <= 1:
            img_result = 'pv'
            total_pv.append(len(pv_patches_selected))
            total_pf.append(0)
            total_u.append(0)
            pv_all_selected_patches = pv_all_selected_patches + pv_patches_selected
            if return_predictions:
                selected_predictions = return_selected_predictions(pv_patches_selected, detections, pv_patches_resized)

        elif len(pv_patches_selected) <= 1 and len(pf_patches_selected) <= 1:
            img_result = 'u'
            total_u.append(1)
            total_pv.append(0)
            total_pf.append(0)
            if return_predictions:
                selected_predictions = []
        
        else:
            raise Exception('Something went wrong and no totals were added!')

        new_row =   {'patient_n': patient_n, 'img': -1, 'model_score_thr': model_score_thr, \
                    'PV_detected': len(pv_patches_resized), 'PV_selected': len(pv_patches_selected), 'PV_prob': avg_pv,\
                    'PF_detected': len(pf_patches_resized), 'PF_selected': len(pf_patches_selected), 'PF_prob': avg_pf, \
                    'dataset': dataset, 'result' : img_result}
        df_logs.loc[len(df_logs)] = new_row
    if verbose:        
        duration = time.time() - total_start_time
        print(f' Total time for {len(img_np)} images {round(duration, 2)} seconds')
        print('total_u', total_u)
        print('total_pf', total_pf)
        print('total_pv', total_pv)
        print('threshold', model_score_thr)
    
    df_logs.to_csv(log_filename, index= False)

    if np.mean(total_u) > 0.5:
        result = 'u'
        selected_patches = random.shuffle(pf_all_selected_patches + pv_all_selected_patches)
    elif np.sum(total_pf ) > np.sum(total_pv):
        result = 'pf'
        selected_patches = pf_all_selected_patches
    elif np.sum(total_pv ) > np.sum(total_pf):
        result = 'pv'
        selected_patches = pv_all_selected_patches
    else:
        raise Exception('could not classify')
    
    if verbose:
        print('patient was classified as:', result)

    if return_predictions:
        return result, selected_patches, selected_predictions
    else:
        return result, selected_patches
