import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder
import itertools
import cv2

def object_detection(img_np, path_cfg, path_ckpt): 
    '''Predicts object detection on image
    Args:
        img_np (np.array): image 1024 x 1024 with white blood cell diameter of 
        path_cfg (str): path to tf object detection model config file
        path_ckpt (str): path to tf checkpoint'''
        
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(path_cfg)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(path_ckpt).expect_partial()

    @tf.function
    def detect_fn(img):
        """Detect objects in img."""

        img, shapes = detection_model.preprocess(img)
        prediction_dict = detection_model.predict(img, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)

        return detections


    input_tensor = tf.convert_to_tensor(np.expand_dims(img_np, 0), dtype=tf.float32)

    detections_raw = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections_raw.pop('num_detections'))


    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections_raw.items()}

    detections['num_detections'] = num_detections

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

def cut_patches(detections, img_full, model_score_thr = 0.0, input_img_size = 1024, out_img_size = 256):
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
    ''' image[start_row:end_row, start_column:end_column] e.g. image[30:250, 100:230] or [x1:x2, y1:y2]
    You can see that the waterfall goes vertically starting at about 30px and ending at around 250px.
    You can see that the waterfall goes horizontally from around 100px to around 230px. 
                '''
    for bbx in tf_detections:

        patch = img_full[int(bbx[3] * width_scale):int(bbx[1] * width_scale),
                    int(bbx[2] * height_scale):int(bbx[0] * height_scale)]


        patch_resized = cv2.resize(patch, (out_img_size, out_img_size), interpolation = cv2.INTER_CUBIC)
        patches_resized.append(patch_resized)


    return patches_resized