import numpy as np
import streamlit as st

import cv2
import tifffile
from PIL import Image
from os import listdir
from os.path import isfile, join, splitext
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


from tensorflow.keras.models import load_model
from tensorflow.image import resize_with_pad
from tensorflow import expand_dims, function, float32, convert_to_tensor
from tensorflow.nn import softmax
from tensorflow.compat.v2.train import Checkpoint

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import preprocess
#import prediction

# from streamlit group
from load_css import local_css
local_css("style.css")

from skimage.util import img_as_ubyte

def imread(image_up):
    ext = splitext(image_up.name)[-1]
    if ext== '.tif' or ext=='tiff':
        img = tifffile.imread(image_up)
        return img
    else:
        # img = plt.imread(image_up)
        # img = img_as_ubyte(img)

        file_bytes = np.asarray(bytearray(image_up.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img = img_bgr[...,::-1]
        # # img_bgr = cv2.imread(opencv_image, flags=cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_COLOR) 

        # img = Image.open(image_up)
        # img = np.array(img)
    return img

@st.cache(show_spinner=False)
def run_segmentation(model, image, diam, channels, flow_threshold, cellprob_threshold):
    masks, flows, styles, diams = model.eval(image, 
            # batch_size = 8,
            diameter = diam, # 100
            channels = channels,
            invert = True,
            # rescale = 0.5,
            net_avg=False,
            flow_threshold = flow_threshold, # 1
            cellprob_threshold = cellprob_threshold, # -4
                            )
    return masks, flows, styles, diams


@st.cache(show_spinner=False)
def transform_image(arr):
    my_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                        # transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
#     image = Image.open(io.BytesIO(image_bytes))
    im = Image.fromarray(arr)
    return my_transforms(im).unsqueeze(0)

class_names = ['parasitized', 'uninfected']

def get_prediction(arr):
    tensor = transform_image(arr)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return class_names[y_hat]

def get_prediction_tf(arr):
    '''function for tf model; resizes and pads an image array and then returns the prediction i.e. the most likely class'''
    arr = resize_with_pad(arr, 100, 100, method= 'bilinear', antialias=False)
    arr = expand_dims(arr, 0) # Create a batch
    predictions = custom_model.predict(arr)
    score = softmax(predictions[0])
    #print(class_names[np.argmax(score)])
    return class_names[np.argmax(score)]
      

st.title('Malaria Detection')
# st.text('Segmentataion -> Single cell ROI -> Classification')

page = st.sidebar.selectbox("Choose slide type", ('Thick Smear', 'Thick Smear | Sample images'))

st.sidebar.title("About")
st.sidebar.info("- Trained on Giemsa stained _plasmodia falsiparum & vivax_  [NIH Data] (https://data.lhncbc.nlm.nih.gov/public/Malaria/index.html)   \n \
- Powered by TensorFlow and [Streamlit] (https://docs.streamlit.io/en/stable/api.html) ")


file_up = None  

        
if page == 'Thick Smear | Sample images':
    img_list = [join('images_thick', f) for f in listdir('images_thick') if isfile(join('images_thick', f))]
    img_captions = [f for f in listdir('images_thick') if isfile(join('images_thick', f))]
    img_captions.append(" 👉 Choose an image here 👈")
    selected_image = st.selectbox("Choose a sample image to analyze", img_captions, (len(img_captions)-1))
    if selected_image != " 👉 Choose an image here 👈":
        selected_image = img_captions.index(selected_image)
        file_up = img_list[selected_image]
        img = plt.imread(file_up)
        image = img_as_ubyte(img)
    if not file_up:
        st.image(img_list, caption = img_captions[:-1], width = int(698/2))
       
else:
    # if not file_up:
    file_up = st.file_uploader("Upload an image", type=["tif", "tiff", "png", "jpg", "jpeg"],  accept_multiple_files= True)
    # if file_up:
    #     st.text(file_up)
        # image = imread(file_up)

if file_up:
    st_preprocess_bar = st.progress(0)
    imgs_cropped = []
    imgs_small_cropped = []
    calc_radiuses = []
    files = []
    for n, file in enumerate(tqdm(file_up)):
        img = imread(file)
        height, width, c = img.shape
        reduction_prct = 600/ min([height, width]) 
        img_small = cv2.resize(img, None, fx=reduction_prct, fy=reduction_prct)
        img_small_cropped, xmin, xmax, ymin, ymax  = preprocess.circle_crop(img_small)
        img_cropped =     img[int(xmin * width):int(xmax * width),
                                int(ymin * height) :int(ymax * height)]

        imgs_small_cropped.append(img_small_cropped)
        imgs_cropped.append(img_cropped)
        files.append(file.name)
        st_preprocess_bar.progress((n +1) /len(file_up))
        try: 
            calc_radius = preprocess.calculate_WBC_radius(img_small_cropped, prct_reduced= reduction_prct)
            calc_radiuses.append(calc_radius)
        except Exception as e:
            print(e)
            pass


    if calc_radiuses:
        radius = st.sidebar.slider('How large is a white blood cell ?', 10.0, 150.0, float(np.mean(calc_radiuses)/reduction_prct))
    else:
        st.warning('Could not detect WBC radius, please ensure WBC radius is around 28 pixel')


    fig = plt.figure(figsize = (8,10), facecolor= '#0e1117')
    
    # Compute Rows required
    total_subplots = len(files)
    if len(files) >= 9:
        columns = 3
    else:
        columns = 2
    rows = total_subplots // columns 
    rows += total_subplots % columns
    
    for i in tqdm(range(1, columns*rows +1)):
        if i > len(files):
            break
        img_cropped =  imgs_small_cropped[i-1]
        ax = fig.add_subplot(rows, columns, i)
        ax.imshow(img_cropped)
        if calc_radiuses:
            circ = Circle((radius/5*2,radius/5*2), radius/5, color = '#f63366')
            ax.add_patch(circ)
        ax.axis("off")
        ax.set_title(f'{files[i-1]}', color='white', fontsize = 8, pad = 8)
    st.pyplot(fig)

    # with st.spinner("Detecting potential parasites"):
    #     dict_ckpt_cfg = {'models/fasterrcnn_inception_v2_1024_PF_train150000/ckpt-151':
    #                     'models/fasterrcnn_inception_v2_1024_PF_train150000/thick_PF_wh64.config'}

    #     path_label = 'models/class_labels_malaria_1class.pbtxt'

        # for path_ckpt,path_cfg in dict_ckpt_cfg.items():
        #     start_time = time.time()
        #     detections = prediction.faster_rcnn(image, path_cfg, path_ckpt)
        #     end_time = time.time()
        #     elapsed_time = end_time - start_time
        #     st.text(f'Took {elapsed_time} seconds') 

        #     threshold = 0.05

        #     label_id_offset = 1
        #     image_np_with_detections = image.copy()

        #     category_index = label_map_util.create_category_index_from_labelmap(path_label,
        #                                                                 use_display_name=True)

        #     viz_utils.visualize_boxes_and_labels_on_image_array(
        #             image_np_with_detections,
        #             detections['detection_boxes'],
        #             detections['detection_classes']+label_id_offset,
        #             detections['detection_scores'],
        #             category_index,
        #             use_normalized_coordinates=True,
        #             max_boxes_to_draw=400,
        #             min_score_thresh= threshold,
        #             agnostic_mode=False)


        #     data = Image.fromarray(image_np_with_detections)

        #     fig, ax = plt.subplots(figsize = (8,8))
        #     ax.imshow(image_np_with_detections)
        #     ax.axis("off")
        #     ax.set_title('Faster RCNN predictions')
        #     st.pyplot(fig)



    # start_time = time.time()
    # end_time = time.time()
    # elapsed_time = end_time - start_time


        
    #     with st.spinner("Loading Model"):
    #         PATH = 'ensemblemodel_pairD.h5'
    #         custom_model=load_model(PATH)
    #         custom_model.summary()
    #         # device = torch.device('gpu')
    #         # # Load cnn model
    #         # PATH = "model.pth"
    #         # model = torch.load(PATH, map_location = device)
    #         # model.eval()
        
    #     # size_thres = diameter*0.5
    #     tmp_img = image.copy()
    #     d_results = {"parasitized": [],
    #                 "uninfected": [],
    #                 }
        
    #     with st.spinner("Searching for parasites"):
    #         since = time.time()
    #         for idx, cell in enumerate(outlines_ls[:]):
                
    #             x = cell.flatten()[::2]
    #             y = cell.flatten()[1::2]

    #             # mask outline
    #             mask = np.zeros(tmp_img.shape, dtype=np.uint8)
    #             channel_count = tmp_img.shape[2]  # i.e. 3 or 4 depending on your image
    #             ignore_mask_color = (255,)*channel_count
    #             # fill contour
    #             cv2.fillConvexPoly(mask, cell, ignore_mask_color)
    #             # extract roi
    #             masked_image = cv2.bitwise_and(tmp_img, mask)
    #             # crop the box around the cell
    #             (topy, topx) = (np.min(y), np.min(x))
    #             (bottomy, bottomx) = (np.max(y), np.max(x))
    #             out = masked_image[topy:bottomy+1, topx:bottomx+1,:]
    #             # predict the stage of the cell p
    #             stage = get_prediction_tf(out)
    #             d_results[stage].append(idx)

    #     time_elapsed = time.time() - since
    #     st.write('time spent on classification {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))


    #     with st.spinner("Plotting results"):
    #         t = "<div> <span class='highlight darkgreen'> Uninfected </span> \
    #         <span class='highlight red'> Parasitized </span>"
    #         st.markdown(t, unsafe_allow_html=True)

    #         colors_stage = { "parasitized": "#FF0000", "uninfected": '#458B00'}
    #         fig, ax = plt.subplots(figsize = (8,8))
    #         # yellow: ring; magenta: troph; cyan: shiz
    #         ax.imshow(image)

    #         for k in class_names:
    #             # if k!= 'uninfected' and len(d_results[k]) > 0:
    #             if len(d_results[k]) > 0:
    #                 for cell in d_results[k]:
    #                     coord = outlines_ls[cell]
    #                     ax.plot(coord[:,0], coord[:,1], c = colors_stage[k], lw=1)
    #         ax.set_title('Predicted infected cells')
    #         ax.axis('off')
    #         st.pyplot(fig)

    #         total_count = sum(len(v)for v in d_results.values())
    #         st.write("Final cell count", total_count)
    #         out_stat = []
    #         for key in class_names:
    #             stage_count = len(d_results[key])
    #             # st.write(key, stage_count, round(stage_count/total_count, 3))
    #             paras = round(stage_count/total_count, 3)
    #             out_stat.append((stage_count, paras))
    #         par = (out_stat[0][1])*100
    #         st.write("Parasitemia (%)", round(par, 2) )
    #         st.markdown(f"""
    #             | Stage       |      Count         |       %             |
    #             | ------------| -------------      | ----------          |
    #             | Parasitized | {out_stat[0][0]}   |  {out_stat[0][1]}   | 
    #             | Uninfected  | {out_stat[1][0]}   |  {out_stat[1][1]}   |

    #         """)
                
    # if page == 'Thick Smear' or page == 'Thick Smear | Sample images':
        
    #     with st.spinner("Loading thick smear model"):
    #         # Load pipeline config and build a detection model
    #         configs = config_util.get_configs_from_pipeline_file('pipeline.config')
    #         model_config = configs['model']
    #         detection_model = model_builder.build(model_config=model_config, is_training=False)

    #         # Restore checkpoint
    #         ckpt = Checkpoint(model=detection_model)
    #         ckpt.restore('ckpt-24').expect_partial()

    #     @function
    #     def detect_fn(image):
    #         """Detect objects in image."""

    #         image, shapes = detection_model.preprocess(image)
    #         prediction_dict = detection_model.predict(image, shapes)
    #         detections = detection_model.postprocess(prediction_dict, shapes)

    #         return detections

    #     with st.spinner("Searching for parasites"):
    #         category_index = label_map_util.create_category_index_from_labelmap('class_labels_malaria.pbtxt',
    #                                                                     use_display_name=True)

    #         input_tensor = convert_to_tensor(np.expand_dims(image, 0), dtype=float32)

    #         detections = detect_fn(input_tensor)

    #         # All outputs are batches tensors.
    #         # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    #         # We're only interested in the first num_detections.
    #         num_detections = int(detections.pop('num_detections'))
    #         detections = {key: value[0, :num_detections].numpy()
    #                     for key, value in detections.items()}

    #         detections['num_detections'] = num_detections

    #         # detection_classes should be ints.
    #         detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    #         label_id_offset = 1
    #         img_with_detections = image.copy()
        
    #     with st.spinner("Visualising results..."):
    #         threshold = 0.2
    #         viz_utils.visualize_boxes_and_labels_on_image_array(
    #                 img_with_detections,
    #                 detections['detection_boxes'],
    #                 detections['detection_classes']+label_id_offset,
    #                 detections['detection_scores'],
    #                 category_index,
    #                 use_normalized_coordinates=True,
    #                 max_boxes_to_draw=200,
    #                 min_score_thresh= threshold,
    #                 agnostic_mode=False)

    #         parasites = sum(detections['detection_scores'][detections['detection_classes'] == 0] > threshold)
    #         WBC = sum(detections['detection_scores'][detections['detection_classes'] == 1] > threshold)
        
    #     fig, ax = plt.subplots(figsize = (8,8))
    #     ax.imshow(img_with_detections)

    #     ax.set_title('Predicted infected cells')
    #     ax.axis('off')
    #     st.pyplot(fig)
        
    #     st.write("Parasites per WBC ", round(parasites/ WBC, 2) )
    #     st.markdown(f"""
    #         | Stage              |      Count         |       %             |
    #         | -------------------| -------------------| --------------------|
    #         | Parasities         | {parasites}        |  {round(parasites / (parasites + WBC), 2)}   | 
    #         | White blood cells  | {WBC}              |  {round(WBC / (parasites + WBC), 2)}   |

    #     """)