import numpy as np
import streamlit as st
from PIL import Image

import cv2
import tifffile
from os import listdir
from os.path import isfile, join, splitext
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.util import img_as_ubyte

import preprocess
import prediction
import time
import tempfile

# from streamlit group
from load_css import local_css
local_css("malariadetection_webapp/docker/style.css")


def imread(image_up):
    if page == 'Thick Smear | Sample images':
        ext = splitext(image_up)[-1]
        if ext== '.tif' or ext=='tiff':
            img = tifffile.imread(image_up)
            return img
        else:
            img_np_bgr = cv2.imread(image_up, flags=cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_COLOR)   
            return img_np_bgr[...,::-1]
    else:
        ext = splitext(image_up.name)[-1]
        if ext== '.tif' or ext=='tiff':
            img = tifffile.imread(image_up)
            return img
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(image_up.read())
            img_np_bgr = cv2.imread(tfile.name, flags=cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_COLOR)   
            return img_np_bgr[...,::-1]
    #     file_bytes = np.asarray(bytearray(image_up.read()), dtype=np.uint8)
    #     img_bgr = cv2.imdecode(file_bytes, 1)
    #   return img_bgr[...,::-1]
        # img = np.array(Image.open(image_up))
        # return img
    

@st.cache(show_spinner=False)
def load_crop_wbc_estimation(file_up):    
    imgs_padded_cropped = []
    imgs_1024_padded_cropped = []
    imgs_small_cropped = []
    calc_radiuses = []
    files = []
    for n, file in enumerate(tqdm(file_up)):
        img = imread(file)
        height, width, c = img.shape
        if min([height, width]) > 600: 
            reduction_prct = 600/ min([height, width]) 
        else:
            st.error('image too small')
            break
        img_small = cv2.resize(img, None, fx=reduction_prct, fy=reduction_prct)
        img_small_cropped, xmin, xmax, ymin, ymax  = preprocess.circle_crop(img_small)
        img_cropped =     img[int(xmin * width):int(xmax * width),
                                int(ymin * height) :int(ymax * height)]
        # pad high resolution image
        height_cropped, width_cropped, c = img_cropped.shape
        img_padded_cropped = preprocess.pad(img_cropped, height_cropped, width_cropped)
        img_padded_cropped_pil = Image.fromarray(img_padded_cropped)
        img_1024_padded_cropped= np.array(img_padded_cropped_pil.resize((1024, 1024), Image.ANTIALIAS))
        # img_1024_padded_cropped = img_padded_cropped.resize(img_padded_cropped, (1024, 1024), interpolation = cv2.INTER_AREA)
        imgs_padded_cropped.append(img_padded_cropped)
        imgs_1024_padded_cropped.append(img_1024_padded_cropped)
        imgs_small_cropped.append(img_small_cropped)
        if page == 'Thick Smear | Sample images':
            files.append(file)
        else:
            files.append(file.name)
        st_preprocess_bar.progress((n +1) /len(file_up))
        try: 
            calc_radius = preprocess.calculate_WBC_radius(img_small_cropped, prct_reduced= reduction_prct)
            calc_radiuses.append(calc_radius)
        except Exception as e:
            print(e)
            pass

    return imgs_padded_cropped, imgs_1024_padded_cropped, imgs_small_cropped, calc_radiuses, files, reduction_prct


class_names = ['parasitized', 'uninfected']


st.title('Malaria Detection')
# st.text('Segmentataion -> Single cell ROI -> Classification')

page = st.sidebar.selectbox("Choose slide type", ('Thick Smear', 'Thick Smear | Sample images'))

st.sidebar.title("About")
st.sidebar.info("- Trained on Giemsa stained _plasmodia falciparum & vivax_  [NIH Data] (https://data.lhncbc.nlm.nih.gov/public/Malaria/index.html)   \n \
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
        file_up = [file_up]
    if not file_up:
        st.image(img_list, caption = img_captions[:-1], width = int(698/2))
       
else:
    # if not file_up:
    file_up = st.file_uploader("Upload all images from one patient", type=["tif", "tiff", "png", "jpg", "jpeg"],  accept_multiple_files= True)
    # if file_up:
    #     st.text(file_up)
        # image = imread(file_up)

if file_up:
    info_element = st.empty()
    images_element = st.empty()
    st_preprocess_bar = st.progress(0)
    info_element.info("Preprocessing image(s)")
    imgs_padded_cropped, imgs_1024_padded_cropped, imgs_small_cropped, calc_radiuses, files, reduction_prct = load_crop_wbc_estimation(file_up)

    if calc_radiuses:
        radius = st.sidebar.slider('How large is a white blood cell ?', 10.0, 150.0, float(np.mean(calc_radiuses)/reduction_prct))
    else:
        st.warning('Could not detect WBC radius, please ensure WBC radius is around 28 pixel')


    fig = plt.figure(figsize = (8,8), facecolor= (0, 0, 0, 0))

    columns = 2
    rows = 2
    
    for i in tqdm(range(1, columns*rows +1)):
        if i > len(files):
            break
        img_cropped =  imgs_small_cropped[i-1]
        ax = fig.add_subplot(rows, columns, i)
        ax.imshow(img_cropped) 
        # ax.autoscale_view('tight')
        if calc_radiuses:
            circ = Circle((radius/5*2,radius/5*2), radius/5, color = '#f63366')
            ax.add_patch(circ)
        ax.axis("off")
        ax.set_title(f'{files[i-1]}', color= 'grey', fontsize = 10) 

    if len(files) > 4:
        info_element.info(f'Showing 4 of {len(files)} images')
    images_element.pyplot(fig)
    
    # if st.sidebar.checkbox("Confirm white blood cell size. Red point in top left corner of the image(s) as reference"):

    # prediction
    st_preprocess_bar.empty()
    info_element.info("Detecting potential parasites, this may take a while...")
    time_start = time.time()
    result, selected_patches = prediction.patient_eval(imgs_1024_padded_cropped, imgs_padded_cropped, model_score_thr = 0.5, verbose=True)
    prediction_time = time.time() - time_start
    info_element.empty()
    images_element.empty()
    st.warning(f'This detection model is only for research purposes. It can detect p. falciparum and p. vivax. All liability is completely disclaimed. ')
    st.success(f'Took {prediction_time/ len(imgs_1024_padded_cropped)} s/ image')
    dict_result = {'u': 'uninfected', 
    'pf': 'plasmodium falciparum', 
    'pv': 'plasmodium vivax'}
    st.subheader(f'Result: {dict_result[result]}.')

    if result in ['pv', 'pf']:
        st.info('The following images show infected cells:')
        fig_results = plt.figure(figsize = (8,8), facecolor= (0, 0, 0, 0))

        columns = 3
        rows = 3
    
        for i in tqdm(range(1, columns*rows +1)):
            if i > len(selected_patches):
                break
            selected_patch =  selected_patches[i-1]
            ax = fig_results.add_subplot(rows, columns, i)
            ax.axis("off")
            ax.imshow(selected_patch)
        st.pyplot(fig_results)
