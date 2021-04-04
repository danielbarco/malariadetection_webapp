
import numpy as np
import streamlit as st
from cellpose import models, io, utils
from scipy.ndimage import find_objects
import time, os
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import cv2
import tifffile
from PIL import Image

# from streamlit group
from load_css import local_css
local_css("style.css")

def imread(image_up):
    ext = os.path.splitext(image_up.name)[-1]
    if ext== '.tif' or ext=='tiff':
        img = tifffile.imread(image_up)
        return img
    else:
        img = plt.imread(image_up)
    return img

@st.cache(show_spinner=False)
def run_segmentation(model, image, diam, channels, flow_threshold, cellprob_threshold):
    masks, flows, styles, diams = model.eval(image, 
            # batch_size = 8,
            diameter = diam, # 100
            channels = channels,
            invert = True,
            # rescale = 0.5,
            net_avg = False,
            flow_threshold = flow_threshold, # 1
            cellprob_threshold = cellprob_threshold, # -4
                            )
    return masks, flows, styles, diams

#from cellpose
# @st.cache
def show_cell_outlines(img, maski, color_mask):

    outlines = utils.masks_to_outlines(maski)
    
    # plot the WordCloud image     
    fig, ax = plt.subplots(figsize = (8, 8))                   
    outX, outY = np.nonzero(outlines)
    imgout= img.copy()
    h = color_mask.lstrip('#')
    hex2rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    imgout[outX, outY] = hex2rgb
    # imgout[outX, outY] = np.array([255,75,75])
    ax.imshow(imgout)
    #for o in outpix:
    #    ax.plot(o[:,0], o[:,1], color=[1,0,0], lw=1)
    ax.set_title('Predicted outlines')
    ax.axis('off')
    
    return fig

@st.cache
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

class_names = ["un", 'ring', 'troph', 'shiz']


st.title('P. falciparum Malaria Detection and Classification')
st.text('Segmentataion -> Single cell ROI -> Classification')

page = st.sidebar.selectbox("Choose a stain", ('Giemsa', 'Stain type 2', 'Sample images'))

st.sidebar.title("About")
st.sidebar.info(" - Segmentation: [Cellpose] (https://github.com/MouseLand/cellpose)    \n \
- Classification of ROI: pretrained Resnet18 + fine-tuning      \n \
- Trained on Giemsa stained P. _falsiparum_     \n \
Powered by PyTorch, [Streamlit] (https://docs.streamlit.io/en/stable/api.html) ")


file_up = None

if page == 'Sample images':
    img_list = ["./images/T0D2_1.tif", "./images/T14D2_2.tif", "./images/T38D2_2.tif", "./images/T48D2_2.tif" ]
    img_captions = ["After 2 hours", "After 14 hours", "After 38 hours", "After 48 hours", "<choose>"]
    st.image(img_list, caption = img_captions[:4], width = int(698/2))
    selected_image = st.selectbox("Choose a sample image to analyze", img_captions, 4)
    if selected_image != "<choose>":
        selected_image = img_captions.index(selected_image)
        file_up = img_list[selected_image]
        # st.text(file_up)
        image = tifffile.imread(file_up)

else:
    file_up = st.file_uploader("Upload an image", type=["tif", "tiff", "png", "jpg", "jpeg"])
    if file_up:
        image = imread(file_up)

if file_up:
    # @st.cache
    # image = Image.open(file_up)
    
    fig, ax = plt.subplots(figsize = (8,8))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title('Selected image')
    st.pyplot(fig)

    st.subheader('Segmentation parameters')

    diameter = st.number_input('Diameter of the cells [pix]', 0, 500, 100, 10)
    st.write('The current number is ', diameter)

    flow_threshold = 1
    # flow_threshold = st.slider('Flow threshold (increase -> more cells)', .0, 1.1, 1.0, 0.1)
    # st.write("", flow_threshold)

    cellprob_threshold = -4
    # cellprob_threshold = st.slider('Cell probability threshold (decrease -> more cells)', -6, 6, -4, 1)
    # st.write("", cellprob_threshold)

    color_mask = '#000000'
    # color_mask = st.color_picker('Pick a color for cell outlines', '#000000')
    # st.write('The current color is', color_mask)

    if st.button('Analyze'):
        # DEFINE CELLPOSE MODEL
        # model_type='cyto' or model_type='nuclei'
        with st.spinner("Running segmentation"):
            model = models.Cellpose(gpu=False, model_type ='cyto')
            # diameter = 100
            # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
            channels = [[0,0]] #* len(files) # IF YOU HAVE GRAYSCALE

            since = time.time()
            # img = io.imread(filename)
            masks, flows, styles, diams = run_segmentation(model, image, diameter, channels, 
                                                flow_threshold, cellprob_threshold)
            st.text('Initial cell count: {} '.format(masks.max()))
            
            time_elapsed = time.time() - since
            st.write('Time spent on segmentation {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        # if st.button('Show results'):
                # DISPLAY RESULTS
            fig = show_cell_outlines(image, masks, color_mask)
            st.pyplot(fig)


        with st.spinner("Loading Model"):
            device = torch.device('cpu')
            # Load cnn model
            PATH = "model.pth"
            model = torch.load(PATH, map_location = device)
            model.eval()
        
        size_threshold = diameter*0.5 # in pix
        tmp_img = image.copy()
        d_results = {"un": [],
                    "ring": [],
                    "troph": [],
                    "shiz": []
                    }
        with st.spinner("Running inference..."):
        # st.text("Running inference ...")
            since = time.time()
      
            objects = find_objects(masks)
            ls_img = []
            ls_outlines = []
            # get crops with precise outlines
            for n in range(masks.max()):
                mn = np.array((masks==(n+1))*255, dtype = np.uint8)
            #     print(np.sum(mn>0))
                cell_mask = np.repeat(mn[:, :, np.newaxis], 3, axis=2)
                masked_image = cv2.bitwise_and(tmp_img, cell_mask)
                cell_pix = objects[n]
                cell_roi = masked_image[cell_pix]
                # remove small particles
                if cell_roi.shape[0] < size_threshold or cell_roi.shape[1] < size_threshold:
                    continue
                tensor_img = transform_image(cell_roi)
                ls_img.append(tensor_img)
                
                # get cell outlines using masks
                contours = cv2.findContours(mn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                out_pix_y, out_pix_x = np.concatenate(contours[-2], axis=0).squeeze().T  
                # shift to the proper posiiton on the image
                ls_outlines.append((out_pix_y, out_pix_x))   

            img_tensor = torch.cat(ls_img, dim=0)
            bs = 128
            ls_preds = []
            part = img_tensor.shape[0]//bs
            for p in range(part+1):
                outputs = model.forward(img_tensor[bs*p:bs*p + bs])
                ls_preds.append(outputs.data.numpy().argmax(1))
            preds = np.concatenate(ls_preds)
            results = [class_names[x] for x in preds]
        time_elapsed = time.time() - since
        st.write('time spent on classification {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


        with st.spinner("Plotting results"):
            t = "<div> <span class='highlight yellow'> Ring </span> \
                    <span class='highlight magenta'> Troph </span>      \
                    <span class='highlight cyan'> Shiz </span>      </div>"
            st.markdown(t, unsafe_allow_html=True)

            colors_stage = { "un": [1, 0, 0], "ring": "#ffc20a", 
                "troph": "#40b0a6", "shiz": "#d35fb7" }
            fig, ax = plt.subplots(figsize = (8,8))
            # yellow: ring; magenta: troph; cyan: shiz
            ax.imshow(image)

            inf_cells = np.where(preds > 0)
            for idx in inf_cells[0]:
                ax.plot(ls_outlines[idx][0], ls_outlines[idx][1], \
                                color = colors_stage[results[idx]], lw = 1)
            ax.set_title('Predicted infected cells')
            ax.axis('off')
            st.pyplot(fig)

            arr_results = np.array(results)
            total_count = arr_results.size 
            st.write("Final cell count", total_count)
            out_stat = []
            for key in class_names:
                stage_count = arr_results[arr_results == key].size
                # st.write(key, stage_count, round(stage_count/total_count, 3))
                parasetemia = round(stage_count/total_count, 3)
                out_stat.append((stage_count, parasetemia))
            st.markdown(f"""
                | Stage      |      Count         |       %             |
                | -----------| -------------      | ----------          |
                | Uninfected | {out_stat[0][0]}   |  {out_stat[0][1]}   | 
                | Ring       | {out_stat[1][0]}   |  {out_stat[1][1]}   |
                | Troph      | {out_stat[2][0]}   |  {out_stat[2][1]}   |
                | Shiz       | {out_stat[3][0]}   |  {out_stat[3][1]}   |
    """)