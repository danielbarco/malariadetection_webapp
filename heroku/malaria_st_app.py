
import streamlit as st

from cellpose import models, utils
import cv2
import tifffile
import time, os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models as torchmodels
from load_css import local_css
local_css("style.css")

# @st.cache
def imread(image_up):
    ext = os.path.splitext(image_up.name)[-1]
    if ext== '.tif' or ext=='tiff':
        img = tifffile.imread(image_up)
        return img
    else:
        img = plt.imread(image_up)
    return img
# @st.cache
def run_segmentation(model, image, diam, channels, flow_threshold, cellprob_threshold):
    masks, flows, styles, diams = model.eval(image, 
            batch_size = 1,
            diameter = diam, # 100
            channels = channels,
            invert = True,
            # rescale = 0.5,
            net_avg=False,
            flow_threshold = flow_threshold, # 1
            cellprob_threshold = cellprob_threshold, # -4
                            )
    return masks, flows, styles, diams
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

# @st.cache
def get_cell_outlines(masks):
    outlines_ls = utils.outlines_list(masks)
    return outlines_ls

# @st.cache
def transform_image(arr):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
#     image = Image.open(io.BytesIO(image_bytes))
    im = Image.fromarray(arr)
    return my_transforms(im).unsqueeze(0)

class_names = ["un", 'ring', 'troph', 'shiz']

def get_prediction(arr):
    tensor = transform_image(arr)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return class_names[y_hat]




st.title('P. falciparum malaria stage classification')
st.text('Segmentataion -> Single cell crops -> Classification')


file_up = st.file_uploader("Upload an image", type=["tif", "tiff", "png", "jpg", "jpeg"])

if file_up:
 
    # image = tifffile.imread(file_up)
    image = imread(file_up)
    # image = cv2.resize(img, (0,0), fx=0.4, fy=0.4) 
    
    # image = io.imread(file_up)
    # st.image(image, caption='Uploaded Image.')
    fig, ax = plt.subplots(figsize = (4,4))
    ax.imshow(image)
    ax.axis("off")
    st.pyplot(fig)
    
    st.subheader('Segmentation parameters')

    diameter = st.number_input('diameter of the cells [pix]', 0, 500, 100, 10)

    st.write('The current number is ', diameter)


    color_mask = "#000000"
    flow_threshold = 1
    cellprob_threshold = -4
    # # default = 0.4
    # flow_threshold = st.slider('Flow threshold (increase -> more cells)', 0.1, 1.1, 1.0, 0.1)
    # st.write("", flow_threshold)

    # # default = 0
    # cellprob_threshold = st.slider('Cell probability threshold (decrease -> more cells)', -6, 6, -4, 1)
    # st.write("", cellprob_threshold)

    # color_mask = st.color_picker('Pick a color for cell outlines', '#000000')
    # st.write('The current color is', color_mask)

    if st.button('Analyze'):
        # DEFINE CELLPOSE MODEL
        # model_type='cyto' or model_type='nuclei'
        with st.spinner("Running model"):

            model = models.Cellpose(gpu=False, model_type ='cyto')
            diameter = 100
            # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
            channels = [[0,0]] #* len(files) # IF YOU HAVE GRAYSCALE
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            since = time.time()
        
            masks, flows, styles, diams = run_segmentation(model, gray, diameter, channels, 
                                                flow_threshold, cellprob_threshold)
            st.text('Initial cell count: {} '.format(masks.max()))
            

            time_elapsed = time.time() - since
            st.write('time spent on segmentation {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        # if st.button('Show results'):
                # DISPLAY RESULTS
            fig = show_cell_outlines(image, masks, color_mask)
            st.pyplot(fig)
            outlines_ls = get_cell_outlines(masks)


    # if st.button('Run classification'):
        with st.spinner("Loading Model"):
            num_classes = 4
            device = torch.device('cpu')
            # Load cnn model
            PATH = "squeezenet_model.pth"
            model = torchmodels.squeezenet1_0(pretrained=True)
            model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model.num_classes = num_classes
            model.load_state_dict(torch.load(PATH, map_location = device))
            model.to(device)
            model.eval()
        
        size_thres = diameter*0.5
        tmp_img = image.copy()
        d_results = {"un": [],
                    "ring": [],
                    "troph": [],
                    "shiz": []
                    }
        with st.spinner("Running inference..."):
        # st.text("Running inference ...")
            since = time.time()
            for idx, cell in enumerate(outlines_ls[:]):
                
                x = cell.flatten()[::2]
                y = cell.flatten()[1::2]

                if (y.max() - y.min()) < size_thres or (x.max() - x.min()) < size_thres:
                    continue

                # mask outline
                mask = np.zeros(tmp_img.shape, dtype=np.uint8)
                channel_count = tmp_img.shape[2]  # i.e. 3 or 4 depending on your image
                ignore_mask_color = (255,)*channel_count
                # fill contour
                cv2.fillConvexPoly(mask, cell, ignore_mask_color)

                masked_image = cv2.bitwise_and(tmp_img, mask)

                # crop the box around the cell
                (topy, topx) = (np.min(y), np.min(x))
                (bottomy, bottomx) = (np.max(y), np.max(x))
                out = masked_image[topy:bottomy+1, topx:bottomx+1,:]
            #     plt.imshow(out)
                stage = get_prediction(out)
                d_results[stage].append(idx)
            #     plt.show()
        time_elapsed = time.time() - since
        st.write('time spent on classification {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


        with st.spinner("Plotting results"):
            t = "<div> <span class='highlight yellow'> Ring </span> \
                    <span class='highlight magenta'> Troph </span>      \
                    <span class='highlight cyan'> Shiz </span>      </div>"
            st.markdown(t, unsafe_allow_html=True)

            colors_stage = { "un": [1, 0, 0], "ring": [1, 1, 0], 
                "troph": [1, 0, 1], "shiz": [0, 1, 1] }
            fig, ax = plt.subplots(figsize = (8,8))
            # yellow: ring; magenta: troph; cyan: shiz
            ax.imshow(image)

            for k in class_names:
                if k!= "un" and len(d_results[k]) > 0:
                    for cell in d_results[k]:
                        coord = outlines_ls[cell]
                        ax.plot(coord[:,0], coord[:,1], color = colors_stage[k], lw=1)
            ax.set_title('Predicted infected cells')
            ax.axis('off')
            st.pyplot(fig)

            total_count = sum(len(v)for v in d_results.values())
            st.write("Final cell count", total_count)
            out_stat = []
            for key in class_names:
                stage_count = len(d_results[key])
                # st.write(key, stage_count, round(stage_count/total_count, 3))
                paras = round(stage_count/total_count, 3)
                out_stat.append((stage_count, paras))
            st.markdown(f"""
                | Stage      |      Count                 |       %    |
                | -----------| -------------              | ---------- |
                | Uninfected |   {out_stat[0][0]}   |  {out_stat[0][1]}    | 
                | Ring       |    {out_stat[1][0]}   |   {out_stat[1][1]}  |
                | Troph      | {out_stat[2][0]}    |    {out_stat[2][1]} |
                | Shiz       |   {out_stat[3][0]}   |   {out_stat[3][1]}   |
    """)