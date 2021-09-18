FROM tensorflow/tensorflow:latest-gpu
EXPOSE 8501

WORKDIR /app

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/devel-gpu-jupyter.Dockerfile


RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    protobuf-compiler \
    wget \
    git

RUN git clone --depth 1 https://github.com/tensorflow/models

# Compile protobuf configs
RUN (cd models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR models/research/

RUN cp object_detection/packages/tf2/setup.py ./

RUN python -m pip install -U pip
RUN python -m pip install .

WORKDIR /

RUN pip install scikit-image
# RUN pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchvision==0.9.1
RUN pip install streamlit==0.81
RUN pip install matplotlib
# RUN pip install tensorflow==2.4.1
RUN pip install cellpose==0.6.1
RUN pip install protobuf==3.13.0
# cellpose requires large models, load these from cache with:
RUN python -c 'from cellpose import models'

COPY models .
COPY load_css.py .
COPY style.css .
COPY images_thick images_thick
COPY images_thin images_thin
COPY app.py .
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]