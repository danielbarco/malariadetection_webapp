FROM nvcr.io/nvidia/tensorrt:23.10-py3
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

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.9 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN git clone --depth 1 https://github.com/tensorflow/models

# Compile protobuf configs
RUN (cd models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR models/research/

RUN cp object_detection/packages/tf2/setup.py ./

RUN pip install "cython<3.0.0" pyyaml==5.4.1 --no-build-isolation -v
RUN python -m pip install -U pip
RUN python -m pip install .

WORKDIR /

RUN pip install scikit-image
# RUN pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip install torchvision==0.9.1
RUN pip install streamlit
RUN pip install matplotlib
RUN pip install tensorflow
# RUN pip install cellpose==0.6.1
RUN pip install protobuf
# cellpose requires large models, load these from cache with:
# RUN python -c 'from cellpose import models'

# COPY models models
# COPY style.css .
# COPY images_thick images_thick
# # COPY images_thin images_thin
# COPY load_css.py .
# COPY preprocess.py .
# COPY prediction.py .
# COPY app.py .
WORKDIR docker/
#ENTRYPOINT ["streamlit","run"]
#CMD ["app.py"]
