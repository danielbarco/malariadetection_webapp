FROM python:3.8-slim-buster

EXPOSE 8501

WORKDIR /app
# COPY requirements.txt requirements.txt

# RUN pip install streamlit
# RUN pip3 install -r requirements.txt
# RUN /usr/local/bin/python -m pip install --upgrade pip

RUN pip install scikit-image
RUN pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install streamlit==0.81
RUN pip install matplotlib
RUN pip install tensorflow==2.4.1
RUN pip install cellpose==0.6.1
RUN pip install object_detection
# cellpose requires large models, load these from cache with:
RUN python -c 'from cellpose import models'


COPY . .
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]