FROM python:3.6-slim-buster

EXPOSE 8501

WORKDIR /app
# COPY requirements.txt requirements.txt

# RUN pip install streamlit
# RUN pip3 install -r requirements.txt
RUN pip install --upgrade pip
RUN pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install cellpose==0.6.1
RUN pip install streamlit==0.79.0
RUN pip install matplotlib

COPY . .
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]