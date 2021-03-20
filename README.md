# malaria-detection-classification  

This project is aimed at developing a web tool that will allow researchers to automatically detect cells with malaria parasite and quantify its density. Importantly, it is able to identify the developmental stage of the parasite (P. _falsiparum_), which has not been done before and only became possible due to the unique dataset that was provided to us. 

## Demo

![Malaria app Demo](demo3.gif)  

**Pipeline**    

- Cell segmentation from microscope images using [cellpose](https://github.com/MouseLand/cellpose)  
- ROI extarction  
- ROI classification using finetuned Resnet18 and/or SqueezeNet  

The app was deployed to [Heroku](https://sleepy-escarpment-93127.herokuapp.com/). Note: it is much slower due to the limitations of the free-tier server.  
Powered by PyTorch and [Streamlit](https://docs.streamlit.io/en/stable/api.html)  
  
