# malaria-detection-classification  

This project is aimed at developing a web tool that will allow researchers to automatically detect cells with malaria parasite and quantify its density. Importantly, it is able to identify the developmental stage of the parasite (P. _falsiparum_), which has not been done before and only became possible due to the unique dataset that was provided to us. 

## Demo

![Malaria app Demo](demo3.gif)  

**Pipeline**    

**1. Save them in:**  
  - Cell segmentation from microscope images using cellpose  
  - ROI extarction  
  - ROI classification using finetuned Resnet18  
  
