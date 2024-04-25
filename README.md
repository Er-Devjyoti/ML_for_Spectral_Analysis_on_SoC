
<br>
<p align="left">
    <picture>
        <img src="https://www.strath.ac.uk/media/1newwebsite/documents/brand/strath_main.jpg" width="30%"/>
    </picture>
    <span style= "font-size: 50pt; font-weight: bold;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Project powered by:&nbsp;&nbsp;&nbsp;&nbsp;</span>             
    <picture>
        <img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="15%"/>
    </picture>
</p>

<h1 align="Center">
  Machine Learning for Spectral Analysis on SoC
</h1>

---

This project is a research initiative to present an exhilarating challenge at the intersection of advanced technology domains. Its core mission involves deploying a sophisticated Convolutional Neural Network (CNN)-based model, which has been designed not just for recognizing signal spectrums but also for the intricate task of signal detection using an object detection algorithm. By leveraging real-time antenna and state-of-the-art Analog-to-Digital Conversion (ADC) technology, our aim is to harness the formidable capabilities of machine learning to decode the complexities of the radio frequency (RF) spectrum with a remarkable degree of precision and operational efficiency. 

This project also demonstrates our groundbreaking intention to deploy this comprehensive model on the cutting-edge AMD-Xilinx RFSoC platform. This step represents a pioneering endeavour in the field, as it promises to unlock the untapped potential for substantial acceleration and real-time processing in RF signal recognition through AI. This fusion of artificial intelligence and RF technology is poised not only to advance our comprehension and utilization of the RF spectrum but also to underscore the remarkable convergence of advanced hardware and artificial intelligence in the dynamic arena of wireless communications.

Our prime research focus, however, delves even deeper into the realms of optimization. We have dedicated significant effort to Quantization analysis using Vitis AI, ensuring that our model can operate with utmost efficiency and reduced computational overhead. It's worth noting that we have successfully completed this entire project, and our methods and findings are meticulously documented, making it possible for others in the field to reproduce our results by following the comprehensive steps outlined in our research documentation. This marks a significant contribution to the broader scientific community, advancing the frontier of RF signal recognition and AI-accelerated processing using the AMD-Xilinx RFSoC hardware accelerator.


## Getting Started

---

In a pioneering move, our project takes a unique approach by harnessing RoCm (Radeon Open Compute) as the GPU accelerator, distinct from the commonly used CUDA framework. This choice not only showcases the versatility and adaptability of our AI model but also contributes to the diversification of available tools and platforms for researchers in the field. Furthermore, the deployment of this advanced model is achieved on the VCK5000 Versal Development Card, an adaptive SoC architecture. This hardware choice represents a significant leap forward in terms of its processing power and adaptability to AI-driven tasks in RF signal recognition. And Hence this project strictly requires Linux-based computational systems.

By opting for open-source GPU acceleration through RoCm and the deployment of the VCK5000, we not only promote an ecosystem of collaboration and innovation but also set a precedent for researchers to explore new horizons in AI and RF signal processing. This comprehensive approach not only advances the state-of-the-art in RF signal recognition but also empowers the wider research community by providing valuable insights, methodologies, and a robust framework for future exploration and advancements in this rapidly evolving field. The long-term impact is a more collaborative, adaptable, and innovative research ecosystem, poised to yield breakthroughs with far-reaching implications in wireless communications, AI, and beyond. 

This Repository (repo) allows to reproduce the results obtained and contains the following codebase: 

1. The YOLO-based Data generation procedure for TorchSig Toolkit.
2. Experimental analysis of the varied object detection models. 
3. YOLO Optimization Algorithm for both Pruning and Quantization (8-bit) using Vitis AI.
4. Compile the YOLOv5m model ready to deploy on an AMD-based SoC/ FPGA board (here we have used DPUCVDX8H architecture).

Note: We have only included YOLO models for the Quantization phases because the models are superior in nature and perform on par with other one-stage object detection. 

## Mandate: Clone this Repository:

---

Clone this repository:

```
git clone https://github.com/Er-Devjyoti/ML_for_Spectral_Analysis_on_SoC.git
```

## Mandate: Local System Check

---

This research project is computationally intensive and a high-end GPU to run the programs is a must. 

To run a Hello World system Check for RoCm/ CUDA, please run the SYSTEM CHECK file present in the **'YOLOv5/SYSTEMCHECK.ipynb'** in your local system. If this program runs efficiently and returns GPU availability please proceed with the next steps. 

If RoCm/ Cuda is missing or not working please check the drivers and troubleshoot the necessary dependencies.


## YOLO-based Data Generation:

---
 
### Step 1: Installation of TorchSig Toolkit

Open a new terminal window.
Clone the `torchsig` repository and simply install using the following commands:

```
git clone https://github.com/TorchDSP/torchsig.git
cd torchsig
pip install .
```

For more information on TorchSig please explore the TorchSig GitHub Page: https://github.com/TorchDSP/torchsig/tree/main

### Step 2: Data Analysis and Dataset Generation for TorchSig Pre-defined WBSig53 Dataset

1. Go to the **'Data Analysis and Preprocessing/Data Analysis.ipynb'** from our repository and run this code to analyse the Dataset.
3. To create the YOLO dataset for detection tasks (1 class - signal) for TorchSig WBSig53 data, please run the required code in **'Dataset Generation/ Detection Dataset Creation.ipynb'**. 
4. To create the YOLO dataset for classification tasks (53 signal classes) for TorchSig WBSig53 data, please run the required code in **'Dataset Generation/ Classification Dataset Creation.ipynb'**.
5. To create a random inference set only containing the images of the spectrogram please run the required code in **'Dataset Generation/ Test-set Creation.ipynb'** folder according to your need.

Note: We are using the Visualizer class that utilizes parallel processing and hence requires heavy usage of GPU to generate parallel Data. 


### Step 3: Installation of Ultralytics YOLOv5 Object Detection Model

---

Install and Colne the Ultraluytics Hub to get started with the Ultralytics Models 

```bash
pip install ultralytics
git clone https://github.com/ultralytics/yolov5 # Cloning the YOLOv5 repo specifically
cd yolov5
pip install -r requirements.txt
```

I would recommend exploring more about the Ultralytics Hub through GitHub Page: https://github.com/ultralytics/yolov5/tree/master


### Step 4: Installation of Vitis AI Docker

---

Get a local copy of Vitis AI by following the command:

```
git clone https://github.com/Xilinx/Vitis-AI
```

## Model Testing and Development:

---
Three state-of-the-art low-latency object detection architectures: DETR, YOLOv3 and YOLOv5 are trained and evaluated. However, only models with Vitis AI supporting layers are finalized.
Note: The DETR use EfficientNet as a backbone but the XCiT-nano (neck) layer is not supported by the Vitis AI and hence not included. 

### Step 1: Explore all the models used in this project like YOLOv3, YOLOv5 and Image Transformer by cloning the respective folder in this repo to the same environment of your setup.
### Step 2: Run, test and Tune each individual model one by one to get the best model performance. 
### Step 3: Pruning and Quantization can be achieved by using Visti AI (If you want to achieve the same result as mine try to run the Quantization.py in your setup, however this process is very resource-intensive and might require mGPU or HPCC). [Please Refer to the documentation]

### Step4: For the Deployment phase try to read the Vitis AI Manual Guide for Adaptive Compute Acceleration Platform (ACAP): https://docs.amd.com/r/1.4-English/ug1354-xilinx-ai-sdk/ReID-Detection


## ETH Zurich partnered HACC Setup:

---
This Project was greatly supported by ETH Zurich Partnered with AMD Xilinx as the major resources and powerhouse (HACC) was provided by them. To know more about how to setup and use this HPC please visit: https://github.com/fpgasystems/hacc/blob/main/docs/first-steps.md
