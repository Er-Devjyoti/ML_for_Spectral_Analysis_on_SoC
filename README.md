<p align="left">
    <picture>
        <img src="https://www.strath.ac.uk/media/1newwebsite/documents/brand/strath_main.jpg" width="35%"/>
    </picture>
    <span style= "font-size: 50pt; font-weight: bold;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Project powered by:&nbsp;&nbsp;&nbsp;&nbsp;</span>             
    <picture>
        <img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="20%"/>
    </picture>
</p>

<h1 align="center">
  Machine Learning for Spectral Analysis on System-on-Chip (SoC)
</h1>

---

This project is a research initiative to present an exhilarating challenge at the intersection of advanced technology domains. Its core mission involves deploying a sophisticated Convolutional Neural Network (CNN)-based model, which has been designed not just for recognizing signal spectrums but also for the intricate task of signal detection using an object detection algorithm. By leveraging real-time antenna and state-of-the-art Analog-to-Digital Conversion (ADC) technology, our aim is to harness the formidable capabilities of machine learning to decode the complexities of the radio frequency (RF) spectrum with a remarkable degree of precision and operational efficiency. 

This project also demonstrates our groundbreaking intention to deploy this comprehensive model on the cutting-edge AMD-Xilinx RFSoC platform. This step represents a pioneering endeavour in the field, as it promises to unlock the untapped potential for substantial acceleration and real-time processing in RF signal recognition through AI. This fusion of artificial intelligence and RF technology is poised not only to advance our comprehension and utilization of the RF spectrum but also to underscore the remarkable convergence of advanced hardware and artificial intelligence in the dynamic arena of wireless communications.

Our prime research focus, however, delves even deeper into the realms of optimization. We have dedicated significant effort to Quantization analysis using Vitis AI, ensuring that our model can operate with utmost efficiency and reduced computational overhead. It's worth noting that we have successfully completed this entire project, and our methods and findings are meticulously documented, making it possible for others in the field to reproduce our results by following the comprehensive steps outlined in our research documentation. This marks a significant contribution to the broader scientific community, advancing the frontier of RF signal recognition and AI-accelerated processing using the AMD-Xilinx RFSoC hardware accelerator.


## Getting Started

In a pioneering move, our project takes a unique approach by harnessing RoCm (Radeon Open Compute) as the GPU accelerator, distinct from the commonly used CUDA framework. This choice not only showcases the versatility and adaptability of our AI model but also contributes to the diversification of available tools and platforms for researchers in the field. Furthermore, the deployment of this advanced model is achieved on the VCK5000 Versal Development Card, an adaptive SoC architecture. This hardware choice represents a significant leap forward in terms of its processing power and adaptability to AI-driven tasks in RF signal recognition.

By opting for open-source GPU acceleration through RoCm and the deployment of the VCK5000, we not only promote an ecosystem of collaboration and innovation but also set a precedent for researchers to explore new horizons in AI and RF signal processing. This comprehensive approach not only advances the state-of-the-art in RF signal recognition but also empowers the wider research community by providing valuable insights, methodologies, and a robust framework for future exploration and advancements in this rapidly evolving field. The long-term impact is a more collaborative, adaptable, and innovative research ecosystem, poised to yield breakthroughs with far-reaching implications in wireless communications, AI, and beyond.

---
## Step 1: Local System Check

This research project is computationally intensive and a high-end GPU to run the programs is a must. 

To run a 'Hello World' system Check for RoCm/ CUDA, please download the SYSTEM CHECK file: https://github.com/Er-Devjyoti/ML_for_Spectral_Analysis_on_SoC/blob/main/YOLOv5/SYSTEMCHECK.ipynb, from this repo and run this file in your local system.

If RoCm/ Cuda is missing or not working please check the drivers and troubleshoot the necessary dependencies.

---
## Step 2: Installation of TorchSig Toolkit

Clone the `torchsig` repository and simply install using the following commands:

```
git clone https://github.com/TorchDSP/torchsig.git
cd torchsig
pip install .
```

For more information on TorchSig please explore the TorchSig GitHub Page: https://github.com/TorchDSP/torchsig/tree/main

---
## Step 3: Installation of Ultralytics YOLOv5 Object Detection Model

Install and Colne the Ultraluytics Hub to get started with the Ultralytics Models 

```bash
pip install ultralytics
git clone https://github.com/ultralytics/yolov5 # Cloning the YOLOv5 repo specifically
cd yolov5
pip install -r requirements.txt
```

I would recommend exploring more about the Ultralytics Hub through GitHub Page: https://github.com/ultralytics/yolov5/tree/master

---
## Step 4: Installation of Vitis AI Docker

Get a local copy of Vitis AI by following the command:

```
git clone https://github.com/Xilinx/Vitis-AI
```

---
