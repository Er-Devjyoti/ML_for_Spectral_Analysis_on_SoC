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
  Machine Learning for Spectral Analysis on FPGA
</h1>

---

Signal Recognition and Classification using Object Detection Algorithms and deployment on a SoC

For the Spectrum Analysis, please clone this Repo and Run the files accordingly in the following order:

1. Dataset Creation
2. CSV generator
3. model.ipynb

Note: the Dataset generation uses Torchsig Toolkit Please refer to the Torchsig WB: https://github.com/torchdsp/torchsig installation process to proceed further.

## Getting Started


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
