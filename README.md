<p align="left">
    <picture>
        <img src="https://www.strath.ac.uk/media/1newwebsite/documents/brand/strath_main.jpg" width="35%"/>
    </picture>
    <span style= "font-size: 50pt; font-weight: bold;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Project powered by:&nbsp;&nbsp;&nbsp;&nbsp;</span>             
    <picture>
        <img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="20%"/>
    </picture>
</p>

---

# Machine Learning for Spectral Analysis on SoC
---

Signal Recognition and Classification using Object Detection Algorithms and deployment on a SoC

For the Spectrum Analysis, please clone this Repo and Run the files accordingly in the following order:

1. Dataset Creation
2. CSV generator
3. model.ipynb

Note: the Dataset generation uses Torchsig Toolkit Please refer to the Torchsig WB: https://github.com/torchdsp/torchsig installation process to proceed further.

## Step 1: Installation of TorchSig Toolkit
---
Clone the `torchsig` repository and simply install using the following commands:
```
git clone https://github.com/TorchDSP/torchsig.git
cd torchsig
pip install .
```
