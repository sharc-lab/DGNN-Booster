# DGNN-Booster: A Generic FPGA Accelerator Framework For Dynamic Graph Neural Network Inference
This repository contains the code for DGNN-Booster (FCCM2023).

---

## Overview

![image](https://github.com/sharc-lab/DGNN-Booster/blob/main/optimized_FPGA_design.jpg)

## Platform

ZCU102 board. We run our design under the frequency of 100MHZ. We generate the bitstream using Vitis HLS 2021.1. We use PYNQ for design deployment. 

## Implementation details
To run the CPU and GPU baseline, directly run the python code under the baseline folder.
To run the HLS, use the code in HLS folder using two tcl files for two differnet designs. 
To run deployment, use the code in Deploy folder.

## Block Design
![image](https://github.com/sharc-lab/DGNN-Booster/blob/main/block_diagram.png)
