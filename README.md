# ATP Prototype for LED-Matrix-based Optical Camera Communication

This repository provides the source code for an Acquisition, Tracking, and Pointing (ATP) prototype for mobile Optical Camera Communication (OCC). 

The system employs a two-axis servo mechanism acting as a gimbal, controlled by PID controllers. A YOLOv8 model is used to detect the LED-matrix transmitter, and the obtained bounding box coordinates guide the servo movements to track the transmitter. The real-time response and control signal graphs will be displayed while the code is running. 

This source code is written in Python and leverages the OpenCV and ONNX Runtime libraries.

## Environment
This code is developed and run on:
* LattePanda Sigma
* ADLINK Pocket AI GPU (NVIDIA RTX A500 GPU)
* Python 3.10

## Requirements


## Demonstration
|:[![Implementation Demo](http://img.youtube.com/vi/Dx8EFrHQ14I/0.jpg)](https://youtu.be/Dx8EFrHQ14I "Acquisition, Tracking, and Pointing (ATP) Prototype for Optical Camera Communication - CLICK TO WATCH"):|

## Acknowledgement
Thanks to [krishnayoga](https://github.com/krishnayoga) and [rangganast](https://github.com/rangganast) for contributing to this source code.