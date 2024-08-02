# Car Detection using FastSAM

## Introduction

The Fast Segment Anything Model (FastSAM) is a CNN-based segmentation model, developed using only 2% of the SA-1B dataset. Despite the limited training data, FastSAM achieves performance on par with the original Segment Anything Model (SAM) while operating 50 times faster.


## Types of Prompts in FastSAM
<img width="768" alt="Types of Prompts in FastSAM" src="https://github.com/user-attachments/assets/635b85b2-e56b-4a7d-b026-a18f08ca831f">

## Model Checkpoints

FastSAM offers many model versions with different sizes. You can download the checkpoints for each version through the following links:

- **FastSAM-x**: [YOLOv8s-based Segment Anything Model](https://docs.ultralytics.com/models/fast-sam/#installation)

In the demonstration below, the default FastSAM model is used.

## People Detection with FastSAM

### Demo
[Explore the demo]() 🔗

<img width="993" alt="People Detection with FastSAM" src="">

As illustrated above, FastSAM effectively identifies and segments documents, disregarding the background without any additional training. This makes it an excellent tool for cropping and background removal in images.

## References
1. [FastSAM GitHub Repository](https://github.com/CASIA-IVA-Lab/FastSAM)
2. File Refernce : \S23\FastSAM_example.ipynb