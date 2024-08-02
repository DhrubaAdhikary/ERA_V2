
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
[Explore the demo](https://huggingface.co/spaces/DhrubaAdhikary1991/FastSAM_people_detect) 🔗

<img width="993" alt="People Detection with FastSAM" src="https://github.com/DhrubaAdhikary/ERA_V2/blob/ada1f8212711c655732f3a7a74864837d383b4aa/S23/HF_App/examples/App_Output.PNG">

As illustrated above, FastSAM effectively identifies and segments documents, disregarding the background without any additional training. This makes it an excellent tool for cropping and background removal in images.
### Predictions
1. ![Prediction Output](https://github.com/DhrubaAdhikary/ERA_V2/blob/ada1f8212711c655732f3a7a74864837d383b4aa/S23/HF_App/examples/App_Output.PNG)
2. ![Prediction Output2](https://github.com/DhrubaAdhikary/ERA_V2/blob/d81da70c742107f1e5a1fe488b9638a61d8b9a98/S23/HF_App/examples/Capture2.PNG)

### HF files
![Files](https://github.com/DhrubaAdhikary/ERA_V2/blob/d81da70c742107f1e5a1fe488b9638a61d8b9a98/S23/HF_App/examples/files_HF.PNG)

## References
1. [FastSAM GitHub Repository](https://github.com/CASIA-IVA-Lab/FastSAM)
2. File Refernce : \S23\FastSAM_example.ipynb



