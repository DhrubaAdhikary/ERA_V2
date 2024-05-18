# YOLO V9 Readme

## Introduction

Welcome to the YOLO V9 repository! YOLO (You Only Look Once) V9 is the latest iteration of the YOLO object detection model, known for its real-time processing and high accuracy. YOLO V9 builds on the strengths of its predecessors while incorporating advanced techniques in deep learning to improve precision, speed, and flexibility.

This repository provides everything you need to get started with YOLO V9, including setup instructions, training guidance, and sample outputs.

## Getting Started
- Clone Yolo v9 from the repo https://github.com/WongKinYiu/yolov9?tab=readme-ov-file
- Annotate your personal images for training using Roboflow . Generate Yolo v9 based coco annotations 
- Update the yaml file define the number of classes and train test and Valid file paths 
- Initiate Training . 

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.8.0 or higher
- CUDA 10.2 or higher (for GPU support)
- AWS account (for EC2 training)


## Training on EC2

Training YOLO V9 on an AWS EC2 instance is straightforward. Below is a screenshot demonstrating the setup and training process.

### YAML file 

```
train: /home/ec2-user/dhruv_UI/Computer_vision_work/S15/leaves_detection/train/images
val: /home/ec2-user/dhruv_UI/Computer_vision_work/S15/leaves_detection/valid/images
test: /home/ec2-user/dhruv_UI/Computer_vision_work/S15/leaves_detection/test/images

nc: 4
names: ['CVPD', 'Kanker', 'Melanosis', 'Sehat']

```



### train yolov9 models on leaves dataset
```
!python "/home/ec2-user/dhruv_UI/Computer_vision_work/yolov9-main/train_dual.py" \
    --workers 8 --device 0 --batch 2 \
        --data "/home/ec2-user/dhruv_UI/Computer_vision_work/yolov9-main/data/leaves.yaml" --img 640 \
            --cfg "/home/ec2-user/dhruv_UI/Computer_vision_work/yolov9-main/models/detect/yolov9-c.yaml" \
                --weights '/home/ec2-user/dhruv_UI/Computer_vision_work/S15/yolov9-e.pt' \
                    --name yolov9-c-leaves \
                        --hyp hyp.scratch-high.yaml \
                            --min-items 0 \
                                --epochs 100 \
                                    --close-mosaic 15
```

![Training on EC2]('/Info/Training screenshot.png')


## Metrics

After training, you can evaluate the performance of YOLO V9 using various metrics such as precision, recall, and mAP (mean Average Precision).

### Example Metrics

| Metric   | Value  |
|----------|--------|
| Precision| 0.97   |
| Recall   | 0.94   |

#### Confusion MXx
![Training Metrics Confusion Mxx]('/Info/confusion_matrix.png')

####
![Training Metrics F1 curve for all classes]('/Info/F1_curve.png')


## Sample Traioning and Validation set Images 

![Training Set]('/Info/train_batch0.jpg')

![Validation Set]('/Info/val_batch2_labels.jpg')

### Prediction Output

![Prediction Sample]('/Info/val_batch1_pred.jpg')


## Gradio App link 

https://huggingface.co/spaces/DhrubaAdhikary1991/YoloV9_Leaf_detection_job

[App Overview on Hugging Face]('/Info/App Overview.png')

[App prediction on Input image]('/Info/App_predition.png')


