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

# train yolov9 models on leaves dataset
```
!python "/home/ec2-user/dhruv_Spain_LLM_UI/Computer_vision_work/yolov9-main/train_dual.py" \
    --workers 8 --device 0 --batch 2 \
        --data "/home/ec2-user/dhruv_Spain_LLM_UI/Computer_vision_work/yolov9-main/data/leaves.yaml" --img 640 \
            --cfg "/home/ec2-user/dhruv_Spain_LLM_UI/Computer_vision_work/yolov9-main/models/detect/yolov9-c.yaml" \
                --weights '/home/ec2-user/dhruv_Spain_LLM_UI/Computer_vision_work/S15/yolov9-e.pt' \
                    --name yolov9-c-leaves \
                        --hyp hyp.scratch-high.yaml \
                            --min-items 0 \
                                --epochs 100 \
                                    --close-mosaic 15
```

![Training on EC2](images/training_on_ec2.png)

### Steps

1. Launch an EC2 instance with a GPU (e.g., p2.xlarge).
2. SSH into the instance and clone the YOLO V9 repository.
3. Follow the installation steps outlined above.
4. Upload your dataset to an S3 bucket.
5. Start training:
    ```sh
    python train.py --data_path s3://your_s3_bucket/dataset --epochs 50
    ```

## Metrics

After training, you can evaluate the performance of YOLO V9 using various metrics such as precision, recall, and mAP (mean Average Precision).

### Example Metrics

| Metric   | Value  |
|----------|--------|
| Precision| 0.89   |
| Recall   | 0.85   |
| mAP      | 0.87   |

## Sample Outputs and Prediction

Below are examples of YOLO V9's predictions on sample images.

### Sample Output

![Sample Output](images/sample_output.png)

### Prediction

To make predictions with YOLO V9, use the following command:

```sh
python predict.py --image_path path/to/your/image.jpg
