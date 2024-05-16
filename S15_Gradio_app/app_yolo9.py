import argparse
import os
from pathlib import Path

import torch

from models.common import DetectMultiBackend
from utils.general import check_requirements, increment_path
from utils.torch_utils import select_device
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import numpy as np

import numpy as np
import cv2
import gradio as gr

import numpy as np

def scale_coords(img1_shape, coords, img0_shape):
    # Scale bounding box coordinates from img1_shape to img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords /= gain
    return coords


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a new shape while keeping the aspect ratio
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Compute ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1]/ shape[1])

    # Compute new size with padding
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # Compute padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 0.1 for numerical imprecision
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



def run_inference(img_path: gr.Image = None,
                image_size: int = 640,
                conf_threshold: float = 0.4,
                iou_threshold: float = 0.5):
    
    weights='slast.pt'
    # img_path
    output_dir='inference_output', 
    device=''
    conf_thres=conf_threshold
    iou_thres=iou_threshold
    max_det=1000
    img_size=image_size
    # Directories
    # output_dir = Path(output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    # stride = model.stride
    stride, names, pt = model.stride, model.names, model.pt
    img_size = [img_size, img_size]  # [height, width]

    # Prepare image
    img0 = cv2.imread(img_path)  # BGR
    img = letterbox(img0, new_shape=img_size)[0]

    # Inference
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # Add batch dimension

    # Inference
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=True, max_det=max_det)

    # Process detections
    #########################
    # Process predictions
    for i, det in enumerate(pred):  # per image
        # seen += 1
        gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = img.clone().detach().cpu().numpy()#img.copy() 
        annotator = Annotator(img0, line_width=3, example=str(names))
        s=''
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            view_img=True
            hide_labels=False
            hide_conf=False
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
        # Stream results
        img0 = annotator.result()
        # cv2.imwrite(output_dir+"/overlayed.png",img0)

    #########################

    print(f"Inference results saved to: {output_dir}")
    return img0


inputs = [
    gr.Image(type="filepath", label="Input Image"),
    gr.Slider(minimum=320, maximum=1280, step=32, label="Image Size", value=640),
    gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label="Confidence Threshold"),
    gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label="IOU Threshold"),
]

outputs = gr.Image(type="filepath", label="Output Image")

title = "Custom_YOLOV9_model 🤖: leaf-detector 👔🧦💫 "

examples = [['1.jpg', 640, 0.5, 0.7],
            ['2.jpg', 640, 0.5, 0.6],
            ['3.jpg', 640, 0.5, 0.8]]

yolo_app = gr.Interface(
    fn=run_inference,
    inputs=inputs,
    outputs=outputs,
    title=title,
    examples=examples,
    cache_examples=False,
)

# Launch the Gradio interface in debug mode with queue enabled
yolo_app.launch(debug=True, share=True).queue()




# run_inference("/Users/kvzm411/Desktop/ERA V2/ERA_V2/S15/yolov9-main/runs/train/yolov9-c-tumor/weights/best.pt",\
#     "/Users/kvzm411/Desktop/ERA V2/ERA_V2/S15/tumor/images/volume_1_slice_65.jpg",\
#         output_dir='/Users/kvzm411/Desktop/ERA V2/ERA_V2/S15_Gradio_app/',\
#             img_size=(640, 640),\
#                 conf_thres=0.25, \
#                     iou_thres=0.45,\
#                     max_det=1000)    
