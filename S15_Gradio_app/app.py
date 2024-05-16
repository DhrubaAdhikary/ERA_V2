import gradio as gr
import os
import sys

# Get the directory of the current script file
script_directory = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the script directory
os.chdir(script_directory)
from ultralytics import YOLO
# Build a YOLOv9c model from scratch
model = YOLO('yolov9e.yaml')
model = YOLO('last.pt')  # load your custom trained model
import torch
#from ultralyticsplus import render_result
from render import custom_render_result



def yoloV9_func(image: gr.Image = None,
                image_size: int = 640,
                conf_threshold: float = 0.4,
                iou_threshold: float = 0.5):
    """This function performs YOLOv9 object detection on the given image.
    Args:
        image (gr.Image, optional): Input image to detect objects on. Defaults to None.
        image_size (int, optional): Desired image size for the model. Defaults to 640.
        conf_threshold (float, optional): Confidence threshold for object detection. Defaults to 0.4.
        iou_threshold (float, optional): Intersection over Union threshold for object detection. Defaults to 0.50.
    """
    # Load the YOLOv9 model from the 'best.pt' checkpoint
    # model_path = "yolov5s.pt"
    # model = torch.hub.load('ultralytics/yolov8', 'custom', path='/content/best.pt', force_reload=True, trust_repo=True)

    # Perform object detection on the input image using the YOLOv8 model
    results = model.predict(image,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            imgsz=image_size)

    # Print the detected objects' information (class, coordinates, and probability)
    box = results[0].boxes
    print("Object type:", box.cls)
    print("Coordinates:", box.xyxy)
    print("Probability:", box.conf)

    # Render the output image with bounding boxes around detected objects
    render = custom_render_result(model=model, image=image, result=results[0])
    return render


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
    fn=yoloV9_func,
    inputs=inputs,
    outputs=outputs,
    title=title,
    examples=examples,
    cache_examples=False,
)

# Launch the Gradio interface in debug mode with queue enabled
yolo_app.launch(debug=True, share=True).queue()