# YoloV3
________
YoloV3 Simplified for training on Colab with custom dataset. 

_A Collage of Training images_
![image](https://github.com/DhrubaAdhikary/ERA_V2/blob/47b23456f3fe6751465f8b01004ef76684524b32/S12/output/Training_set_visual.png)


We have added a very 'smal' Coco sample imageset in the folder called smalcoco. This is to make sure you can run it without issues on Colab.

Full credit goes to [this](https://github.com/ultralytics/yolov3), and if you are looking for much more detailed explainiation and features, please refer to the original [source](https://github.com/ultralytics/yolov3). 

You'll need to download the weights from the original source. 
1. Create a folder called weights in the root (YoloV3) folder
2. Download from: https://drive.google.com/file/d/1vRDkpAiNdqHORTUImkrpD7kK_DkCcMus/view?usp=share_link
3. Place 'yolov3-spp-ultralytics.pt' file in the weights folder:
  * to save time, move the file from the above link to your GDrive
  * then drag and drop from your GDrive opened in Colab to weights folder
4. run this command
`python train.py --data data/smalcoco/smalcoco.data --batch 10 --cache --epochs 25 --nosave`

For custom dataset:
1. Clone this repo: https://github.com/miki998/YoloV3_Annotation_Tool
2. Follow the installation steps as mentioned in the repo. 
3. For the assignment, download 500 images of your unique object. 
4. Annotate the images using the Annotation tool. 
```
data
  --tumor
    --images/
      --img001.jpg
      --img002.jpg
      --...
    --labels/
      --img001.txt
      --img002.txt
      --...
    custom.data #data file
    custom.names #your class names
    custom.txt #list of name of the images you want your network to be trained on. Currently we are using same file for test/train
```
5. As you can see above you need to create **custom.data** file. For 1 class example, your file will look like this:
```
classes=1
train=./data/tumor/custom.txt
valid=./data/tumor/custom1.txt
names=./data/tumor/custom.names
```

6. You need to add custom.names file as you can see above. For our example, we downloaded images of Brain Tumor. Our custom.names file look like this:
```
Tumor
```
8. Tumor above will have a class index of 0. 
9. For COCO's 80 classes, VOLOv3's output vector has 255 dimensions ( (4+1+80)*3). Now we have 1 class, so we would need to change it's architecture. for 1 class (5+1)*3 = 18 
10. Copy the contents of 'yolov3-spp.cfg' file to a new file called 'yolov3-custom.cfg' file in the data/cfg folder. 
11. Search for 'filters=255' (you should get entries entries). Change 255 to 18 = (4+1+1)*3
12. Search for 'classes=80' and change all three entries to 'classes=1'
13. Don't forget to perform the weight file steps mentioned in the sectio above. 
15. Run this command `python train.py --data data/customdata/custom.data --batch 10 --cache --cfg cfg/yolov3-custom.cfg --epochs 3 --nosave`

As you can see in the collage image above, a lot is going on, and if you are creating a set of say 100 images, you'd get a bonanza of images via default augmentations being performed. 

** Training Set Visualized **

![alt text](https://github.com/DhrubaAdhikary/ERA_V2/blob/47b23456f3fe6751465f8b01004ef76684524b32/S12/output/Training_set_visual.png)



**Training Logs**
Run 'tensorboard --logdir=runs' to view tensorboard at http://localhost:6006/
Namespace(epochs=150, batch_size=2, accumulate=4, cfg='cfg/yolov3-custom.cfg', data='/content/data/tumor/custom.data', multi_scale=False, img_size=[512, 512, 512], rect=False, resume=False, nosave=True, notest=False, evolve=False, bucket='', cache_images=True, weights='weights/yolov3-spp-ultralytics.pt', name='', device='', adam=False, single_cls=False)
/content/data/tumor/custom.data
{'classes': '1', 'train': './data/tumor/custom.txt', 'valid': './data/tumor/custom.txt', 'names': './data/tumor/custom.names'}


Caching images (0.0GB): 100% 24/24 [00:00<00:00, 2960.51it/s]
Image sizes 512 - 512 train, 512 test
Using 2 dataloader workers
Starting training for 150 epochs...

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
/usr/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
     0/149        0G      5.92       125         0       130         3       512: 100% 12/12 [02:56<00:00, 14.70s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1:   0% 0/12 [00:00<?, ?it/s]/usr/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
/usr/local/lib/python3.10/dist-packages/torch/functional.py:507: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3549.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [01:04<00:00,  5.39s/it]
                 all        24        28  0.000165     0.643  0.000502  0.000331

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     1/149        0G      5.08      39.2         0      44.3         5       512: 100% 12/12 [03:01<00:00, 15.14s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [01:11<00:00,  5.93s/it]
                 all        24        28  0.000651     0.179   0.00131    0.0013

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     2/149        0G      4.98      8.62         0      13.6         4       512: 100% 12/12 [02:57<00:00, 14.80s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [01:06<00:00,  5.53s/it]
                 all        24        28         0         0  0.000389         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     3/149        0G      5.49      3.13         0      8.62         1       512: 100% 12/12 [02:56<00:00, 14.72s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [01:03<00:00,  5.32s/it]
                 all        24        28         0         0  0.000908         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     4/149        0G      5.21      2.22         0      7.43         2       512: 100% 12/12 [02:49<00:00, 14.15s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [02:35<00:00, 12.95s/it]
                 all        24        28         0         0  0.000865         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     5/149        0G      4.46      2.06         0      6.52         2       512: 100% 12/12 [02:53<00:00, 14.44s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [01:44<00:00,  8.71s/it]
                 all        24        28         0         0   0.00223         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     6/149        0G      4.54      2.68         0      7.22         4       512: 100% 12/12 [02:53<00:00, 14.43s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [01:09<00:00,  5.83s/it]
                 all        24        28         0         0   0.00442         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     7/149        0G      4.14      2.35         0      6.49         4       512: 100% 12/12 [02:54<00:00, 14.56s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:52<00:00,  4.37s/it]
                 all        24        28         0         0  8.51e-06         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     8/149        0G       4.8      2.64         0      7.44         4       512: 100% 12/12 [02:54<00:00, 14.56s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.31s/it]
                 all        24        28         0         0         0         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     9/149        0G      4.15      1.99         0      6.14         2       512: 100% 12/12 [02:52<00:00, 14.34s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:56<00:00,  4.73s/it]
                 all        24        28         0         0         0         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    10/149        0G       3.5      2.17         0      5.67         2       512: 100% 12/12 [02:54<00:00, 14.52s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.29s/it]
                 all        24        28         0         0         0         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    11/149        0G      4.97      2.01         0      6.98         2       512: 100% 12/12 [02:53<00:00, 14.47s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.33s/it]
                 all        24        28         0         0         0         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    12/149        0G      3.02      1.73         0      4.76         4       512: 100% 12/12 [02:52<00:00, 14.37s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:56<00:00,  4.75s/it]
                 all        24        28         0         0    0.0015         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    13/149        0G      4.12      1.61         0      5.73         4       512: 100% 12/12 [02:53<00:00, 14.44s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.25s/it]
                 all        24        28         0         0   0.00143         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    14/149        0G      4.63      1.86         0      6.49         4       512: 100% 12/12 [02:52<00:00, 14.41s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.25s/it]
                 all        24        28         0         0   0.00169         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    15/149        0G      4.04      1.38         0      5.43         3       512: 100% 12/12 [02:53<00:00, 14.49s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:56<00:00,  4.69s/it]
                 all        24        28         0         0  0.000571         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    16/149        0G      3.19      1.28         0      4.47         3       512: 100% 12/12 [02:57<00:00, 14.75s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:55<00:00,  4.65s/it]
                 all        24        28         0         0  7.62e-05         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    17/149        0G       3.5      1.48         0      4.98         2       512: 100% 12/12 [02:52<00:00, 14.37s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:56<00:00,  4.70s/it]
                 all        24        28         0         0  0.000154         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    18/149        0G      4.27      1.12         0      5.39         3       512: 100% 12/12 [02:52<00:00, 14.41s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:50<00:00,  4.21s/it]
                 all        24        28         0         0  4.95e-05         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    19/149        0G      4.53      1.13         0      5.66         1       512: 100% 12/12 [02:51<00:00, 14.33s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.28s/it]
                 all        24        28         0         0   0.00102         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    20/149        0G      3.52     0.966         0      4.49         3       512: 100% 12/12 [02:53<00:00, 14.43s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.32s/it]
                 all        24        28         0         0   0.00156         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    21/149        0G      2.84      1.06         0       3.9         4       512: 100% 12/12 [02:57<00:00, 14.78s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:50<00:00,  4.23s/it]
                 all        24        28         0         0   0.00631         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    22/149        0G      2.82         1         0      3.82         4       512: 100% 12/12 [02:53<00:00, 14.45s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:53<00:00,  4.47s/it]
                 all        24        28         0         0   0.00474         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    23/149        0G      3.33      1.18         0      4.51         3       512: 100% 12/12 [02:54<00:00, 14.52s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:52<00:00,  4.41s/it]
                 all        24        28         0         0   0.00221         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    24/149        0G       2.7     0.911         0      3.61         3       512: 100% 12/12 [02:52<00:00, 14.40s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.29s/it]
                 all        24        28         0         0   0.00981         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    25/149        0G      3.87      1.12         0      4.99         5       512: 100% 12/12 [02:54<00:00, 14.56s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.32s/it]
                 all        24        28         0         0   0.00599         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    26/149        0G      3.54         1         0      4.54         2       512: 100% 12/12 [02:54<00:00, 14.54s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.32s/it]
                 all        24        28         0         0    0.0161         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    27/149        0G      3.58      0.94         0      4.52         3       512: 100% 12/12 [02:54<00:00, 14.51s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:52<00:00,  4.37s/it]
                 all        24        28         0         0    0.0103         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    28/149        0G      2.98         1         0      3.98         4       512: 100% 12/12 [02:53<00:00, 14.46s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:52<00:00,  4.35s/it]
                 all        24        28    0.0717    0.0357    0.0147    0.0477

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    29/149        0G      2.74     0.827         0      3.56         4       512: 100% 12/12 [02:53<00:00, 14.45s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.32s/it]
                 all        24        28    0.0349    0.0357    0.0161    0.0353

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    30/149        0G      3.23     0.841         0      4.07         3       512: 100% 12/12 [02:54<00:00, 14.51s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.33s/it]
                 all        24        28    0.0263    0.0113   0.00742    0.0158

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    31/149        0G      2.39     0.719         0       3.1         2       512: 100% 12/12 [02:52<00:00, 14.36s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:53<00:00,  4.44s/it]
                 all        24        28         0         0    0.0189         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    32/149        0G      2.89      0.81         0       3.7         3       512: 100% 12/12 [02:51<00:00, 14.28s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:55<00:00,  4.58s/it]
                 all        24        28         0         0   0.00241         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    33/149        0G      2.96     0.776         0      3.73         1       512: 100% 12/12 [02:52<00:00, 14.39s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:54<00:00,  4.51s/it]
                 all        24        28    0.0285    0.0357   0.00301    0.0317

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    34/149        0G      2.83     0.986         0      3.82         3       512: 100% 12/12 [02:53<00:00, 14.44s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:52<00:00,  4.39s/it]
                 all        24        28    0.0643    0.0276   0.00499    0.0386

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    35/149        0G      2.36     0.712         0      3.07         3       512: 100% 12/12 [02:51<00:00, 14.30s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.32s/it]
                 all        24        28         0         0   0.00935         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    36/149        0G      2.96      0.79         0      3.75         2       512: 100% 12/12 [02:53<00:00, 14.45s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:52<00:00,  4.42s/it]
                 all        24        28         0         0    0.0137         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    37/149        0G      2.39     0.734         0      3.13         4       512: 100% 12/12 [02:53<00:00, 14.46s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.31s/it]
                 all        24        28         0         0   0.00837         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    38/149        0G      1.77     0.719         0      2.49         3       512: 100% 12/12 [02:56<00:00, 14.69s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.29s/it]
                 all        24        28         0         0   0.00494         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    39/149        0G      2.65     0.785         0      3.43         3       512: 100% 12/12 [02:53<00:00, 14.48s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.30s/it]
                 all        24        28         0         0    0.0194         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    40/149        0G      2.59      0.78         0      3.37         4       512: 100% 12/12 [02:54<00:00, 14.51s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.29s/it]
                 all        24        28      0.39    0.0714    0.0577     0.121

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    41/149        0G      2.48     0.773         0      3.25         3       512:  67% 8/12 [01:55<00:56, 14.08s/it]
Model Bias Summary:    layer        regression        objectness    classification
                          89      -0.09+/-0.15      -5.29+/-1.01       0.02+/-0.01 
                         101       0.04+/-0.12      -5.11+/-0.40       0.04+/-0.00 
                         113       0.02+/-0.05      -4.48+/-0.76      -0.00+/-0.01 
    41/149        0G       2.8     0.787         0      3.58         4       512: 100% 12/12 [02:53<00:00, 14.48s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.25s/it]
                 all        24        28     0.113    0.0357    0.0269    0.0543

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    42/149        0G       2.3      0.77         0      3.07         2       512: 100% 12/12 [02:57<00:00, 14.78s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:50<00:00,  4.25s/it]
                 all        24        28     0.187     0.107     0.078     0.136

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    43/149        0G      2.35     0.634         0      2.99         3       512: 100% 12/12 [02:49<00:00, 14.09s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:53<00:00,  4.44s/it]
                 all        24        28     0.218     0.107    0.0926     0.144

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    44/149        0G      3.44     0.725         0      4.17         4       512: 100% 12/12 [02:53<00:00, 14.49s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:53<00:00,  4.47s/it]
                 all        24        28     0.152    0.0714    0.0799    0.0971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    45/149        0G      2.84     0.758         0       3.6         4       512: 100% 12/12 [02:53<00:00, 14.45s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:53<00:00,  4.46s/it]
                 all        24        28     0.252     0.107     0.142      0.15

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    46/149        0G      2.47     0.637         0      3.11         2       512: 100% 12/12 [02:54<00:00, 14.53s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.30s/it]
                 all        24        28     0.388     0.107     0.132     0.168

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    47/149        0G      2.19     0.759         0      2.95         2       512: 100% 12/12 [02:56<00:00, 14.71s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:52<00:00,  4.38s/it]
                 all        24        28     0.153    0.0357     0.114    0.0579

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    48/149        0G      2.34     0.721         0      3.06         4       512: 100% 12/12 [02:55<00:00, 14.64s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.32s/it]
                 all        24        28    0.0442    0.0126    0.0479    0.0197

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    49/149        0G       1.9      0.71         0      2.61         4       512: 100% 12/12 [02:55<00:00, 14.66s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:53<00:00,  4.43s/it]
                 all        24        28         0         0    0.0301         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    50/149        0G      2.17     0.706         0      2.88         4       512: 100% 12/12 [02:52<00:00, 14.41s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.25s/it]
                 all        24        28         0         0    0.0547         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    51/149        0G      2.26     0.561         0      2.82         2       512: 100% 12/12 [02:52<00:00, 14.41s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:50<00:00,  4.20s/it]
                 all        24        28         0         0     0.029         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    52/149        0G      2.29     0.618         0       2.9         3       512: 100% 12/12 [02:50<00:00, 14.21s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.27s/it]
                 all        24        28         0         0     0.012         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    53/149        0G      1.91     0.642         0      2.55         2       512: 100% 12/12 [02:53<00:00, 14.46s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:51<00:00,  4.28s/it]
                 all        24        28         0         0   0.00807         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    54/149        0G      2.14     0.726         0      2.86         2       512: 100% 12/12 [02:54<00:00, 14.55s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:52<00:00,  4.35s/it]
                 all        24        28         0         0   0.00839         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    55/149        0G      2.19     0.546         0      2.74         3       512: 100% 12/12 [02:56<00:00, 14.69s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:52<00:00,  4.36s/it]
                 all        24        28         1    0.0357    0.0445     0.069

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    56/149        0G      1.76     0.608         0      2.37         3       512: 100% 12/12 [02:53<00:00, 14.47s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:50<00:00,  4.23s/it]
                 all        24        28         0         0    0.0281         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    57/149        0G      2.16      0.55         0      2.71         4       512: 100% 12/12 [02:52<00:00, 14.41s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:52<00:00,  4.37s/it]
                 all        24        28         1    0.0357    0.0394     0.069

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    58/149        0G      2.35     0.707         0      3.06         3       512: 100% 12/12 [02:55<00:00, 14.61s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 12/12 [00:52<00:00,  4.35s/it]
                 all        24        28         1    0.0357    0.0636     0.069

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
  0% 0/12 [00:00<?, ?it/s]

**Results**
After training for 100 Epochs, results look awesome!

![image](https://github.com/DhrubaAdhikary/ERA_V2/blob/47b23456f3fe6751465f8b01004ef76684524b32/S12/output/Validation_Image.png)
