<h1 align="center">
  <br>
  <a href="https://github.com/kra0s22"><img src="https://raw.githubusercontent.com/kra0s22/Training_YOLOv7_on_Custom_Data/master/Images/yologo_2.png" alt="Markdownify" width="200"></a>
  <br>
  Testing YOLOv7 on Google Colab
  <br>
</h1>

<h4 align="center"> An initial aproach of YOLOv7 using <a href="https://colab.research.google.com/" target="_blank">Google Colab</a>.</h4>

<p align="center">
<a href="https://colab.research.google.com/drive/1wt8lQXGpkrlhlG4yOm__AypLoMzgq_td?usp=share_link" rel="nofollow"><img src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;"></a>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#download">Download</a> •
  <a href="#model_implementation">Model implementation</a> •
  <a href="#solution_analysis">Solution analysis</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#credits">Credits</a> •
  <a href="#related">Related</a>
</p>

## Introduction

For this project we will explore the use of YOLO7 over Colab functionalities, developing a weapon detection model for secure and reliable purpose.

## Key Features

* Instalation guide of YOLO and other dependeces in Colab.
* Example usage of RoboFlow and YOLOv7.
* Training and test of a YOLOv7 model and save up in Google Drive.
* Evaluate the obtained solutions based on the different evaluation methos used.
* Run a evaluation images on the model to test the results.
* Exportation of the trained model to a local directory for future experiments.

## How To Use

To clone and run the .ipynb, you'll need [Git](https://git-scm.com). From your command line:

```bash
# Clone this repository
$ git clone https://github.com/kra0s22/Training_YOLOv7_on_Custom_Data.git

# Go into the repository
$ cd Training_YOLOv7_on_Custom_Data
```

> **Note**
> If you're using Linux Bash for Windows, [see this guide](https://www.howtogeek.com/261575/how-to-run-graphical-linux-desktop-applications-from-windows-10s-bash-shell/).


## Download

You can [download](https://github.com/kra0s22/Training_YOLOv7_on_Custom_Data/archive/refs/heads/master.zip) the latest version of this repository.

## Model implementation

Our final model is the following, expectin to fulfill our initial pourpose and detect the most cuantity of weapos expecting to be reliable.
```bash
# run this cell to begin training1
%cd /content/yolov7
!python train.py --batch 20 --epochs 80 --data {dataset.location}/data.yaml --weights 'yolov7_training.pt'
```

The hyperparameters are calculated automatically by YOLOV7 through a block of calculations for a good all pourpose training and test algorithm as it is seen in the following cell


```bash
hyperparameters: lr0=0.01, lrf=0.1, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.3, cls_pw=1.0, obj=0.7, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.2, scale=0.9, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.15, copy_paste=0.0, paste_in=0.15, loss_ota=1
```
With the above hyperparámets, YOLO generate the following model:
```bash
Model Summary: 415 layers, 37196556 parameters, 37196556 gradients, 105.1 GFLOPS
```
(the entire model construction is exposed during the training execution)


## Solution analysis
In this sections we will explain the mAP (mean Average Precision) used for the development of this project.

To begin with, AP is a popular metric for accuracy measuring in objects detections. This measurement gest values from 0 to 1 to rate the detected image prediction. To build this metris it is necesary to understand precision, recall and IoU.

### Precision
Starting withe the following 4 possible predictions:

* TP = True positive
* TN = True negative
* FP = False positive
* FN = False negative

Precision measeures how accurate is your prediciton and uses the following definition:

$$Precision = \frac{TP}{TP + FP}$$

![P curve](https://raw.githubusercontent.com/kra0s22/Training_YOLOv7_on_Custom_Data/master/export/runs/train/exp2/P_curve.png)

### Recall
Recall measures how good you find all the positives and uses the following definition:

$$Recall = \frac{TP}{TP + FN}$$

![R curve](https://raw.githubusercontent.com/kra0s22/Training_YOLOv7_on_Custom_Data/master/export/runs/train/exp2/R_curve.png)


### IoU (Intersection over union)

IoU measures the overlap between 2 boundaries. We use that to measure how much our predicted boundary overlaps with the ground truth (the real object boundary). In some datasets, we predefine an IoU threshold (mainly 0.5) in classifying whether the prediction is a true positive or a false positive.

### AP
AP (Average precision) is a popular metric in measuring the accuracy of object detectors like Faster R-CNN, SSD, etc. Average precision computes the average precision value for recall value over 0 to 1

$$AP = $$\int_0^1 p(r) \,dx= \frac13$$$$

![PR curve](https://raw.githubusercontent.com/kra0s22/Training_YOLOv7_on_Custom_Data/master/export/runs/train/exp2/PR_curve.png)

### COCO mAP

Latest research papers tend to give results for the COCO dataset only. In COCO mAP, a 101-point interpolated AP definition is used in the calculation. For COCO, AP is the average over multiple IoU (the minimum IoU to consider a positive match). AP@[.5:.95] corresponds to the average AP for IoU from 0.5 to 0.95 with a step size of 0.05. For the COCO competition, AP is the average over 10 IoU levels on 80 categories (AP@[.50:.05:.95]: start from 0.5 to 0.95 with a step size of 0.05). The following are some other metrics collected for the COCO dataset.

mAP (mean average precision) is the average of AP. In some context, we compute the AP for each class and average them. But in some context, they mean the same thing. For example, under the COCO context, there is no difference between AP and mAP.
![PR curve](https://raw.githubusercontent.com/kra0s22/Training_YOLOv7_on_Custom_Data/master/export/runs/train/exp2/results.png)

So the orange line is transformed into the green lines and the curve will decrease monotonically instead of the zigzag pattern. The calculated AP value will be less suspectable to small variations in the ranking. Mathematically, we replace the precision value for recall ȓ with the maximum precision for any recall ≥ ȓ.


### F1
F1 combines precision and recall into one metric by calculating the harmonic mean between those two. It is actually a special case of the more general function F beta:

$$F1 = \frac{precision \cdot{recall}}{precision + recall} \cdot {Beta}$$

![F1 curve](https://raw.githubusercontent.com/kra0s22/Training_YOLOv7_on_Custom_Data/master/export/runs/train/exp2/F1_curve.png)

The above training and validation execution ends with the plots exposed in the following [link](https://wandb.ai/kra0s22/YOLOR/workspace?workspace=user-kra0s22).

## Evaluation
As it is said in the following [discussions](https://github.com/ultralytics/yolov5/discussions/7906), YOLO5 had 3 files that are designed for different purposes and utilize different dataloaders with different settings. Currently, in YOLO7 the functionalities are dispossed differently, **train.py** dataloaders are designed for a speed-accuracy compromise, **test.py** contains the possibility of use the **train**, **val**, **test**, **speed** or **study** functionality (default object confidence threshold 0.001 and IOU threshold for NMS 0.65) and **detect.py** is designed for best real-world inference results (default object confidence threshold 0.25 and IOU threshold for NMS 0.45).

For a real-world situation, we can evaluate the performance of our custom training using the provided **evalution** script and the best results model in **/runs/train/exp/weights/best.pt**. Similarly to the train.py function, detect.py has a lot of arguments accesible from --help or using the following [webpage](https://github.com/WongKinYiu/yolov7/blob/main/detect.py#L154).

Our first execution of **detect.py** stud with the confidence threshold and IOU threshold by default, showing good detections but by contras showing a lot of FP and low precision as it is visible in the following images:

![Failure1](https://raw.githubusercontent.com/kra0s22/Training_YOLOv7_on_Custom_Data/master/Images/failure1.jpg)

![Failure2](https://raw.githubusercontent.com/kra0s22/Training_YOLOv7_on_Custom_Data/master/Images/failure2.jpg)



We could stop or detection and evalution process here but it would not cover or initial pourpose of consistent detection of weapons. Because of it, our next detection process had the following parameters:
```bash
# Run evaluation
# necessary to change to this directory, otherwise the yolo7'2 detection, validation or test .py will not work.
%cd /content/yolov7
!python detect.py --conf-thres 0.6 --iou-thresh 0.7 --weights ./runs/train/best.pt --source {dataset.location}/valid/images --name r-w-images
```

![Failure3](https://raw.githubusercontent.com/kra0s22/Training_YOLOv7_on_Custom_Data/master/Images/failure3.jpg)

![Failure4](https://raw.githubusercontent.com/kra0s22/Training_YOLOv7_on_Custom_Data/master/Images/failure4.jpg)


![success1](https://raw.githubusercontent.com/kra0s22/Training_YOLOv7_on_Custom_Data/master/Images/success1.jpg)
![success2](https://raw.githubusercontent.com/kra0s22/Training_YOLOv7_on_Custom_Data/master/Images/success2.jpg)
![success3](https://raw.githubusercontent.com/kra0s22/Training_YOLOv7_on_Custom_Data/master/Images/success3.jpg)
![success4](https://raw.githubusercontent.com/kra0s22/Training_YOLOv7_on_Custom_Data/master/Images/success4.jpg)

Due to its high restrictions this detections has a high FN rate but a confindent weapon detection, se the few weapons detected are reliable. In cosequence, we will aply a more intermedium images with les restrictions at it is seen in the following cell:

```bash
# Run evaluation
# necessary to change to this directory, otherwise the yolo7'2 detection, validation or test .py will not work.
%cd /content/yolov7
!python detect.py --conf-thres 0.4 --iou-thresh 0.5 --weights ./runs/train/best.pt --source {dataset.location}/valid/images --name r-w-images
```

![rw2failure1](https://raw.githubusercontent.com/kra0s22/Training_YOLOv7_on_Custom_Data/master/Images/rw2failure1.jpg)

![rw2failure2](https://raw.githubusercontent.com/kra0s22/Training_YOLOv7_on_Custom_Data/master/Images/rw2failure2.jpg)


![rw2success](https://raw.githubusercontent.com/kra0s22/Training_YOLOv7_on_Custom_Data/master/Images/rw2success.jpg)
![success2](https://raw.githubusercontent.com/kra0s22/Training_YOLOv7_on_Custom_Data/master/Images/success2.jpg)




## Credits

This software uses the following open source packages:

- [Darknet](https://pjreddie.com/darknet/yolo/)
- [RoboFlow](https://roboflow.com/)
- [Stackedit.io - a markdown parser](https://stackedit.io/)
- [Markdown Viewer and Editor](https://thumbsdb.herokuapp.com/markdown/?)
- [ReadMe - template](https://github.com/amitmerchant1990/electron-markdownify#readme)
- [Explanation of Precision, Recall, IoU, mAP and F1](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)
- [Discussion of different functionalities of YOLO's .py](https://github.com/ultralytics/yolov5/discussions/7906)

## Related

- [How to use git on colab](https://medium.com/analytics-vidhya/how-to-use-google-colab-with-github-via-google-drive-68efb23a42d)
- [RoboFlow blogspot](https://blog.roboflow.com/yolov7-custom-dataset-training-tutorial/) - Blog How to Train YOLOv7 on a Custom Dataset from

## You may also like...
- [Tweets and Users collector using MongoDB and V2 API](https://github.com/kra0s22/TFG-Tweets_and_Users_recollector_using_V2_API)
- [Project developed on Unity from the AI for videogames subject [ESP]](https://github.com/kra0s22/PROYECTO-IADEVIDEOJUEGOS-Simon-Alberto-Jose)



---

> GitHub [@kra0s22](https://github.com/kra0s22) &nbsp;&middot;&nbsp;
> Twitter [@kra0s22](https://twitter.com/kra0s22)


