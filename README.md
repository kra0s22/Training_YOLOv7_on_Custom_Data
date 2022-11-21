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
  <a href="#credits">Credits</a> •
  <a href="#related">Related</a> •
  <a href="#license">License</a>
</p>

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
Aqui va yolo, como se obtienen los hyperpar'ametros y otro svalroes de yolo para el entrenamiento.

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
![P curve](https://drive.google.com/uc?export=view&id=1pUxsnoWEo6xxQkjoancih6c9Y7e4-IGV)

### Recall
Recall measures how good you find all the positives and uses the following definition:

$$Recall = \frac{TP}{TP + FN}$$

![R curve](https://drive.google.com/uc?export=view&id=10FMCVMBP3UrnNIsMAAAnvPnKuKjVjCHW)


### IoU (Intersection over union)

IoU measures the overlap between 2 boundaries. We use that to measure how much our predicted boundary overlaps with the ground truth (the real object boundary). In some datasets, we predefine an IoU threshold (mainly 0.5) in classifying whether the prediction is a true positive or a false positive.

### AP
AP (Average precision) is a popular metric in measuring the accuracy of object detectors like Faster R-CNN, SSD, etc. Average precision computes the average precision value for recall value over 0 to 1

$$AP = $$\int_0^1 p(r) \,dx= \frac13$$$$

![PR curve](https://drive.google.com/uc?export=view&id=1JliHoR-0pb25T12X79mk9gH9iPlp-kzZ)

### COCO mAP

Latest research papers tend to give results for the COCO dataset only. In COCO mAP, a 101-point interpolated AP definition is used in the calculation. For COCO, AP is the average over multiple IoU (the minimum IoU to consider a positive match). AP@[.5:.95] corresponds to the average AP for IoU from 0.5 to 0.95 with a step size of 0.05. For the COCO competition, AP is the average over 10 IoU levels on 80 categories (AP@[.50:.05:.95]: start from 0.5 to 0.95 with a step size of 0.05). The following are some other metrics collected for the COCO dataset.

mAP (mean average precision) is the average of AP. In some context, we compute the AP for each class and average them. But in some context, they mean the same thing. For example, under the COCO context, there is no difference between AP and mAP.

![Graphs](https://drive.google.com/uc?export=view&id=1pFQ3aWPaMWn3BjWZXA_g_Ph8j4f-mv-x)

So the orange line is transformed into the green lines and the curve will decrease monotonically instead of the zigzag pattern. The calculated AP value will be less suspectable to small variations in the ranking. Mathematically, we replace the precision value for recall ȓ with the maximum precision for any recall ≥ ȓ.


### F1
Simply put, it combines precision and recall into one metric by calculating the harmonic mean between those two. It is actually a special case of the more general function F beta:

$$F1 = \frac{precision \cdot{recall}}{precision + recall} \cdot {Beta}$$

![F1 curve](https://drive.google.com/uc?export=view&id=1A2ivvryH0N4AOWlLjK76RGbCQbA02Kif)

The above training and validation execution ends with the plots exposed in the following [link](https://wandb.ai/kra0s22/YOLOR/workspace?workspace=user-kra0s22).

## Evaluation

We can evaluate the performance of our custom training using the provided evalution script (**train.py**) and the best results model in **best.pt**.

Aqui van imagenes obtenidas en el entrenamiento que parezcan ser buenas

## Credits

This software uses the following open source packages:

- [Darknet](https://pjreddie.com/darknet/yolo/)
- [RoboFlow](https://roboflow.com/)
- [Stackedit.io - a markdown parser](https://stackedit.io/)
- [Markdown Viewer and Editor](https://thumbsdb.herokuapp.com/markdown/?)
- [ReadMe - template](https://github.com/amitmerchant1990/electron-markdownify#readme)
- [Explanation of Precision, Recall, IoU, mAP and F1](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)

## Related

- [How to use git on colab](https://medium.com/analytics-vidhya/how-to-use-google-colab-with-github-via-google-drive-68efb23a42d)
- [RoboFlow blogspot](https://blog.roboflow.com/yolov7-custom-dataset-training-tutorial/) - Blog How to Train YOLOv7 on a Custom Dataset from

## You may also like...
- [Tweets and Users collector using MongoDB and V2 API](https://github.com/kra0s22/TFG-Tweets_and_Users_recollector_using_V2_API)
- [Project developed on Unity from the AI for videogames subject [ESP]](https://github.com/kra0s22/PROYECTO-IADEVIDEOJUEGOS-Simon-Alberto-Jose)



---

> GitHub [@kra0s22](https://github.com/kra0s22) &nbsp;&middot;&nbsp;
> Twitter [@kra0s22](https://twitter.com/kra0s22)


