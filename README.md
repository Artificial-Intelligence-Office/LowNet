# LowNet

Implementation of our ICIP 2020 paper "Lownet: privacy preserved Ultra-Low Resolution Posture Image Classification"

In this project, we created LowNet architecture, which is suitable for low resolution image classification. 
We are releasing TIP38(Thermal Image Posture 38 class) yoga posture image dataset captured by infrared camera. We propose "Lownet" model with relu activation fucntions that have variable slopes.
<p align = "center">
<img src="images/architecture.png" width="600" >
</p>
<!--
![](architecture.png 60x20)
-->

## Custom loss : 
<!--
<img src="https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />
-->
<!--
<img src="https://latex.codecogs.com/svg.latex?\Large&space;
       L= \sum_{i=1}^{N} {{α * (1- {widehat {y}} rsub {i} right )} ^ {β} { y} rsub {i } *\log ({{y} rsub {i}} over {{widehat {y}} rsub {i}} right )} " " />
-->
We created custom loss function by mixing focal loss and KLD( Kullback-Leibler Divergence) loss function.
<img src="images/Loss.png" width="500" >


## Datasets

A total of 23 volunteers, 6 males and 17 females were recruited, for the collection of posture dataset. The figure shows 38 posture samples we adopted in this study.  The upper row displays 64×64 cropped images from the original infrared images. Original images' resolution were 480x480. These images cropped were further down-sampled to 16×16 posture images to serve as ultra-low-resolution training data for the LowNet model.
These ultra-low-resolution images effectively prevent the leakage of personal identification, thwart the invasion of privacy, and reduce the sensor unit cost. The volunteers performed each posture for 10 seconds in front of the infrared camera. Among the postures, 11 of them had two versions, e.g., Half-moon posture has two versions, i.e., one with the right leg raised and the other with the left leg raised. Different versions of the same posture were considered as one class. Moreover, we adopted three versions of faint posture, i.e., facing up, down, and side.
Five images were extracted every two seconds from each 10 seconds long posture video. We randomly chose 25 samples from each class to create a balanced dataset. Thus, a total of 4374 samples employed in the dataset where there were 115 samples for each posture. Then the dataset is divided into a training_set and a test_set, which are consisting of samples of 18 (3420 samples) and 5 (954 samples) people, respectively, where each person’s data belongs to only of the sets.

<!--
![](big_img_64x64.jpg)
-->
#### 64x64 resolution images
<img src="images/big_img_64x64_2_20.jpg" width=1000 class="center">

#### 16x16 resolution images
<img src="images/big_img_16x16_2_20.jpg" width=1000 class="center">
<!--
![](big_img_16x16.jpg)
-->


# Results
We trained our model with different losses. Lownet model with our custom loss function yields better f1 score on our dataset.

<p align = "center">
<img src="images/loss_compare.png" width="500" >
</p>

Trained other SOTA models on our dataset and compared our LowNet model with these models. Our model performed better.
# Our paper

If you found this repository useful, please cite our paper
```
@InProceedings{Munhjargal_2020_ICIP,
  author = {Munkhjargal Gochoo, Tan-Hsu Tan, Fady Alnajjar, Jun-Wei Hsieh, and Ping-Yang Chen},
  title = {Lownet: privacy preserved Ultra-Low Resolution Posture Image Classification},
  booktitle = {The IEEE International Conference on Image Processing (ICIP)},
  month = {Oct},
  year = {2020}
}
```
