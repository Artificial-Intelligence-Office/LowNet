# LowNet

Implementation of our ICIP 2020 paper "Lownet: privacy preserved Ultra-Low Resolution Posture Image Classification"

In this project, we created LowNet architecture, which is suitable for low resolution image classification. 
We are releasing TIP38(Thermal Image Posture 38 class) yoga posture image dataset captured by thermal camera.
<p align = "center">
<img src="architecture.png" width="600" >
</p>
<!--
![](architecture.png 60x20)
-->

## Custom loss : 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />

<img src="https://latex.codecogs.com/svg.latex?\Large&space;
       L= \sum_{i=1}^{N} {{α * (1- {widehat {y}} rsub {i} right )} ^ {β} { y} rsub {i } *\log ({{y} rsub {i}} over {{widehat {y}} rsub {i}} right )} " " />


```
 $ L= sum from {i=1} to {N} {{α left (1- {widehat {y}} rsub {i} right )} ^ {β} { y} rsub {i } log left ({{y} rsub {i}} over {{widehat {y}} rsub {i}} right )} $
```
## Datasets

<!--
![](big_img_64x64.jpg)
-->
#### 64x64 resolution images
<img src="big_img_64x64_2_20.jpg" width=1000 class="center">

#### 16x16 resolution images
<img src="big_img_16x16_2_20.jpg" width=1000 class="center">
<!--
![](big_img_16x16.jpg)
-->




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
