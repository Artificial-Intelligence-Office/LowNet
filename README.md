# LowNet

Implementation of our ICIP 2020 paper "Lownet: privacy preserved Ultra-Low Resolution Posture Image Classification"

In this project, we created LowNet architecture, which is suitable for low resolution image classification. 
We are releasing TIP38(Thermal Image Posture 38 class) yoga posture image dataset.

![alt text](https://github.com/[MoyoG]/[LowNet]/images/ID_102_pose_01_0_0.jpg?raw=true)



Custom loss : 
 $ L= sum from {i=1} to {N} {{α left (1- {widehat {y}} rsub {i} right )} ^ {β} { y} rsub {i } log left ({{y} rsub {i}} over {{widehat {y}} rsub {i}} right )}
