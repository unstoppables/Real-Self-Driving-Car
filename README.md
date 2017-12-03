This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

### Team Members
- Venugopala Krishna Bhat (waypoint logic, integration)
- Satish Avhad (control of acceleration and deceleration, integration)
- Gaylord Marville (traffic light detection and classifcation, integration)
- Torben Fischer (traffic light classification, integration)

### Approach

* Traffic Light Detection and Classification

To classify the traffic lights we followed two different approaches. 

The first one included two separate models: a retrained model based on Faster-R-CNN for the traffic light detection, which is using the Tensorflow Object Detection API and a CNN classifier based on LeNet, which classifies the cropped image of the light into green, yellow and red. The input size of the images are 32x32x3. 

The Training data consisted of real life images, taken from the Bosch Dataset. For the classifier, the annotated boxes in the Bosch dataset were using to crop the image to the light. The dataset was then extended by images taken from the simulator. These images had to be labeled manually by drawing boxes with Sloth (https://github.com/cvhciKIT/sloth).

The second approach uses a CNN classifier to classify the camera raw images into red traffic light or no red traffic light. This approach works alot faster, but might be of disadvantage in real life testing. It is inspired and based on a CNN architecture found at another team: https://github.com/ksmith6/sdc-capstone 

The dataset for this classifier was also based on the Bosch dataset and simulator images. The input size of the images are 400x400x3. 


Layer (type)                     Output Shape          Param #     Connected to                     
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 398, 398, 32)  896         convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 396, 396, 32)  9248        convolution2d_1[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 198, 198, 32)  0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 196, 196, 32)  9248        maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 98, 98, 32)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 96, 96, 32)    9248        maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 48, 48, 32)    0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 46, 46, 32)    9248        maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 23, 23, 32)    0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 21, 21, 32)    9248        maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
maxpooling2d_5 (MaxPooling2D)    (None, 10, 10, 32)    0           convolution2d_6[0][0]            
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D)  (None, 8, 8, 32)      9248        maxpooling2d_5[0][0]             
____________________________________________________________________________________________________
maxpooling2d_6 (MaxPooling2D)    (None, 4, 4, 32)      0           convolution2d_7[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 4, 4, 32)      0           maxpooling2d_6[0][0]             
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 512)           0           dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 512)           0           flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 128)           65664       dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 128)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 2)             258         dropout_3[0][0]                  
____________________________________________________________________________________________________
Total params: 122,306
Trainable params: 122,306
Non-trainable params: 0

* Waypoint Logic


To find the closest path waypoint to the upcoming traffic light, we divide the problem in 3 parts:

1) find the closest path waypoint to our current vehicule position (car_position)
2) find the closest red light's stop line to "car_position" (closest_tl)
3) find the closest path waypoint to "closest_tl"

How to solve the complexity problem of calculating distances over large
set of points in a reasonable amount of time. Here we used the so called "Kd Tree algorithm"
(k-dimensional tree) which is one solution to the Nearest neighbor search problem (NNS)
which is the optimization problem of finding the point in a given set that is closest
(or most similar) to a given point https://en.wikipedia.org/wiki/Nearest_neighbor_search
note: this algorithm work for K dimension but we are using it for 2 here.
This method implies to first build a tree of our set of points using recursivity,
and then search in that tree the closest point to our target point.
Complexity:
Given a set S of points in a space M, the naive method has a running time of O(dN), where N is the cardinality of S and d is the dimensionality of M.The K-d Tree method is O(log n) complex.

A good youtube video about K-d trees https://youtu.be/u4M5rRYwRHs



### Installion Instructions (from Udacity)

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases/tag/v1.2).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing) that was recorded on the Udacity self-driving car (a bag demonstraing the correct predictions in autonomous mode can be found [here](https://drive.google.com/open?id=0B2_h37bMVw3iT0ZEdlF4N01QbHc))
2. Unzip the file
```bash
unzip traffic_light_bag_files.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_files/loop_with_traffic_light.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
