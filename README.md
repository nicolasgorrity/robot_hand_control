# robot_hand_control

## Description
This project aims to control a Khepera IV robot thanks to hand gesture detected from a camera.
The camera is on a control computer that is connected to the robot to make it execute the desired commands.
All the image acquisition chain, command prediction and command control are implemented in ROS packages.

This project was tested on Ubuntu 18.04 with ROS melodic.

## Installation
Before installing this project, make sure [ROS melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) is correctly installed on your machine.

```
git clone https://github.com/nicolasgorrity/robot_control.git
cd robot_control/ros_ws
source /opt/ros/melodic/setup.bash
catkin build
```
Then source ROS to the current workspace
```
source devel/setup.bash
```

## Use
In a terminal, run
```
roscore
```
Before running any node make sure ROS is correctly sourced to this workspace. In another terminal:
```
source PATH/robot_control/ros_ws/devel/setup.bash
```
- If you want to classify gestures with KNN:
```
roslaunch khepera_control khepera_knn.launch
```
- If you want to classify gestures with the Multi-Layer Perceptron, you first have to train it:
```
rosrun khepera_classification khepera_classification_mlp_train_node
```
Then you can type
```
roslaunch khepera_control khepera_mlp.launch
```

The terminal will output the result of classification in real time:
![Classification output in console](.example_images/console.png?raw=true "Classification output in console")

A window of `rqt_image_view` will pop up. From here you can see:
- the raw camera image with topic `/khepera/image`
![Camera image](.example_images/camera.png?raw=true "Camera image")

- the extracted hand with topic `/khepera/hand_detect`
![Extracted hand](.example_images/extracted_hand.png?raw=true "Extracted hand")

- the resulting contour with topic `/khepera/contours_img`
![Hand contour](.example_images/hand_contour.png?raw=true "Hand contour")

A `rqt_reconfigure` window will also appear. This allows you to dynamically edit parameters of the image processing package `khepera_image` in real time. You can adjust the `threshold` for hand detection, enable an experimental background substractor or switch from the camera video input to some images of the dataset:
![Dynamic configuration](.example_images/cfg.png?raw=true "Dynamic configuration")
