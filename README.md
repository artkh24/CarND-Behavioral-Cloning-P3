# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/left_image.jpg "Left image"
[image2]: ./examples/center_image.jpg "Center image"
[image3]: ./examples/right_image.jpg "Right image"

![alt text][image1]
![alt text][image2]
![alt text][image3]
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode(Provided by Udacity)
* model.h5 containing a trained convolution neural network 
* Behavioral Cloning.md summarizing the results
* video.mp4 Video of vehicle autonomous driving

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I used nVidia model and added Dropout layers to help model generalize better and the car shows good driving skills after five epochs of model training

Below is the nVidia model

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   					    | 
| Normalization         | Normalizing around zero  x/255-0.5            |
| Cropping              | outputs 70x320x3                              |
| Convolution 5x5     	| 2x2 stride, activation Relu, outputs 33x158x24|             	
| Convolution 5x5     	| 2x2 stride, outputs 15x77x36                  |
| Dropout			    |												|
| RELU			        | keep_prob=0.5									|
| Convolution 5x5     	| 2x2 stride, activation Relu, outputs 6x37x48  |
| Convolution 3x3     	| 1x1 stride, activation Relu, outputs 4x35x64  |
| Convolution 3x3     	| 1x1 stride, activation Relu, outputs 2x33x64  |
| Dropout			    |												|
| RELU			        | keep_prob=0.5									|
| Flattening			|flattens array, outputs 4224					|
| Fully connected		| outputs 100        							|
| Fully connected		| outputs 50        							|
| Fully connected		| outputs 10        							|
| Fully connected		| outputs 1        						    	|

#### 2. Attempts to reduce overfitting in the model

First I tried the nVidia model without modification.  I split my data to training 80% and validation 20% set and I train this model for 3 epochs, but it shows low training losses and high and not decreasing validation losses its mean that the model overfits, after that I added one Dropout layer before Flattening and run it for 4 epochs but shill the model overfits,then I added one more Dropout layer and the model started generalize well
The model was trained and validated on different data sets using generator
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 99).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a center left and right images to train the model

For details about how I created the training data, see the next section. 


#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the nVidia model but with adding 2 Dropout layers
I normalize the images then crop them to use only car driving zone, Then I augment the data by adding the same image flipped with a negative angle
For the left and right images their angles were corrected by correction factor
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 3. Creation of the Training Set & Training Process

I generated data using simulator
I have one track driving forward one driving backward and also additional data from bridge to end of track where 
we have 2 big turns

With all these data I trained model with five epochs. Below you can find the loss diagram

[image4]: ./examples/Loss.jpg "model mean squared error loss"
![alt text][image4]
