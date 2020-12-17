# **Behavioral Cloning** 

## Writeup Template
('20.12/16)

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Behavior_Cloning_Architecture.jpg "Model Visualization"
[image2]: ./wooks_data/IMG/center_2020_12_14_10_11_18_450.jpg "Gathering IMG"
[image3]: ./wooks_data/IMG/center_2020_12_14_10_12_02_745.jpg "Recovery Image"
[image4]: ./wooks_data/IMG/center_2020_12_14_10_12_02_667.jpg "Recovery Image"
[image5]: ./wooks_data/IMG/center_2020_12_14_10_12_02_387.jpg "Recovery Image"
[image6]: ./wooks_data/IMG/center_2020_12_14_10_11_22_468.jpg "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[video1]: ./run1.mp4 "video file"
[image8]: ./learning_epochs.png "learning loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Using my model.py file, the architecture can be learned under keras framework

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Overall explain, 
First,
The acquired data_set (csv files and images) are splitted into train_samples and validation samples.
Each samples data was selected with shuffle and feeded into generator
Second,
Learning architecture acquired the batch dataset with shuffled, divided.
The architecture was designed as blow (model.summary())
* learning architecture
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
_________________________________________________________________
lambda_1 (Lambda)            (None, 160, 320, 3)       0          << 'for normalization'
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0          << 'image shortening for memory and performance''
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 33, 160, 24)       1824      
_________________________________________________________________
dropout_1 (Dropout)          (None, 33, 160, 24)       0         
_________________________________________________________________
activation_1 (Activation)    (None, 33, 160, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 80, 36)        21636     
_________________________________________________________________
dropout_2 (Dropout)          (None, 17, 80, 36)        0         
_________________________________________________________________
activation_2 (Activation)    (None, 17, 80, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 9, 40, 48)         43248     
_________________________________________________________________
dropout_3 (Dropout)          (None, 9, 40, 48)         0         
_________________________________________________________________
activation_3 (Activation)    (None, 9, 40, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 9, 40, 64)         27712     
_________________________________________________________________
dropout_4 (Dropout)          (None, 9, 40, 64)         0         
_________________________________________________________________
activation_4 (Activation)    (None, 9, 40, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 9, 40, 64)         36928     
_________________________________________________________________
dropout_5 (Dropout)          (None, 9, 40, 64)         0         
_________________________________________________________________
activation_5 (Activation)    (None, 9, 40, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 23040)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               2304100   
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
_________________________________________________________________
Total params: 2,441,019
Trainable params: 2,441,019
Non-trainable params: 0
_________________________________________________________________

Third,
That architecture was compiled with 'model.compile(loss='mse', optimizer='adam')
And model fit was executed using model.fit_generator.
When I tried it first, I did it one batch process, therefore I did is like this 'model.fit(X_train, Y_train, validation_split=0.2, shuffle = True, nb_epoch=3, verbose=1)'
But Finally, I used generator function like this 'model.fit_generator(train_generator, 
            steps_per_epoch=math.ceil(len(train_samples)*6/batch_size), 
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_samples)*6/batch_size), 
            epochs=8, verbose=1)'

Fourth,
The model learning loss was plotted
And Save model using 'model.save('model.h5')
Yes, 'model.h5' was the result

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

As you can see above,
My model consists of the five convolution filter layes with dropout, relu activation, and neural network included the three fully connected layer
The lambda layer was for normalization, and cropping layer was for focusing the image what we are interesting

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting 
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 28). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 135).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
I used a combination of center lane driving, recovering from the left and right sides of the road.
Also, I drived the car backward direction

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the NVIDIA Network. I thought this model might be appropriate.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model.
Then I put the dropout layer between each layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I collected more data not falling off the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 92-133) consisted of a convolution neural network with the following layers and layer sizes ...
Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would increase more variance to my dataset. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 5 as evidenced by history_object plot. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image8]

### Simulation

#### 1. Is the car able to navigate correctly on test data?

Required: No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges.
Results: My vehicle drived smoothly, but I saw that there was a little mistake such as poping up onto ledges.
What I need to improve: Gathering more good data, it would increase the better performance, in this case, I think if more recover driving data exists, better performance gets.

![alt text][video1]



