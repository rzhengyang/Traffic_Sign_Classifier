# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 27839
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data range.

![ax](https://github.com/rzhengyang/Traffic_Sign_Classifier/blob/master/ax.png1)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it can reduce the impact of colour of this test.And this can reduce the pic deepth from 3 to 1.

Here is an example of a traffic sign image before and after grayscaling.

![gray](https://github.com/rzhengyang/Traffic_Sign_Classifier/blob/master/gray.png)


As a last step, I normalized the image data because we can use the same stander to produce all pics.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                |    Description                                | 
|:---------------------:|:---------------------------------------------:| 
| Input                | 32x32x1 Gray image                              | 
| Convolution 5x5         | 1x1 stride, valid padding, outputs 28x28x6     |
| RELU                    |     leaky_relu       |
| Max pooling              | 2x2 stride,  outputs 14x14x6                 |
| DROPOUT           |    |
| Convolution 5x5        |1x1 stride, valid padding, outputs 10x10x16      |
| RELU                    |   leaky_relu          |
| Max pooling              | 2x2 stride,  outputs 5x5x16                 |
| DROPOUT           |    |
| Fully connected        | inputs 400, outputs 120    |
| RELU                    |     leaky_relu       |
| DROPOUT           |    |
| Fully connected        |  inputs 120, outputs 84      |
| RELU                    |     leaky_relu       |
| DROPOUT           |    |
| Fully connected        |  inputs 84, outputs 43   |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an sess in tensorflow.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.7%

* validation set accuracy of 95.4%
* test set accuracy of 93.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen ?  leNet
* What were some problems with the initial architecture?  accuracy is below 90% at validation test
* How was the architecture adjusted and why was it adjusted?Add dropout, add L2loss way.

* Which parameters were tuned? How were they adjusted and why? EPOCHS = 100     BATCH_SIZE = 120    keep_prob = 0.5



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![1](https://github.com/rzhengyang/Traffic_Sign_Classifier/blob/master/test%20pic%20out/1.jpg)
![2](https://github.com/rzhengyang/Traffic_Sign_Classifier/blob/master/test%20pic%20out/2.jpg) 
![3](https://github.com/rzhengyang/Traffic_Sign_Classifier/blob/master/test%20pic%20out/3.jpg) 
![4](https://github.com/rzhengyang/Traffic_Sign_Classifier/blob/master/test%20pic%20out/4.jpg) 
![5](https://github.com/rzhengyang/Traffic_Sign_Classifier/blob/master/test%20pic%20out/5.jpg) 
![6](https://github.com/rzhengyang/Traffic_Sign_Classifier/blob/master/test%20pic%20out/6.jpg) 
![7](https://github.com/rzhengyang/Traffic_Sign_Classifier/blob/master/test%20pic%20out/7.jpg)  
![8](https://github.com/rzhengyang/Traffic_Sign_Classifier/blob/master/test%20pic%20out/8.jpg) 


The 4-8 images difficult to classify because its have different enviorment signal around.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                    |    Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Go straight or left              | Go straight or left                                       | 
| Keep right             | Keep right                                         |
| Speed limit (30km/h)                   | Stop                                          |
| Go straight or right          | Go straight or right                        |
| Speed limit (70km/h)      | Speed limit (30km/h)                       |
| Speed limit (120km/h)   | End of speed limit (80km/h)   |
| No entry                        | No entry                 |
| Road work                    | End of no passing   |


The model was able to correctly guess 4 of the 8 traffic signs, which gives an accuracy of 50%. This compares favorably to the accuracy on the test set of lower

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![result][https://github.com/rzhengyang/Traffic_Sign_Classifier/blob/master/result.jpg] 




