# Self_Driving_Car_Project
The problem seems to be quite complex, and that is truth as well. This project shows solution just to a part of this problem. We will be feeding sequences of images from dashcam and this model will predict the steering angle in degrees.We will only to optimize the model using CNN only.
# Dataset_Link
The link to the dataset is [here](https://drive.google.com/file/d/1PgjoLe1mm_gLHKzThmksaqH-RpGeD12m/view?usp=sharing) The datset contains around 45k images and corresponding steering angles. Its around 2.23 GB and you will be able to train a small model on your pc itself.
# Model insights
The steering angle was in degrees and it was converted into radians first.(Multiplying by np.pi/180)

The model was designed with basic CNN, Flatten, Dense , Batch Normalization and Dropout Layers.

The activation used in inner layers was relu

The activation used in output layer was tanh. (I tried for Linear Activation as well but tanh produced someway better results)

Dropout layers were added for regularization and to prevent overfitting.

The model predicted the value of steering angle in radians so later it was converted back to degrees.

The val_loss at last comes out to be 0.1597 and it was able to correctly predict the steering angles.

# Loss function
For validation_data Mean Square Error (MSE) was used and in metrics Mean Absolute Error(MAE) is used for analyzing the performance of the model after every epoch.
Value of l2_norm_constant 0.001 that is used while training the Model

# Training
I trained my model on google colab.It took me around 9-10 hours of training while I was trying for different hyper_parameters and I have trained on 2 different Model Architectures that i think was most suitable for the problem.

Validation data was 20% of the total Training data that i used for analyzing the performance of the Model

# Testing

run.py and run2.py containse code for testing the model and its performance. This output video is on testing data (not on training data)

# Sample_Video
![alt-text](https://github.com/2000aman/Self_Driving_Car_Project/blob/master/Self_Driving_Car_Video.gif)

# Reference
 Research paper: End to End Learning for Self-Driving Cars by Nvidia. [https://arxiv.org/pdf/1604.07316.pdf]
