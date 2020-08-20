import pandas
import numpy as np
from keras.preprocessing import image
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import Adam
from keras.applications import ResNet50
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Input
from keras.layers import *
from keras.optimizers import Adam
from keras import backend as K
from keras import regularizers
from keras.applications import MobileNet
from keras.applications import ResNet50
import scipy.misc
# import model
import cv2
from subprocess import call
import os

# print(os.listdir("./driving_dataset/"))

model = Sequential()

model.add(Conv2D(24,(5,5),activation='relu',input_shape=(128,128,3),kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(36,(5,5),activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(48,(3,3),activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Flatten())


model.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))

model.add(Dense(32,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))

model.add(Dense(10,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))

model.add(Dense(1,activation='tanh',kernel_regularizer=regularizers.l2(0.01)))

model.summary()

model.compile(optimizer = Adam(0.0001),loss = 'mse',metrics=['mae'])
model.load_weights('Model_with_less_parameters.h5')

img_str = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img_str.shape

smoothed_angle = 0


i = 36000
while(cv2.waitKey(10) != ord('q') and i<45567):
	img1 = image.load_img("dataset/Train_data/" + str(i) + ".jpg",color_mode='rgb')
	img1 = image.image.img_to_array(img1)/255.0
	img2 = image.load_img("dataset/Train_data/" + str(i) + ".jpg",color_mode='rgb')
	#print(img2.shape)
	img2 = image.image.img_to_array(img2)/255.0
	img2 = img2[76:,:,:]
	img2 = cv2.resize(img2,(128,128))
    #img2 = img2/255.0
	#img = image.load_img("Self_Driving_Car_Project/dataset/Train_data/" + str(i) + ".jpg",color_mode='rgb',target_size=[200,200,3])
	#i#mg = image.img_to_array(img)/255.0
	img_resh = np.reshape(img2,[1,128,128,3])
	degrees = model.predict(img_resh) * 180.0 / scipy.pi
	print("Predicted steering angle: " + str(degrees) + " degrees")
	cv2.imshow("frame", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
	smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
	M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
	dst = cv2.warpAffine(img_str,M,(cols,rows))
	cv2.imshow("steering wheel", dst)
	i += 1

cv2.destroyAllWindows()
