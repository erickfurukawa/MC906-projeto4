from __future__ import absolute_import, division, print_function
import os
import cv2
import random

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def draw_confusion_matrix(true,preds):
    conf_matrix = confusion_matrix(true,preds)
    sns.heatmap(conf_matrix, annot=True,annot_kws={"size":12},fmt='g', cbar=False,cmap="viridis")
    #plt.xlabel
    plt.xticks([0.5,1.5,2.5,3.5],['akatsuki','hibiki','ikazuchi','inazuma'],rotation='horizontal')
    plt.yticks([0.5,1.5,2.5,3.5],['akatsuki','hibiki','ikazuchi','inazuma'],rotation='horizontal')
    plt.show()


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(4),['akatsuki','hibiki','ikazuchi','inazuma'],rotation='horizontal')
    plt.yticks([])
    thisplot = plt.bar(range(4), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

#get training data
dataQtd= 4*964
imgData = []
i = 0
# akatsuki
imgFolder = "./cropper/akatsuki"
fileList = os.listdir(imgFolder)
for filePath in fileList:
    if i >=964:
        break;
    i+=1
    image = cv2.imread(imgFolder+"/"+filePath,cv2.IMREAD_COLOR)
    grayImage = cv2.imread(imgFolder+"/"+filePath,cv2.IMREAD_GRAYSCALE)
    imgData.append((image,grayImage,0))

i = 0
# hibiki
imgFolder = "./cropper/hibiki"
fileList = os.listdir(imgFolder)
for filePath in fileList:
    if i >=964:
        break;
    i+=1
    image = cv2.imread(imgFolder+"/"+filePath,cv2.IMREAD_COLOR)
    grayImage = cv2.imread(imgFolder+"/"+filePath,cv2.IMREAD_GRAYSCALE)
    imgData.append((image,grayImage,1))

i = 0
# ikazuchi
imgFolder = "./cropper/ikazuchi"
fileList = os.listdir(imgFolder)
for filePath in fileList:
    if i >=964:
        break;
    i+=1
    image = cv2.imread(imgFolder+"/"+filePath,cv2.IMREAD_COLOR)
    grayImage = cv2.imread(imgFolder+"/"+filePath,cv2.IMREAD_GRAYSCALE)
    imgData.append((image,grayImage,2))

i = 0
# inazuma
imgFolder = "./cropper/inazuma"
fileList = os.listdir(imgFolder)
for filePath in fileList:
    if i >=964:
        break;
    i+=1
    image = cv2.imread(imgFolder+"/"+filePath,cv2.IMREAD_COLOR)
    grayImage = cv2.imread(imgFolder+"/"+filePath,cv2.IMREAD_GRAYSCALE)
    imgData.append((image,grayImage,3))

#----------------------------------------------------------------
#get testing data
testDataQtd= 0
testImgData = []
# akatsuki
imgFolder = "./cropper/akatsuki_test"
fileList = os.listdir(imgFolder)
testDataQtd += len(fileList)
for filePath in fileList:
    image = cv2.imread(imgFolder+"/"+filePath,cv2.IMREAD_COLOR)
    grayImage = cv2.imread(imgFolder+"/"+filePath,cv2.IMREAD_GRAYSCALE)
    testImgData.append((image,grayImage,0))

# hibiki
imgFolder = "./cropper/hibiki_test"
fileList = os.listdir(imgFolder)
testDataQtd += len(fileList)
for filePath in fileList:
    image = cv2.imread(imgFolder+"/"+filePath,cv2.IMREAD_COLOR)
    grayImage = cv2.imread(imgFolder+"/"+filePath,cv2.IMREAD_GRAYSCALE)
    testImgData.append((image,grayImage,1))

# ikazuchi
imgFolder = "./cropper/ikazuchi_test"
fileList = os.listdir(imgFolder)
testDataQtd += len(fileList)
for filePath in fileList:
    image = cv2.imread(imgFolder+"/"+filePath,cv2.IMREAD_COLOR)
    grayImage = cv2.imread(imgFolder+"/"+filePath,cv2.IMREAD_GRAYSCALE)
    testImgData.append((image,grayImage,2))

# inazuma
imgFolder = "./cropper/inazuma_test"
fileList = os.listdir(imgFolder)
testDataQtd += len(fileList)
for filePath in fileList:
    image = cv2.imread(imgFolder+"/"+filePath,cv2.IMREAD_COLOR)
    grayImage = cv2.imread(imgFolder+"/"+filePath,cv2.IMREAD_GRAYSCALE)
    testImgData.append((image,grayImage,3))
#----------------------------------------------------------------


#shuffle the data, and separate images from labels
random.shuffle(imgData)
train_images = np.empty((dataQtd,64,64,3),np.float32)
train_grayImages = np.empty((dataQtd,64,64),np.float32)
train_labels = np.empty(dataQtd, np.int32)
for i in range(dataQtd):
    (image,grayImage,label) = imgData[i]
    train_images[i] = image
    train_grayImages[i] = grayImage
    train_labels[i] = int(label)


#shuffle the test data, and separate images from labels
random.shuffle(testImgData)
test_images = np.empty((testDataQtd,64,64,3),np.float32)
test_grayImages = np.empty((testDataQtd,64,64),np.float32)
test_labels = np.empty(testDataQtd, np.int32)
for i in range(testDataQtd):
    (image,grayImage,label) = testImgData[i]
    test_images[i] = image
    test_grayImages[i] = grayImage
    test_labels[i] = int(label)


class_names = ['akatsuki', 'hibiki','ikazuchi', 'inazuma']
train_images /= 255 # normalizing the pixels
train_grayImages /= 255
test_images /= 255
test_grayImages /= 255

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(cv2.cvtColor(train_images[i],cv2.COLOR_BGR2RGB))
    plt.xlabel(class_names[train_labels[i]])
plt.show()

train_grayImages = train_images.reshape(dataQtd,64,64,3)
test_grayImages = test_images.reshape(testDataQtd,64,64,3)

model = keras.Sequential([
    keras.layers.Conv2D(32,kernel_size=3, activation='relu',input_shape=(64,64,3)),
    keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'),
    keras.layers.Conv2D(64,kernel_size=3, activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(4, activation=tf.nn.softmax)
])


#compiling
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#training
model.fit(train_grayImages, train_labels, epochs=10, validation_split=0.1)

#testing
test_loss, test_acc = model.evaluate(test_grayImages, test_labels)
print('Test accuracy:', test_acc)

#predictions
predictions = model.predict(test_grayImages)

pred_labels = []
for i in range(len(predictions)):
    predicted_label = np.argmax(predictions[i])
    pred_labels.append(predicted_label)

draw_confusion_matrix(test_labels,pred_labels)
"""
for i in range(20):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions,  test_labels)
    plt.show()"""
