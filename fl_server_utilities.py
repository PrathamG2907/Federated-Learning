import numpy as np
import random
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
from keras import backend as K



def load_data(paths): # function to load data as floating point arrays
    data_set = []
    digits = []
    for (i, imgpath) in enumerate(paths):# load the image and extract the digit of img from imgpath
        image = np.array(cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)).flatten()
        label = imgpath.split(os.path.sep)[-2]
        image=image/255
        data_set.append(image)
        digits.append(label)
    return data_set, digits


def make_batched_users(data_set, label_list, num_users=10 , bs=64):# function to divide the data into number of clients for training
    usernames = ['user_{}'.format(i+1) for i in range(num_users)]
    data = list(zip(data_set, label_list))
    random.shuffle(data)
    size = len(data)//num_users
    data_shard_list = [data[i:i + size] for i in range(0, size*num_users, size)]
    users_batched = dict()
    for i in range(num_users):
        user_data, label = zip(*data_shard_list[i])
        batched_data= (tf.data.Dataset.from_tensor_slices((list(user_data), list(label)))).batch(bs)
        users_batched[usernames[i]] = batched_data

    return users_batched

def build_model(shape, classes): #function to build the new model
    model = Sequential()
    model.add(Dense(100, input_shape=(shape,)))
    model.add(Activation("relu"))
    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    return model


def scale_model_weights(weight, batched_user_data, username):# function to scale weigths according to data count

    usernames = list(batched_user_data.keys())
    #calculate total no of data points
    global_count = sum([tf.data.experimental.cardinality(batched_user_data[username]).numpy() for username in usernames])
    # get the total number of data points held by a user
    local_count = tf.data.experimental.cardinality(batched_user_data[username]).numpy()
    
    scale= local_count/global_count
    
    scaled_weights = []
    nums = len(weight)  # weights are a list containing arrays of values
    for i in range(nums):
        scaled_weights.append(scale* weight[i])
    return scaled_weights


def sum_scaled_weights(scaled_weight_list):# calculates the sum of scaled weigths
    final_weights = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        final_weights.append(layer_mean)
        
    return final_weights


def test_model(X_test, Y_test,  model, epoch):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('epoch: {} | acc: {:.3%} | loss: {}'.format(epoch, acc, loss))
    return acc, loss
