
import numpy as np
import keras.backend as K

from keras.models import Model

from keras.layers import Dense
from keras.layers.core import Activation

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from densenet121 import DenseNet
from skimage.transform import resize
from skimage import filters
import skimage.morphology as sm
import cv2 

# config = tf.ConfigProto()
# sess = tf.Session(config=config)
# KTF.set_session(sess)


# WEIGHT_DIR = '/Users/xiaolibird/Desktop/机器视觉/DenseNet/output0627_best_weights.h5'
# BASE_WEIGHT_DIR = '/Users/xiaolibird/Desktop/机器视觉/DenseNet/imagenet_models/densenet121_weights_tf.h5'

# WEIGHT_DIR = '/Users/xiaolibird/Desktop/simple-keras-rest-api/DenseNet/imagenet_models/output0627_best_weights.h5'
# BASE_WEIGHT_DIR = '/Users/xiaolibird/Desktop/simple-keras-rest-api/DenseNet/imagenet_models/densenet121_weights_tf.h5'

# Windows DIR
BASE_WEIGHT_DIR = 'imagenet_models/densenet121_weights_tf.h5'
WEIGHT_DIR = '../output0627_best_weights.h5'

class MyDenseNet(object):
    """
    A class for load network parameters and generate prediction
    """

    def __init__(self):
        self.img_width = 224
        self.img_height = 224
        if K.image_data_format() == 'channels_first':
            self.input_shape = (3, self.img_width, self.img_height)
        else:
            self.input_shape = (self.img_width, self.img_height, 3)
        self.model = None
        # label_names
        self.class_names = ['All_of_Negative', 'Aspergillus', 'Bacillus', 'Clostridium', 'Pseudomonas', 'Staphylococcus']

    def load_model(self):
        """
        load pre-trained layers and fine-tuned layers
        :return: model
        """
        print("=============start loading models=============")
        # load models from basemodel and fine-tune layers
        base_model = DenseNet(reduction=0.5, classes=1000, weights_path=BASE_WEIGHT_DIR)
        base_model.layers.pop()
        base_model.layers.pop()
        x4 = Dense(6, activation='relu')(base_model.layers[-1].output)
        o = Activation('softmax')(x4)

        model = Model(inputs=base_model.input, outputs=[o])
        model.load_weights(WEIGHT_DIR)

        self.model = model
        print("=============finish loading models=============")

    def predict_one_image(self, img):
		print("=============start predicting=============")
		result = {}
		
		#1
		r = img[:, :, 0]
		thresh = filters.threshold_otsu(r)
		ret, thresh_img = cv2.threshold(r, thresh, 255, cv2.THRESH_BINARY)
		mor_img = sm.opening(thresh_img, sm.disk(15))
		contours = cv2.findContours(mor_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		cnt = contours[0]
		x, y, w, h = cv2.boundingRect(cnt)
		img = img[y-100:y+h+100, x-300:x+w+50, :]
		
		#2
		#img = img[400:1150, 350:1600, :]
		
		img = resize(img, self.input_shape)
		prediction = self.model.predict(img[np.newaxis, :], batch_size=1)
		result['score'] = np.max(prediction)
		result['class'] = self.class_names[np.argmax(prediction)]
		
		return result

    def predict_a_batch(self, batch):
        pass

    def predict_a_folder(self):
        pass

