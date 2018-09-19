
import numpy as np
import keras.backend as K

from keras.models import Model

from keras.layers import Dense
from keras.layers.core import Activation

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from DenseNet.densenet121 import DenseNet
from skimage.transform import resize


# config = tf.ConfigProto()
# sess = tf.Session(config=config)
# KTF.set_session(sess)


# WEIGHT_DIR = '/Users/xiaolibird/Desktop/机器视觉/DenseNet/output0627_best_weights.h5'
# BASE_WEIGHT_DIR = '/Users/xiaolibird/Desktop/机器视觉/DenseNet/imagenet_models/densenet121_weights_tf.h5'

# WEIGHT_DIR = '/Users/xiaolibird/Desktop/simple-keras-rest-api/DenseNet/imagenet_models/output0627_best_weights.h5'
# BASE_WEIGHT_DIR = '/Users/xiaolibird/Desktop/simple-keras-rest-api/DenseNet/imagenet_models/densenet121_weights_tf.h5'

# Windows DIR
WEIGHT_DIR = ".\\DenseNet\\imagenet_models\\output0627_best_weights.h5"
BASE_WEIGHT_DIR = ".\\DenseNet\\imagenet_models\\densenet121_weights_tf.h5"

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
        img = resize(img, self.input_shape)

        prediction = self.model.predict(img[np.newaxis, :], batch_size=1)
        result['score'] = np.max(prediction)
        result['class'] = self.class_names[np.argmax(prediction)]

        return result

    def predict_a_batch(self, batch):
        pass

    def predict_a_folder(self):
        pass

