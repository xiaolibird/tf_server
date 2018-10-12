
import numpy as np
import keras.backend as K

from keras.models import Model

from keras.layers import Dense
from keras.layers.core import Activation

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from DenseNet.densenet121 import DenseNet
from skimage.transform import resize
from skimage import io, transform, filters, measure, segmentation, morphology


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
        #1
        r = img[:, :, 0]
        thresh = filters.threshold_otsu(r)
        bw =morphology.opening(r> thresh, morphology.disk(10))
        cleared = bw.copy()
        segmentation.clear_border(cleared)
        label_image =measure.label(cleared) 
        regions = measure.regionprops(label_image)
        for region in regions:
            if region.area < 100:
                continue

            minr, minc, maxr, maxc = region.bbox
			#print(minr, minc, maxr, maxc)

        img = img[int(minr*0.8):int(maxr*1.1), int(minc*0.5):int(maxc*1.1)]

        h1 = int((minc+maxc)/2)
        h2 = int((maxc-minc)/4)
        w1 = int((minr+maxr)/2)
        w2 = int((maxr-minr)/4)
        crop = r[w1-w2:w1+w2, h1-h2:h1+h2]
        crop_flatten = np.array(crop).flatten()

        ## CDF
        NumPixel = np.zeros([256])
        for i in range(len(crop_flatten)):
            NumPixel[crop_flatten[i]] = NumPixel[crop_flatten[i]]+1
        CumPixel = np.zeros([256])
        for k in range(256):
            if k == 0:
                CumPixel[k] = NumPixel[k]
            else:
                CumPixel[k] = NumPixel[k] + CumPixel[k-1]
        cmax = np.max(CumPixel)
        cmin = np.min(CumPixel)

        ### normalization
        def maxminmormalization(x,Max,Min):
            x = (x-Min)/(Max-Min)
            return x

        cumnormal = maxminmormalization(CumPixel,cmax,cmin)

		#2
		#img = img[400:1150, 350:1600, :]
        img = resize(img, self.input_shape)
        prediction = self.model.predict(img[np.newaxis, :], batch_size=1)
        result['score'] = np.max(prediction)
        result['class'] = self.class_names[np.argmax(prediction)]
        result['distribution_score'] = np.sum(cumnormal)
        return result

    def predict_a_batch(self, batch):
        pass

    def predict_a_folder(self):
        pass

