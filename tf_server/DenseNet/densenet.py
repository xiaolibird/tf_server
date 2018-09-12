import itertools
from collections import Counter
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from keras.models import Model
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint,TensorBoard

from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import  Activation
from keras.layers.pooling import GlobalAveragePooling2D


from keras import backend as K
#K.set_image_dim_ordering('tf')

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# set the first GPU avaliable 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   #distributed according to the needs.
sess = tf.Session(config=config)
KTF.set_session(sess)


import cv2
import numpy as np
from keras.optimizers import SGD
import keras.backend as K

#create a folder for the results
import os, sys
result_dir = 'output0627'
# print ("\nCreate directory for the results (if not already existing)")
# if os.path.exists(result_dir):
    # print ("Dir already existing")
# elif sys.platform=='win32':
    # os.system('mkdir ' + result_dir)
# else:
    # os.system('mkdir -p ' +result_dir)

# We only test DenseNet-121 in this script for demo purpose
from densenet121 import DenseNet 


from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

from shutil import copyfile

import time
import datetime

def get_class_weights(y):
		counter = Counter(y)
		majority = max(counter.values())
		return  {cls: float(majority/count) for cls, count in counter.items()}

def plot_confusion_matrix(cm, classes,
													normalize=False,
													title='Confusion matrix',
													cmap=plt.cm.Blues):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		if normalize:
				cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
				print("Normalized confusion matrix")
		else:
				print('Confusion matrix, without normalization')

		print(cm)

		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
				plt.text(j, i, format(cm[i, j], fmt),
									horizontalalignment="center",
									color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')



def pop_layer(model):
		if not model.outputs:
				raise Exception('Sequential model cannot be popped: model is empty.')
		model.layers.pop()
		if not model.layers:
				model.outputs = []
				model.inbound_nodes = []
				model.outbound_nodes = []
		else:
				model.layers[-1].outbound_nodes = []
				model.outputs = [model.layers[-1].output]
		model.built = False
    
    
if K.image_dim_ordering() == 'th':
	# Use pre-trained weights for Theano backend
	weights_path = 'imagenet_models/densenet121_weights_th.h5'	
else:
	# Use pre-trained weights for Tensorflow backend
	weights_path = 'imagenet_models/densenet121_weights_tf.h5'


# dimensions of our images.
img_width, img_height = 224, 224


if K.image_data_format() == 'channels_first':
		input_shape = (3, img_width, img_height)
else:
		input_shape = (img_width, img_height, 3)


strategymode=2


fold1train_data_dir = '/home/zju/wujunya/colony/data/train'
fold1validation_data_dir ='/home/zju/wujunya/colony/data/test'
fold1test_data_dir = '/home/zju/wujunya/colony/data/test'

num_epochs = 100
epoch_perrun = 100
batch_size = 16


# Test pretrained model


# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
	rescale=1. / 255,
	shear_range=0.1,
	zoom_range=0.1,
	horizontal_flip=True,
	vertical_flip=False,
	width_shift_range=0.2,
	height_shift_range=0.2,
	rotation_range=0.2,
	fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)


fold1train_generator = train_datagen.flow_from_directory(
	fold1train_data_dir,
	target_size=(img_height, img_width),
	batch_size=batch_size,
	class_mode='categorical',
	shuffle=True)

fold1validation_generator = test_datagen.flow_from_directory(
	fold1validation_data_dir,
	target_size=(img_height, img_width),
	batch_size=batch_size,
	class_mode='categorical',
	shuffle=False)


	
fold1test_generator = test_datagen.flow_from_directory(
	fold1test_data_dir,
	target_size=(img_height, img_width),
	batch_size=batch_size,
	class_mode='categorical',
	shuffle=False)

class_names=['AllofNegative', 'Aspergillus', 'Bacillus', 'Clostridium', 'Pseudomonas', 'Staphylococcus']

# results for fold1
if (strategymode==1) :
	basemodel = DenseNet(reduction=0.5, classes=6, weights_path='my_model_densenet_6_good.h5')
elif(strategymode==2)  :
	basemodel = DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)
else	:
  basemodel = DenseNet(reduction=0.5, classes=1000)
  
#basemodel.summary()
basemodel.layers.pop()
basemodel.layers.pop()
x4 = Dense(6, activation='relu')(basemodel.layers[-1].output)
o =  Activation('softmax')(x4)
model=Model(inputs= basemodel.input, outputs=[o])
#model.summary()

nb_layers = len(basemodel.layers)
print("number of basemodel.layers:",nb_layers)


model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])


nb_train_samples = fold1train_generator.samples
nb_validation_samples = fold1validation_generator.samples
nb_test_samples = fold1test_generator.samples

print("nb_train_samples:",nb_train_samples)
print("nb_validation_samples:",nb_validation_samples)


train_labels=fold1train_generator.classes
train_classes = np.unique(train_labels)
cw = compute_class_weight("balanced", train_classes, train_labels)
cw2=get_class_weights(train_labels)


# fine-tune the model
modelfinal=model
print("number of modelfinal.layers:",len(modelfinal.layers))

tensorboard = TensorBoard(log_dir="../logdir/"+result_dir)
checkpoint = ModelCheckpoint(filepath='../'+result_dir+'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True)
callback_lists = [tensorboard,checkpoint]

history = model.fit_generator(
		fold1train_generator,
		steps_per_epoch=nb_train_samples/batch_size,#samples_per_epoch->steps_per_epoch
		epochs=num_epochs,
		validation_data=fold1validation_generator,
		validation_steps=nb_validation_samples/batch_size,#nb_val_samples->validation_steps
		class_weight=cw2,
		callbacks=callback_lists)

score = model.evaluate_generator(fold1validation_generator,steps=nb_validation_samples/batch_size)
	
filename ="../history.txt"
file = open(filename, "w")
file.write(str(history.history['loss'])+"\n")#"loss:"+
file.write(str(history.history['acc'])+"\n")#"acc:"+
file.write(str(history.history['val_loss'])+"\n")#"val_loss:"+
file.write(str(history.history['val_acc'])+"\n")#"val_acc:"+
file.close()

print("model.metrics_names:", model.metrics_names)
print("score:", score) 

y_test1=fold1test_generator.classes
prediction1 = modelfinal.predict_generator(fold1test_generator,steps=nb_test_samples/batch_size)
np.savetxt("../prediction1.txt", prediction1)

y_pred1=np.argmax(prediction1, axis=1)
np.savetxt("../y_pred1.txt", y_pred1)
np.savetxt("../y_test1.txt", y_test1)

