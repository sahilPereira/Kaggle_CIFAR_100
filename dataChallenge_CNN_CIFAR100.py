'''
Created on Feb 21, 2018

@author: Sahil Pereira
'''

import numpy as np
import keras
from keras.datasets import cifar100
from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras import backend as K

# Set random for recreating results:
np.random.seed(2017)

# Training parameters
batch_size = 128
epochs = 5 #200
data_augmentation = True
num_classes = 100
dropout_probability = 0.3
weight_decay = 0.0005


# Step 1: Data Config:
# --------------------------------------------------------------------------------------------------------------
image_size = 32

# Load data into test and training sets
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

print('train data shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Typecast data samples to flot32. It is usefull when using GPU.
x_train = x_train.astype('float32')   
x_test = x_test.astype('float32')

# Mean and STD preprocessing
x_train = (x_train - x_train.mean(axis=0)) / (x_train.std(axis=0))
x_test = (x_test - x_test.mean(axis=0)) / (x_test.std(axis=0))

# Scale pixles of data samples between 0 and 1.
# x_train /= 255
# x_test /= 255

# convert training and test labels to one hot encoding
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


# --------------------------------------------------------------------------------------------------------------
# Plot sample images from the dataset
# plt.figure()                                      # create new figure
# fig_size = [20, 20]                               # specify figure size
# plt.rcParams["figure.figsize"] = fig_size         # set figure size

# #Plot firs 100 train image of dataset
# for i in range(1,101):                          
#     ax = plt.subplot(10, 10, i)                   # Specify the i'th subplot of a 10*10 grid
#     img = x_train[i,:,:,:]                        # Choose i'th image from train data
#     ax.get_xaxis().set_visible(False)             # Disable plot axis.
#     ax.get_yaxis().set_visible(False)
#     plt.imshow(img)
	
# plt.show()
# --------------------------------------------------------------------------------------------------------------


# Step 2: NETWORK/TRAINING CONFIGURATION:
# --------------------------------------------------------------------------------------------------------------

depth = 16 
k = 8 # CHANGE AFTER TESTING
lr_schedule_epochs = [60, 120, 160]

# learning rate schedule
def lr_schedule(epoch):
	'''
		Initial learning rate is 0.1,
		Learning rate decay ratio is 0.2
	'''
	if (epoch + 1) < lr_schedule[0]:
		return 0.1
	elif (epoch + 1) < lr_schedule[1]:
		return 0.02
	elif (epoch + 1) < lr_schedule[2]:
		return 0.004
	return 0.0008

sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)

# Other config from code; throughtout all layer:
use_bias = False        # following functions 'FCinit(model)' and 'DisableBias(model)' in utils.lua
weight_init="he_normal" # follows the 'MSRinit(model)' function in utils.lua

# Keras specific
if K.image_dim_ordering() == "th":
	channel_axis = 1
	input_shape = (3, image_size, image_size)
else:
	channel_axis = -1
	input_shape = (image_size, image_size, 3)

# --------------------------------------------------------------------------------------------------------------


# Step 3: CREATE WRN MODEL:
# --------------------------------------------------------------------------------------------------------------
def _wide_basic(n_input_plane, n_output_plane, stride):
	def f(net):
		# format of conv_params:
		#               [ [nb_col="kernel width", nb_row="kernel height",
		#               subsample="(stride_vertical,stride_horizontal)",
		#               border_mode="same" or "valid"] ]
		# B(3,3): orignal <<basic>> block
		conv_params = [ [3,3,stride,"same"],
						[3,3,(1,1),"same"] ]
		
		n_bottleneck_plane = n_output_plane

		# Residual block
		for i, v in enumerate(conv_params):
			if i == 0:
				if n_input_plane != n_output_plane:
					net = BatchNormalization(axis=channel_axis)(net)
					net = Activation("relu")(net)
					convs = net
				else:
					convs = BatchNormalization(axis=channel_axis)(net)
					convs = Activation("relu")(convs)
				convs = Convolution2D(n_bottleneck_plane, nb_col=v[0], nb_row=v[1],
									 subsample=v[2],
									 border_mode=v[3],
									 init=weight_init,
									 W_regularizer=l2(weight_decay),
									 bias=use_bias)(convs)
			else:
				convs = BatchNormalization(axis=channel_axis)(convs)
				convs = Activation("relu")(convs)
				if dropout_probability > 0:
				   convs = Dropout(dropout_probability)(convs)
				convs = Convolution2D(n_bottleneck_plane, nb_col=v[0], nb_row=v[1],
									 subsample=v[2],
									 border_mode=v[3],
									 init=weight_init,
									 W_regularizer=l2(weight_decay),
									 bias=use_bias)(convs)

		# Shortcut Conntection: identity function or 1x1 convolutional
		#  (depends on difference between input & output shape - this
		#   corresponds to whether we are using the first block in each
		#   group; see _layer() ).
		if n_input_plane != n_output_plane:
			shortcut = Convolution2D(n_output_plane, nb_col=1, nb_row=1,
									 subsample=stride,
									 border_mode="same",
									 init=weight_init,
									 W_regularizer=l2(weight_decay),
									 bias=use_bias)(net)
		else:
			shortcut = net

		return merge([convs, shortcut], mode="sum")
	
	return f


# "Stacking Residual Units on the same stage"
def _layer(block, n_input_plane, n_output_plane, count, stride):
	def f(net):
		net = block(n_input_plane, n_output_plane, stride)(net)
		for i in range(2,int(count+1)):
			net = block(n_output_plane, n_output_plane, stride=(1,1))(net)
		return net
	
	return f


def create_model():
	logging.debug("Creating model...")
	
	assert((depth - 4) % 6 == 0)
	n = (depth - 4) / 6
	
	inputs = Input(shape=input_shape)

	n_stages=[16, 16*k, 32*k, 64*k]

	conv1 = Convolution2D(nb_filter=n_stages[0], nb_row=3, nb_col=3, 
						  subsample=(1, 1),
						  border_mode="same",
						  init=weight_init,
						  W_regularizer=l2(weight_decay),
						  bias=use_bias)(inputs) # "One conv at the beginning (spatial size: 32x32)"

	# Add wide residual blocks
	block_fn = _wide_basic
	conv2 = _layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1,1))(conv1)# "Stage 1 (spatial size: 32x32)"
	conv3 = _layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2,2))(conv2)# "Stage 2 (spatial size: 16x16)"
	conv4 = _layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2,2))(conv3)# "Stage 3 (spatial size: 8x8)"

	batch_norm = BatchNormalization(axis=channel_axis)(conv4)
	relu = Activation("relu")(batch_norm)
											
	# Classifier block
	pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), border_mode="same")(relu)
	flatten = Flatten()(pool)
	predictions = Dense(output_dim=num_classes, init=weight_init, bias=use_bias,
						W_regularizer=l2(weight_decay), activation="softmax")(flatten)

	model = Model(input=inputs, output=predictions)
	return model
# --------------------------------------------------------------------------------------------------------------



# Step 4: Create and fit model:
# --------------------------------------------------------------------------------------------------------------
model = create_model()
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])

# print summary
model.summary()


	
# Data Augmentation based on page 6 (see README for full details)
train_datagen = ImageDataGenerator(featurewise_center=True,
									featurewise_std_normalization=True,
									zca_whitening=True)
# featurewise_center=False,
# samplewise_center=False,
# featurewise_std_normalization=False,
# samplewise_std_normalization=False,
# zca_whitening=False,
# rotation_range=0,
# width_shift_range=0.1,
# height_shift_range=0.1,
# horizontal_flip=True,
# vertical_flip=False)

train_datagen.fit(x_train, augment=True, rounds=2)

test_datagen = ImageDataGenerator(
				 featurewise_center=True,
				 featurewise_std_normalization=True,
				 zca_whitening=True)
test_datagen.fit(x_train)

callbacks = [LearningRateScheduler(schedule=lr_schedule)]

# fit the model on the batches generated by train_datagen.flow()
model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
					samples_per_epoch=x_train.shape[0],
					nb_epoch=epochs,
					validation_data=test_datagen.flow(x_test, y_test, batch_size=batch_size),
					nb_val_samples=x_test.shape[0])
					# callbacks=callbacks)
# --------------------------------------------------------------------------------------------------------------


# Step 5: Save and test model:
# --------------------------------------------------------------------------------------------------------------

# Save model
model.save('./w_16_8_test.h5') # This function saves the model in 'cnn.h5' file

# You can load a pretrained model by using this function.
# model = keras.models.load_model('./cnn.h5')

# trained model predicts the label of its input data
prd = model.predict(x_test)

# These lines are used for converting one hot coding back to the original label form.
prd_y = np.argmax(prd, axis=1)
y_test_orig = np.argmax(y_test, axis=1)

# print predicted and true lables of first 10 test samples
print('Predicted Labels:\t', prd_y[0:10])
print('True labels:\t\t', y_test_orig[0:10])

# Check accuracy:
nb_correct_labels = np.sum(prd_y == y_test_orig)
print('Test accuracy is: ', nb_correct_labels/len(y_test))