'''
Source: https://github.com/titu1994/Wide-Residual-Networks

Modified by: Sahil Pereira
'''

import numpy as np
import pickle
import sklearn.metrics as metrics

import wide_residual_network as wrn
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras import backend as K


# Set parameters
# ----------------------------------------------------------------------------------------
batch_size = 128
nb_epoch = 200
img_rows, img_cols = 32, 32
num_classes = 100
val_test_size = 0.0 # validation test data size


# For WRN-16-8 put N = 2, k = 8
# For WRN-28-10 put N = 4, k = 10
# For WRN-40-4 put N = 6, k = 4
blocks_N = 4
width_k = 10
dropout_prob = 0.0
learning_rate_decay = 1e-6
depth = blocks_N*6 + 4


# Output file names
# ----------------------------------------------------------------------------------------
checkpoint_file = "WRN-%s-%s_%s_v3_Weights.h5" % (depth, width_k, nb_epoch)
final_model_file = "WRN-%s-%s_%s_v3_model_final.h5" % (depth, width_k, nb_epoch)
prediction_results_file = "predictionResults_v3-%s-%s_%s.csv" % (depth, width_k, nb_epoch)


# Load data
# ----------------------------------------------------------------------------------------
train_data = pickle.load(open("train_data_py2", "rb"))
train_label = pickle.load(open("train_label_py2", "rb"))
test_data = pickle.load(open("test_data_py2", "rb"))

# Split data into training and test sets
trainX, testX, trainY, testY = train_test_split(train_data, train_label, test_size=val_test_size)

# Print training and test shapes for confirmation
print("Training set: ", trainX.shape)
print("Training labels: ", len(trainY))
print("Test set: ", testX.shape)


# Transform training and test data
# ----------------------------------------------------------------------------------------
trainX = np.array(trainX)
testX = np.array(testX)
test_data = np.array(test_data)

# reshape data into an image
trainX = np.reshape(trainX, (trainX.shape[0],3,32,32))
testX = np.reshape(testX, (testX.shape[0],3,32,32))
test_data = np.reshape(test_data, (test_data.shape[0],3,32,32))

# chagne shape of image to 32x32x3
trainX = trainX.transpose([0, 2, 3, 1])
testX = testX.transpose([0, 2, 3, 1])
test_data = test_data.transpose([0, 2, 3, 1])

# convert to type float32
trainX = trainX.astype('float32')
testX = testX.astype('float32')
test_data = test_data.astype('float32')

# rescale images to help with training
trainX /= 255
testX /= 255
test_data /= 255

# transform labels to one-hot encoding
trainY = kutils.to_categorical(trainY, num_classes)
testY = kutils.to_categorical(testY, num_classes)


# Image data generator modification
# ----------------------------------------------------------------------------------------
datagen = ImageDataGenerator(zca_whitening=True,
                               width_shift_range=4./32,
                               height_shift_range=4./32,
                               horizontal_flip=True,
                               fill_mode='reflect')

datagen.fit(trainX)

# Apply normalization (ZCA and others) to the test data
for i in range(len(testX)):
    testX[i] = datagen.standardize(testX[i])

for i in range(len(test_data)):
    test_data[i] = datagen.standardize(test_data[i])


# Create Wide ResNet model 
# ----------------------------------------------------------------------------------------

init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)
print("Image Shape: ", init_shape)

model = wrn.create_wide_residual_network(init_shape, nb_classes=num_classes, N=blocks_N, k=width_k, dropout=dropout_prob)

# print the summary of the model
model.summary()


# Compile model and fit to training data
# ----------------------------------------------------------------------------------------
lr_schedule = [60, 120, 160] # epoch_step
def scheduler(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.1
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.02
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.004
    return 0.0008

sgd = SGD(lr=0.1, momentum=0.9, decay=learning_rate_decay, nesterov=True)

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
print("Finished compiling")

#model.load_weights("weights/WRN-16-8 Weights.h5")
# print("Model loaded.")

lr_decay = callbacks.LearningRateScheduler(scheduler)

model.fit_generator(datagen.flow(trainX, trainY, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                   callbacks=[lr_decay])


# You can load a pretrained model by using this function.
model.save(final_model_file)


# Predict test data labels and save to file
# ----------------------------------------------------------------------------------------
# Predict label of test data
prd = model.predict(test_data)

# These lines are used for converting one hot coding back to the original label form.
prd_y = np.argmax(prd, axis=1)


resultFile = open(prediction_results_file, "w")
resultFile.write('ids,labels\n')

for i in range(len(prd_y)):
    resultFile.write("%s,%s\n" % (i, prd_y[i]))

resultFile.close()


