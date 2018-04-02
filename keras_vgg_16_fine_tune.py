from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import keras

import numpy as np
import cv2
import os

# select the gpu

os.environ["CUDA_VISIBLE_DEVICES"]="0"


sgd_learning_rate = 0.0001
sgd_batch_size = 64
sgd_epoch = 10

# Data load from here

image_path = '/datahdd/workdir/donghyun/faster_rcnn_kdh/PascalDataSetCroppedEdited/' # CroppedEdited : number of data non deleted

filenumber = 0
X_train = list()
Y_train = list()

while(1):
    path = image_path + 'pascal_voc_'+str(filenumber)


    if os.path.isfile(path+'.jpg') is True & os.path.isfile(path+'.txt') is True:
        X_image = cv2.imread(path+'.jpg')
        Y_label = np.loadtxt(path+'.txt', delimiter = ' ')
        X_train.append(X_image)
        Y_train.append(Y_label)
        #print(str(filenumber) + ' is loaded')


    else:
        print('image loading stopped at ' + str(filenumber-1))
        break

    filenumber += 1

#######################################
#            SEPARATE DATA            #
#######################################

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# shuffling all of the data set and separate train & val set
shuffled_indexes = np.arange(len(X_train))
np.random.shuffle(shuffled_indexes)

shuffle_indexes = shuffled_indexes[0:int(float(0.1*len(X_train)))]
X_test = X_train[shuffle_indexes, :]
Y_test = Y_train[shuffle_indexes, :]

np.savetxt('./index/test_shuffled_index_reduced.txt', shuffle_indexes, delimiter = ' ', fmt = '%i')

print('TEST SET INDEX saved')

shuffle_indexes = shuffled_indexes[int(float(0.1 * len(X_train))):len(X_train)]
X_train = X_train[shuffle_indexes, :]
Y_train = Y_train[shuffle_indexes, :]

np.savetxt('./index/train_shuffled_index_reduced.txt', shuffle_indexes, delimiter = ' ', fmt = '%i')
print('TRAIN SET INDEX saved')


base_model = VGG16(weights='imagenet', include_top= False)

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation = 'relu')(x)

predictions = Dense(20, activation ='softmax')(x)

model = Model(inputs=base_model.input, outputs = predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=X_train, y=Y_train, batch_size=64, epochs = 10, validation_data = (X_test, Y_test))


for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

for layer in model.layers[:18]:
    layer.trainable = False

for layer in model.layers[18:]:
    layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum = 0.9), loss='categorical_crossentropy', metrics=['accuracy'])

filepath = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose = 0, save_best_only = True, period = 1)
model.save('./model/vgg_16_fine_tune_'+'batchsize_'+str(sgd_batch_size)+'epoch_'+str(sgd_epoch))


model.fit(x=X_train, y=Y_train, batch_size= sgd_batch_size, epochs = sgd_epoch, validation_data = (X_test, Y_test))


