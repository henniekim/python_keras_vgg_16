

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.initializers import he_normal
from keras.initializers import Zeros
from keras.activations import relu
from keras.layers import Flatten
from keras.activations import softmax
from keras import optimizers
from keras.losses import categorical_crossentropy
from keras.metrics import top_k_categorical_accuracy

from keras.applications import VGG16, VGG19


import os
import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="3"
##--data load --##

image_path = '/datahdd/workdir/donghyun/faster_rcnn_kdh/PascalDataSetReduced/'
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

np.savetxt('test_shuffled_index_reduced.txt', shuffle_indexes, delimiter = ' ', fmt = '%i')

print('TEST SET INDEX saved')

shuffle_indexes = shuffled_indexes[int(float(0.1 * len(X_train))):len(X_train)]
X_train = X_train[shuffle_indexes, :]
Y_train = Y_train[shuffle_indexes, :]

np.savetxt('train_shuffled_index_reduced.txt', shuffle_indexes, delimiter = ' ', fmt = '%i')
print('TRAIN SET INDEX saved')


model = Sequential()


##-------------------------------------------------------------------------##
model.add(Conv2D( filters = 64, kernel_size = (3, 3), strides = 1, padding = "same", activation = 'relu',
                   input_shape = (224, 224, 3)))

model.add(Conv2D( filters = 64, kernel_size = (3, 3), strides = 1, padding = "same", activation = 'relu',
                  ))

model.add(MaxPooling2D( pool_size = (2,2), strides = (2,2), padding= 'same', data_format = None))


##-------------------------------------------------------------------------##
model.add(Conv2D( filters = 128, kernel_size = (3, 3), strides = 1, padding = "same", activation = 'relu',
                  ))

model.add(Conv2D( filters = 128, kernel_size = (3, 3), strides = 1, padding = "same", activation = 'relu',
                  ))

model.add(MaxPooling2D( pool_size = (2,2), strides = (2,2), padding= 'same', data_format = None))


##-------------------------------------------------------------------------##
model.add(Conv2D( filters = 256, kernel_size = (3, 3), strides = 1, padding = "same", activation = 'relu',
                  ))

model.add(Conv2D( filters = 256, kernel_size = (3, 3), strides = 1, padding = "same", activation = 'relu',
                  ))

model.add(Conv2D( filters = 256, kernel_size = (3, 3), strides = 1, padding = "same", activation = 'relu',
                  ))

model.add(MaxPooling2D( pool_size = (2,2), strides = (2,2), padding= 'same', data_format = None))


##-------------------------------------------------------------------------##
model.add(Conv2D( filters = 512, kernel_size = (3, 3), strides = 1, padding = "same", activation = 'relu',
                  ))

model.add(Conv2D( filters = 512, kernel_size = (3, 3), strides = 1, padding = "same", activation = 'relu',
                  ))

model.add(Conv2D( filters = 512, kernel_size = (3, 3), strides = 1, padding = "same", activation = 'relu',
                  ))

model.add(MaxPooling2D( pool_size = (2,2), strides = (2,2), padding= 'same', data_format = None))


##-------------------------------------------------------------------------##
model.add(Conv2D( filters = 512, kernel_size = (3, 3), strides = 1, padding = "same", activation = 'relu',
                  ))

model.add(Conv2D( filters = 512, kernel_size = (3, 3), strides = 1, padding = "same", activation = 'relu',
                  ))

model.add(Conv2D( filters = 512, kernel_size = (3, 3), strides = 1, padding = "same", activation = 'relu',
                  ))

model.add(MaxPooling2D( pool_size = (2,2), strides = (2,2), padding= 'same', data_format = None))


##-------------------------------------------------------------------------##
model.add(Flatten())

model.add(Dense( units = 1024, activation = 'relu'))


model.add(Dense( units = 20, activation = 'softmax'))

model.summary()


##---OPTIMIZERS---##
adam = optimizers.adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay= 0, amsgrad = False)
momentum = optimizers.SGD(lr=0.01, momentum = 0.9, decay=1e-6)


model.compile(optimizer = adam, loss = categorical_crossentropy, metrics=['accuracy'])
# when using the categorical_crossentropy loss, your targets should be in categorical format (one- hot encoding)


model.fit(X_train, Y_train, batch_size = 64, epochs = 100, validation_data=(X_test, Y_test))
#score = model.evaluate(X_test, Y_test, batch_size = 64)



