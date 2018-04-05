from keras.models import load_model

import numpy as np
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"


sgd_batch_size = 64
sgd_epoch = 10
filepath = './model/vgg_16_fine_tune_'+'batchsize_'+str(sgd_batch_size)+'epoch_'+str(sgd_epoch)
image_path = './samples/mmi_plant2.jpg'

model = load_model(filepath)
inputdata = cv2.imread(image_path)
cv2.imshow('img', inputdata)
cv2.waitKey(300)


inputdata = cv2.resize(inputdata, dsize =(224,224))
inputdata = inputdata.reshape(1, 224,224,3)

result = model.predict(x = inputdata, batch_size = 1)
category = np.argmax(result)

if category == 0:
  label = 'person'
elif category == 1:
  label = 'bird'
elif category == 2:
  label = 'cat'
elif category == 3:
  label = 'cow'
elif category == 4:
  label = 'dog'
elif category == 5:
  label = 'horse'
elif category == 6:
  label = 'sheep'
elif category == 7:
  label = 'aeroplane'
elif category == 8:
  label = 'bicycle'
elif category == 9:
  label = 'boat'
elif category == 10:
  label = 'bus'
elif category == 11:
  label = 'car'
elif category == 12:
  label = 'motorbike'
elif category == 13:
  label = 'train'
elif category == 14:
  label = 'bottle'
elif category == 15:
  label = 'chair'
elif category == 16:
  label = 'diningtable'
elif category == 17:
  label = 'pottedplant'
elif category == 18:
  label = 'sofa'
elif category == 19:
  label = 'tvmonitor'

print('category : '+label)
print('accuracy : '+str(result[0, category]))

