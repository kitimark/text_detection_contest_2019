import glob
from keras.optimizers import Adam
import numpy as np
import cv2

from textdetection.model import get_model
from textdetection.validation import *

IMAGE_SIZE = (512,512)

model = get_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

print(model.summary())

model.compile(optimizer=Adam(lr=1e-4), loss=bce_dice_loss, metrics=['accuracy',iou])

model.load_weights('model/my_model_2.h5')

path = glob.glob("Contest/Input/*.jpg")
mean_score = list()

for myfile in path:
    test_im = cv2.imread(myfile)
    true_size = test_im.shape
    imshow_size = (512,round(true_size[0]*512/true_size[1]))
    #cv2.imshow('Input',cv2.resize(test_im, imshow_size))

    test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
    test_im = cv2.resize(test_im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    test_im = test_im/255.
    test_im = np.expand_dims(test_im, axis=0)
    segmented = model.predict(test_im)
    segmented = np.around(segmented)
    # im_true = cv2.imread(myfile.replace("Input","Output"),0)
    segmented = (segmented[0, :, :, 0]*255).astype('uint8')
    im_pred = cv2.resize(segmented, imshow_size)
    #cv2.imshow('Output',im_pred)
    # im_pred = cv2.resize(im_pred, (im_true.shape[1],im_true.shape[0]), interpolation = cv2.INTER_AREA)
    #im_true =  cv2.resize(im_true, IMAGE_SIZE)
    #im_pred =  cv2.resize(im_pred, IMAGE_SIZE)
    myfile = myfile.replace("Input","Result")
    cv2.imwrite(myfile,im_pred)

print("Total:",np.mean(mean_score))