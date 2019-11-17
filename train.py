from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils import plot_model
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2

from textdetection.validation import *
from textdetection.model import get_model

BATCH_SIZE = 4
MAX_EPOCH = 100
IMAGE_SIZE = (512,512)
TRAIN_IM = 436
VALIDATE_IM = 15

model = get_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

plot_model(model, 'model.png')
print(model.summary())

model.compile(optimizer=Adam(lr=1e-4), loss=bce_dice_loss, metrics=['accuracy',iou])

def myGenerator(type):
    datagen = ImageDataGenerator(rescale=1./255)

    input_generator = datagen.flow_from_directory(
        ''+type,
        classes = ['Input'],
        class_mode=None,
        color_mode='rgb',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed = 1)

    expected_output_generator = datagen.flow_from_directory(
        ''+type,
        classes = ['Output'],
        class_mode=None,
        color_mode='grayscale',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed = 1)

    while True:
        in_batch = input_generator.next()
        out_batch = expected_output_generator.next()
        yield in_batch, out_batch

checkpoint = ModelCheckpoint('my_model.h5', verbose=1, monitor='val_iou',save_best_only=True, mode='max')


class ShowPredictSegment(Callback):
    def on_epoch_end(self, epoch, logs={}):
        testfileloc = ['Validation/Input/100.jpg',
                       'Validation/Input/101.jpg',
                       'Validation/Input/102.jpg',
                       'Validation/Input/103.jpg']

        for k in range(len(testfileloc)):
            test_im = cv2.imread(testfileloc[k])
            true_size = test_im.shape
            if true_size[1] >=  true_size[0]:
                imshow_size = (300, round(true_size[0] * 300 / true_size[1]))
            else:
                imshow_size = (round(true_size[1] * 300 / true_size[0]),300)
            cv2.imshow('Input'+str(k), cv2.resize(test_im, imshow_size))
            cv2.moveWindow('Input'+str(k), 20 + 350 * k,10)

            test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
            test_im = cv2.resize(test_im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
            test_im = test_im / 255.
            test_im = np.expand_dims(test_im, axis=0)
            segmented = model.predict(test_im)
            segmented = np.around(segmented)
            segmented = (segmented[0, :, :, 0] * 255).astype('uint8')
            cv2.imshow('Output'+str(k), cv2.resize(segmented, imshow_size))
            cv2.moveWindow('Output'+str(k), 20 + 350 * k,400)
            cv2.waitKey(100)

show_result = ShowPredictSegment()

h = model.fit_generator(myGenerator('Dataset'),
                        steps_per_epoch=TRAIN_IM/BATCH_SIZE,
                        epochs=MAX_EPOCH,
                        validation_data=myGenerator('validation'),
                        validation_steps=VALIDATE_IM/BATCH_SIZE,
                        callbacks=[checkpoint,show_result])

plt.plot(h.history['iou'])
plt.plot(h.history['val_iou'])
plt.show()
