from keras import backend as K
from keras.losses import binary_crossentropy

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)

def iou(y_true, y_pred):
    y_true = K.cast(K.greater(y_true, 0.5), dtype='float32')
    y_pred = K.cast(K.greater(y_pred, 0.5), dtype='float32')
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(K.clip(y_true + y_pred, 0, 1), axis=3), axis=2), axis=1)
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))