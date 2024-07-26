import tensorflow.keras.backend as K

# loss functions
def dice_score(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3]  )
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


def dice_loss(y_true, y_pred):
    return 1 - dice_score(y_true, y_pred)

def bce_loss(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=[1, 2, 3])

def dice_bce_loss(y_true, y_pred):
    dice_loss_value = dice_loss(y_true, y_pred)
    bce_loss_value = bce_loss(y_true, y_pred)
    return dice_loss_value + bce_loss_value


# Experimental
def iou_score(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)

def iou_loss(y_true, y_pred):
    return 1 - iou_score(y_true, y_pred)