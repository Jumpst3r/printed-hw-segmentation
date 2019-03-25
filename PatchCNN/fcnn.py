import os
import pickle

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import tensorflow as tf
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import *
from keras.models import *
from keras.preprocessing.image import ImageDataGenerator
from skimage import filters, img_as_float, img_as_ubyte
from skimage.color import gray2rgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from classifier_fcnn import classify

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


def normalize(im):
    return im / 255.
    



image_datagen_train = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True,rotation_range=5,
        zoom_range=0.2,
        fill_mode='nearest',
        width_shift_range=0.2,
        height_shift_range=0.2)
mask_datagen_train = ImageDataGenerator(preprocessing_function=normalize, rotation_range=5,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest')

image_generator_train = image_datagen_train.flow_from_directory(
    'fcn_im_in_train', target_size=(512, 512), class_mode=None, batch_size=10, seed=123, shuffle=True)

mask_generator_train = mask_datagen_train.flow_from_directory(
    'fcn_masks_train',
    class_mode=None,
    target_size=(512, 512),
    batch_size=10, seed=123, shuffle=True)

image_datagen_valid = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
mask_datagen_valid = ImageDataGenerator(preprocessing_function=normalize)

image_generator_valid = image_datagen_valid.flow_from_directory(
    'fcn_im_in_valid', target_size=(512, 512), class_mode=None, batch_size=10, seed=123, shuffle=True)

mask_generator_valid = mask_datagen_valid.flow_from_directory(
    'fcn_masks_valid',
    class_mode=None,
    target_size=(512, 512),
    batch_size=10, seed=123, shuffle=True)

# combine generators into one which yields image and masks
train_generator = zip(image_generator_train, mask_generator_train)

valid_generator = zip(image_generator_valid, mask_generator_valid)

def FCN8(nClasses,  input_height=512, input_width=512):
    IMAGE_ORDERING = "channels_last"

    img_input = Input(shape=(input_height, input_width, 3))  # Assume 224,224,3

    x = Conv2D(32, (3, 3), activation='relu', padding='same',
               name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',
                     data_format=IMAGE_ORDERING)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',
                     data_format=IMAGE_ORDERING)(x)

    conv7 = (Conv2D(3, (1, 1), activation='relu', padding='same',
                    name="conv7", data_format=IMAGE_ORDERING))(x)

    conv7_4 = Conv2DTranspose(nClasses, kernel_size=(4, 4),  strides=(
        4, 4), use_bias=False, data_format=IMAGE_ORDERING)(conv7)
    o = (Activation('softmax'))(conv7_4)

    model = Model(img_input, o)

    return model


model = FCN8(nClasses=3,
             input_height=512,
             input_width=512)
model.summary()

model_name_save = 'models.hdf5'
checkpoint = [ModelCheckpoint(filepath=model_name_save)]

if os.path.isfile(model_name_save):
    print ("Resumed model's weights from {}".format(model_name_save))
    # load weights
    model.load_weights(model_name_save)



######################################## IoU metric ############################################

def castF(x):
    return K.cast(x, K.floatx())


def castB(x):
    return K.cast(x, bool)


def iou_loss_core(true, pred):  # this can be used as a loss if you make it negative
    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)

    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())


def IoU(true, pred):  # any shape can go - can't be a loss function

    tresholds = [0.5 + (i*.05) for i in range(10)]

    # flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = castF(K.greater(pred, 0.5))

    # total white pixels - (batch,)
    trueSum = K.sum(true, axis=-1)
    predSum = K.sum(pred, axis=-1)

    # has mask or not per image - (batch,)
    true1 = castF(K.greater(trueSum, 1))
    pred1 = castF(K.greater(predSum, 1))

    # to get images that have mask in both true and pred
    truePositiveMask = castB(true1 * pred1)

    # separating only the possible true positives to check iou
    testTrue = tf.boolean_mask(true, truePositiveMask)
    testPred = tf.boolean_mask(pred, truePositiveMask)

    # getting iou and threshold comparisons
    iou = iou_loss_core(testTrue, testPred)
    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]

    # mean of thressholds for true positives and total sum
    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
    truePositives = K.sum(truePositives)

    # to get images that don't have mask in both true and pred
    trueNegatives = (1-true1) * (1 - pred1)  # = 1 -true1 - pred1 + true1*pred1
    trueNegatives = K.sum(trueNegatives)

    return (truePositives + trueNegatives) / castF(K.shape(true)[0])

########################################################################################################################


def weighted_categorical_crossentropy(weights):
    """ weighted_categorical_crossentropy

        Args:
            * weights<ktensor|nparray|list>: crossentropy weights
        Returns:
            * weighted categorical crossentropy function
    """
    if isinstance(weights, list) or isinstance(np.ndarray):
        weights = K.variable(weights)

    def loss(target, output, from_logits=False):
        if not from_logits:
            output /= tf.reduce_sum(output,
                                    len(output.get_shape()) - 1,
                                    True)
            _epsilon = tf.convert_to_tensor(
                K.epsilon(), dtype=output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
            weighted_losses = target * tf.log(output) * weights
            return - tf.reduce_sum(weighted_losses, len(output.get_shape()) - 1)
        else:
            raise ValueError(
                'WeightedCategoricalCrossentropy: not valid with logits')
    return loss


def focal_loss(gamma=2., alpha=0.2):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


'''
if os.path.isfile('models.hdf5'):
    model.load_weights('models.hdf5')
'''

model.compile(loss=[weighted_categorical_crossentropy([1, 0.8, 0.01])],
              optimizer='adam',
              metrics=[IoU])
checkpoint = [ModelCheckpoint(filepath='models.hdf5')]

history = model.fit_generator(train_generator, epochs=10, steps_per_epoch=100, callbacks=checkpoint, validation_data=valid_generator, validation_steps=10)

test = io.imread("test.png")
test = gray2rgb(test)
test = (test - test.mean()) / test.std()


im = model.predict(np.array([test]))[0]

io.imsave("FCNRES.png", im)

model.save('fcnn.h5')
