from keras.layers import *
from keras.models import *
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from fcn_helper_function import *


def normalize(im):
    return im / 255.

data_root = "../data/"


image_datagen_train = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
mask_datagen_train = ImageDataGenerator(preprocessing_function=normalize)

image_generator_train = image_datagen_train.flow_from_directory(
    data_root+'fcn_im_in_train', target_size=(150, 150), class_mode=None, batch_size=10, seed=123, shuffle=True)

mask_generator_train = mask_datagen_train.flow_from_directory(
    data_root+'fcn_masks_train',
    class_mode=None,
    target_size=(150, 150),
    batch_size=10, seed=123, shuffle=True)

# combine generators into one which yields image and masks
train_generator = zip(image_generator_train, mask_generator_train)


def FCN(nClasses,  input_height=512, input_width=512):
    IMAGE_ORDERING = "channels_last"

    img_input = Input(shape=(input_height, input_width, 3))  # Assume 224,224,3

    x = Conv2D(32, (3, 3), activation='relu', padding='same',
               data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(32, (3, 3), activation='relu',
               padding='same', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format=IMAGE_ORDERING)(x)

    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', data_format=IMAGE_ORDERING)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), data_format=IMAGE_ORDERING)(x)

    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', data_format=IMAGE_ORDERING)(x)

    x = (Conv2D(3, (1, 1), activation='relu',
                padding='same', data_format=IMAGE_ORDERING))(x)

    x = Conv2DTranspose(nClasses, kernel_size=(4, 4),  strides=(
        4, 4), use_bias=False, data_format=IMAGE_ORDERING)(x)
    o = (Activation('softmax'))(x)

    model = Model(img_input, o)

    return model


model = FCN(nClasses=3,
            input_height=150,
            input_width=150)
model.summary()
model.compile(loss=[weighted_categorical_crossentropy([1, 1, 0.01])],
              optimizer='adam',
              metrics=[IoU])

history = model.fit_generator(train_generator, epochs=50, steps_per_epoch=100)

model.save('models/fcnn_bin.h5')