from keras.layers import *
from keras.models import *
import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from img_utils import getbinim
from get_raw_output import classify
import skimage.io as io
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from fcn_helper_function import *


def normalize(im):
    return im / 255.

data_root = "../dataset_generation/data/"
WIDTH = 512
USE_AUGMENTATION = False


if USE_AUGMENTATION:
    image_datagen_train = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.2)
    mask_datagen_train = ImageDataGenerator(preprocessing_function=normalize,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.2)
else:
    image_datagen_train = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
    mask_datagen_train = ImageDataGenerator(preprocessing_function=normalize)


image_generator_train = image_datagen_train.flow_from_directory(
    data_root+'fcn_im_in_train', target_size=(WIDTH, WIDTH), class_mode=None, batch_size=20, seed=123, shuffle=True)

mask_generator_train = mask_datagen_train.flow_from_directory(
    data_root+'fcn_masks_train',
    class_mode=None,
    target_size=(WIDTH, WIDTH),
    batch_size=20, seed=123, shuffle=True)

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
            input_height=WIDTH,
            input_width=WIDTH)
model.summary()
model.compile(loss=[weighted_categorical_crossentropy([1, 1, 0.1])],
              optimizer='adam',
              metrics=[IoU])

################################################# Tensorboard callbacks ############################################
keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0,  
          write_graph=True, write_images=True)

tbCallBack = keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor.astype('uint8'))
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)

class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, tag):
        super().__init__() 
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):
        # Load image
        image = io.imread("input/Inkedprinted-7-2_LI.jpg", as_gray=True)
        # Do something to the image
        image = classify(image, self.model)
        image = make_image(image)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
        writer = tf.summary.FileWriter('logs')
        writer.add_summary(summary, epoch)
        writer.close()

        return

tbi_callback = TensorBoardImage('FCN learning process')


model.fit_generator(train_generator, epochs=50, steps_per_epoch=100, callbacks=[tbCallBack, tbi_callback])


model.save('models/fcnn_bin.h5')