import skimage.io as io
from keras.layers import *
from keras.models import *
from keras.utils import to_categorical
from skimage import img_as_float
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from fcn_helper_function import *

np.random.seed(123)

X_train = []
X_valid = []
y_train = []
y_valid = []

print("Reading images...")

inputs_train = io.imread_collection("printed-hw-seg/train/data/*.png")
inputs_valid = io.imread_collection("printed-hw-seg/val/data/*.png")

masks_train = io.imread_collection("printed-hw-seg/train/gt/*.png")
masks_valid = io.imread_collection("printed-hw-seg/val/gt/*.png")


def mask2rgb(mask):
    result = np.zeros((mask.shape))
    result[:, :][np.where((mask[:, :] == [0, 0, 1]).all(axis=2))] = [1, 0, 0]
    result[:, :][np.where((mask[:, :] == [0, 0, 2]).all(axis=2))] = [0, 1, 0]
    result[:, :][np.where((mask[:, :] == [0, 0, 4]).all(axis=2))] = [0, 0, 1]
    return result

for im_in,im_mask in zip(inputs_train, masks_train):
    X_train.append(img_as_float(gray2rgb(im_in)))
    y_train.append(mask2rgb(im_mask))

for im_in,im_mask in zip(inputs_valid, masks_valid):
    X_valid.append(img_as_float(gray2rgb(im_in)))
    y_valid.append(mask2rgb(im_mask))


X_train = np.array(X_train)
X_valid = np.array(X_valid)

y_train = np.array(y_train)
y_valid = np.array(y_valid)

print("Done!")

print('Number of training samples:' + str(len(X_train)))
print('Number of validation samples:' + str(len(y_valid)))


def FCN(nClasses,  input_height=256, input_width=256):
    IMAGE_ORDERING = "channels_last"

    img_input = Input(shape=(input_height, input_width, 3))

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
            input_height=256,
            input_width=256)
model.summary()
model.compile(loss=[weighted_categorical_crossentropy([1, 1, 0.1])],
              optimizer='adam',
              metrics=[IoU])

################################################# Tensorboard callbacks ############################################

model.fit(x=X_train, y=y_train, epochs=100, batch_size=32, validation_data=(X_valid,y_valid))


model.save('models/fcnn_bin.h5')