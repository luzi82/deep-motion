# more experimentation with U-Net as a base
# see get_unet_2() for final network decision

from keras.layers import Layer, Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Deconvolution2D, Activation
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential, Model

# theano only model?

def get_unet():
    inputs = Input((6, 320, 640))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(3, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    return model


# this is the model we use now. basically unet, with an extra block pair, and batch norm
# note: as a result of using batch norm, the model's weights is locked to input_shape
# TODO should probably make a new net without batch norm....
def get_unet_2(input_shape):
    inputs = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(bn1)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    bn2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn2)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    bn3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(bn3)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    bn4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(bn4)
    bn4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    bn5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(bn5)
    bn5 = BatchNormalization()(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(bn5)

    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool5)
    bn5_2 = BatchNormalization()(conv5_2)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(bn5_2)
    bn5_2 = BatchNormalization()(conv5_2)

    up5_2 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(bn5_2), bn5])
    conv6_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(up5_2)
    bn6_2 = BatchNormalization()(conv6_2)
    conv6_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(bn6_2)
    bn6_2 = BatchNormalization()(conv6_2)

    up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(bn6_2), bn4])
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    bn6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(bn6)
    bn6 = BatchNormalization()(conv6)

    up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(bn6), bn3])
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    bn7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(bn7)
    bn7 = BatchNormalization()(conv7)

    up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(bn7), bn2])
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    bn8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn8)
    bn8 = BatchNormalization()(conv8)

    up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(bn8), bn1])
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    bn9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(bn9)
    bn9 = BatchNormalization()(conv9)

    conv10 = Conv2D(int(input_shape[2]/2), (1, 1), activation='sigmoid')(bn9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def conv_block(input, nb_filter):
    conv1 = Conv2D(nb_filter, nb_row=3, nb_col=3, padding='same')(input)
    relu1 = Activation("relu")(conv1)
    conv2 = Conv2D(nb_filter, nb_row=3, nb_col=3, padding='same')(relu1)
    relu2 = Activation("relu")(conv2)
    conv3 = Conv2D(nb_filter, nb_row=3, nb_col=3, padding='same')(relu2)
    relu3 = Activation("relu")(conv3)
    pool = MaxPooling2D(pool_size=(2, 2))(relu3)
    return pool


def deconv_block(input, nb_filter, output_shape):
    deconv = Deconvolution2D(nb_filter, nb_row=4, nb_col=4, output_shape=output_shape, padding='same', subsample=(2,2))(input)
    relu1 = Activation("relu")(deconv)
    conv1 = Conv2D(nb_filter, nb_row=3, nb_col=3, padding='same')(relu1)
    relu2 = Activation("relu")(conv1)
    conv2 = Conv2D(nb_filter, nb_row=3, nb_col=3, padding='same')(relu2)
    relu3 = Activation("relu")(conv2)
    return relu3

# never got better results than unet_2, even though it's more organized (less error prone)...
def get_unet_3(input_shape, batch_size):
    input = Input(input_shape)
    conv_block1 = conv_block(input=input, nb_filter=96)
    conv_block2 = conv_block(input=conv_block1, nb_filter=96)
    conv_block3 = conv_block(input=conv_block2, nb_filter=128)
    conv_block4 = conv_block(input=conv_block3, nb_filter=128)
    conv_block5 = conv_block(input=conv_block4, nb_filter=128)

    deconv_block5 = deconv_block(input=conv_block5, nb_filter=128, output_shape=(batch_size, 128, input_shape[1]/16, input_shape[2]/16))
    merge4 = Concatenate(axis=3)([deconv_block5, conv_block4])
    deconv_block4 = deconv_block(input=merge4, nb_filter=128, output_shape=(batch_size, 128, input_shape[1]/8, input_shape[2]/8))
    merge3 = Concatenate(axis=3)([deconv_block4, conv_block3])
    deconv_block3 = deconv_block(input=merge3, nb_filter=128, output_shape=(batch_size, 128, input_shape[1]/4, input_shape[2]/4))
    merge2 = Concatenate(axis=3)([deconv_block3, conv_block2])
    deconv_block2 = deconv_block(input=merge2, nb_filter=96, output_shape=(batch_size, 96, input_shape[1]/2, input_shape[2]/2))
    deconv_block1 = deconv_block(input=deconv_block2, nb_filter=96, output_shape=(batch_size, 96, input_shape[1], input_shape[2]))

    output = Conv2D(3, (1, 1), activation="relu")(deconv_block1)

    model = Model(input=input, output=output)

    return model

def get_unet_bak():
    inputs = Input((6, 32, 64))
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv4)

    # too much res for 36x64
    # conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
    # conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
    #
    # up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool4)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv4])
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

    temp = UpSampling2D(size=(2, 2))(conv7)

    up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv3])
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv2])
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)

    up1 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv9), conv1])
    conv10 = Conv2D(3, (1, 1), activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    return model
