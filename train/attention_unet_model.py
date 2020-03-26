from train.loss_functions import jaccard_distance
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate, \
    BatchNormalization, Activation, UpSampling2D, Add, Activation, Multiply, Input


def conv_block(input, num_channels, filter_size=3, activation='relu', padding='same', bn=True, drop=None):
    conv1 = Conv2D(num_channels, filter_size, padding=padding, kernel_initializer='he_normal')(input)
    if bn:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation)(conv1)

    conv2 = Conv2D(num_channels, filter_size, padding=padding, kernel_initializer='he_normal')(conv1)
    if bn:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation)(conv2)

    if drop is not None:
        conv2 = Dropout(drop)(conv2)

    return conv2


def attention_block(up_features, skip_features, F_int):
    W_g = Conv2D(F_int, kernel_size=1)(skip_features)
    W_g = BatchNormalization()(W_g)

    W_x = Conv2D(F_int, kernel_size=1)(up_features)
    W_x = BatchNormalization()(W_x)

    addition = Add()([W_g, W_x])
    relu = Activation('relu')(addition)

    psi = Conv2D(1, kernel_size=1)(relu)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)

    mult = Multiply()([psi, skip_features])

    return mult


def get_model(input_shape=(96, 96, 1), pretrained_weights=None, use_atten=False, drop=None, bn=False):
    inputs = Input(shape=input_shape)

    conv_down1 = conv_block(inputs, num_channels=64, drop=drop, bn=bn)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv_down1)

    conv_down2 = conv_block(pool1, num_channels=128,drop=drop, bn=bn)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv_down2)

    conv_down3 = conv_block(pool2, num_channels=256,drop=drop, bn=bn)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv_down3)

    conv_down4 = conv_block(pool3, num_channels=512,drop=drop, bn=bn)

    # up
    # convT1 = Conv2DTranspose(filters=256,kernel_size=(2, 2), padding='same', strides=(2,2))(conv_down4)
    convT1 = UpSampling2D(size=(2,2))(conv_down4)
    if use_atten:
        conv_down3 = attention_block(up_features=convT1, skip_features=conv_down3,F_int=256)
    merge1 = concatenate([convT1, conv_down3], axis=3)
    conv_up1 = conv_block(merge1, num_channels=256, drop=drop, bn=bn)

    # convT2 = Conv2DTranspose(filters=128,kernel_size=(2, 2), padding='same', strides=(2,2))(conv_up1)
    convT2 = UpSampling2D(size=(2,2))(conv_up1)
    if use_atten:
        conv_down2 = attention_block(up_features=convT2, skip_features=conv_down2,F_int=256)
    merge2 = concatenate([convT2, conv_down2], axis=3)
    conv_up2 = conv_block(merge2, num_channels=128, drop=drop, bn=bn)

    # convT3 = Conv2DTranspose(filters=64,kernel_size=(2, 2), padding='same', strides=(2,2))(conv_up2)
    convT3 = UpSampling2D(size=(2,2))(conv_up2)
    if use_atten:
        conv_down1 = attention_block(up_features=convT3, skip_features=conv_down1,F_int=256)
    merge3 = concatenate([convT3, conv_down1], axis=3)
    conv_up3 = conv_block(merge3, num_channels=64, drop=drop, bn=bn)

    up4 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_up3)
    up5 = Conv2D(1, 1, activation='sigmoid')(up4)

    model = Model(input=inputs, output=up5)

    model.compile(optimizer=Adam(lr=0.0001), loss=jaccard_distance, metrics=['accuracy'])

    model.summary()
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


if __name__ == '__main__':
    model = get_model()
    model.summary()
