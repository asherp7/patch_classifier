from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Model


def get_model():
    input = Input(shape=(35, 35, 1))
    conv_1 = Conv2D(filters=48, kernel_size=(4, 4), activation='relu')(input)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
    conv_2 = Conv2D(filters=48, kernel_size=(5, 5), activation='relu')(pool_1)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    conv_3 = Conv2D(filters=48, kernel_size=(5, 5), activation='relu')(pool_2)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
    flatten_1 = Flatten()(pool_3)
    dense_1 = Dense(200, activation='relu')(flatten_1)
    output = Dense(2, activation='softmax')(dense_1)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = get_model()
    model.summary()