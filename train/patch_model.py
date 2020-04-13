from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Model
from keras import backend as K
import tensorflow as tf


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def macro_double_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.

    Args:
    y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
    y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
    cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0)  # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost)  # average on all labels
    return macro_cost


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

    # SoftMax:
    output = Dense(2, activation='softmax')(dense_1)

    # Sigmoid:
    # output = Dense(1, activation='sigmoid')(dense_1)

    model = Model(inputs=input, outputs=output)
    # model.compile(optimizer='adam',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])

    # Adi's loss function:
    model.compile(loss=macro_double_soft_f1, optimizer='adam', metrics=['accuracy', f1])
    return model


if __name__ == '__main__':
    model = get_model()
    model.summary()