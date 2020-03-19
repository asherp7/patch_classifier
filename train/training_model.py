import os
from datetime import datetime
from train.generator import DataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from train.training_utils import limit_gpu_memory


def create_weights_save_dir(save_path, title):
    now = datetime.now() # current date and time
    date_time = now.strftime("_%Y-%m-%d_%H-%M-%S")
    weights_path = os.path.join(save_path, title+date_time)
    os.mkdir(weights_path)
    return weights_path


def train_model(model, train_gen, validation_gen, num_epochs, save_path, memory_fraction, title):
    limit_gpu_memory(memory_fraction)
    weights_path = create_weights_save_dir(save_path, title)
    weight_file_path = os.path.join(weights_path, "weights-{epoch:02d}-{val_acc:.2f}.hdf5")
    checkpoint = ModelCheckpoint(weight_file_path, monitor='val_acc')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=1e-6)
    callbacks_list = [checkpoint, reduce_lr]
    model.fit_generator(generator=train_gen,
                        epochs=num_epochs,
                        callbacks=callbacks_list,
                        validation_data=validation_gen)
