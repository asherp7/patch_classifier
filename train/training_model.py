import os
from datetime import datetime
from train.model import get_model
from train.generator import DataGenerator
from keras.callbacks import ModelCheckpoint
from train.training_utils import limit_gpu_memory


def create_weights_save_dir(save_path, title):
    now = datetime.now() # current date and time
    date_time = now.strftime("_%Y-%m-%d_%H-%M-%S")
    weights_path = os.path.join(save_path, title+date_time)
    os.mkdir(weights_path)
    return weights_path


def train_model(train_data_path, validation_data_path, num_epochs, batch_size, save_path, memory_fraction,
                do_augmentations, title):
    limit_gpu_memory(memory_fraction)
    model = get_model()
    model.summary()
    train_gen = DataGenerator(train_data_path, 'patches', 'labels', do_augmentations, batch_size=batch_size)
    validation_gen = DataGenerator(validation_data_path, 'patches', 'labels', False, batch_size=batch_size)
    weights_path = create_weights_save_dir(save_path, title)
    weight_file_path = os.path.join(weights_path, "weights-{epoch:02d}-{val_acc:.2f}.hdf5")
    checkpoint = ModelCheckpoint(weight_file_path, monitor='val_acc')
    callbacks_list = [checkpoint]
    model.fit_generator(generator=train_gen, epochs=num_epochs, callbacks=callbacks_list,
                        validation_data=validation_gen)
