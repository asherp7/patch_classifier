import os
from model import get_model
from generator import DataGenerator
from keras.callbacks import ModelCheckpoint
from training_utils import limit_gpu_memory



def create_weights_save_dir(weight_dir, train_title):
    pass


def train_model(train_data_path, validation_data_path, num_epochs, batch_size, weight_dir, memory_fraction):
    limit_gpu_memory(memory_fraction)
    model = get_model()
    model.summary()
    train_gen = DataGenerator(train_data_path, 'patches', 'labels', False, batch_size=batch_size)
    validation_gen = DataGenerator(validation_data_path, 'patches', 'labels', False, batch_size=batch_size)
    weight_file_path = os.path.join(weight_dir, "weights-{epoch:02d}-{val_acc:.2f}.hdf5")
    checkpoint = ModelCheckpoint(weight_file_path, monitor='val_acc', verbose=1)
    callbacks_list = [checkpoint]
    model.fit_generator(generator=train_gen, epochs=num_epochs, callbacks=callbacks_list,
                        validation_data=validation_gen)


if __name__ == '__main__':
    # train_h5_path = '/mnt/local/aszeskin/asher/liver_data/BL_patches.h5'
    # validation_h5_path = '/mnt/local/aszeskin/asher/liver_data/FU_patches.h5'
    train_h5_path = '/mnt/local/aszeskin/asher/liver_data/split_BL/patches_train.h5'
    validation_h5_path = '/mnt/local/aszeskin/asher/liver_data/split_BL/patches_validation.h5'
    weight_dir = '/mnt/local/aszeskin/asher/weights'
    batch_size = 64
    memory_fraction = 0.2
    train_model(train_h5_path, validation_h5_path, 100, batch_size, weight_dir, memory_fraction)