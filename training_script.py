from train.training_model import train_model
from train.generator import DataGenerator
from train.patch_model import get_model

if __name__ == '__main__':
    # train_h5_path = '/mnt/local/aszeskin/asher/liver_data/BL_patches.h5'
    # validation_h5_path = '/mnt/local/aszeskin/asher/liver_data/FU_patches.h5'
    # train_h5_path = '/mnt/local/aszeskin/asher/liver_data/split_BL/patches_train.h5'
    # validation_h5_path = '/mnt/local/aszeskin/asher/liver_data/split_BL/patches_validation.h5'
    # train_h5_path = '/mnt/local/aszeskin/asher/liver_data/sample_step_2_split_BL/patches_train.h5'
    # validation_h5_path = '/mnt/local/aszeskin/asher/liver_data/sample_step_2_split_BL/patches_validation.h5'
    # train_h5_path = '/mnt/local/aszeskin/asher/liver_data/BL_all_patches.h5'
    # validation_h5_path = '/mnt/local/aszeskin/asher/liver_data/FU_patches.h5'
    train_h5_path = '/mnt/local/aszeskin/asher/liver_data/combined_patches_step_2_train.h5'
    validation_h5_path  = '/mnt/local/aszeskin/asher/liver_data/combined_patches_step_2_validation.h5'
    weights_save_path = '/mnt/local/aszeskin/asher/weights'
    title = 'fixed_normalization'
    batch_size = 64
    memory_fraction = 0.2
    do_augmentations = True
    dim = (35, 35, 1)
    model = get_model()
    model.summary()
    train_gen = DataGenerator(train_h5_path, 'patches', 'labels', do_augmentations, batch_size=batch_size, dim=dim)
    validation_gen = DataGenerator(validation_h5_path, 'patches', 'labels', False, batch_size=batch_size,
                                   dim=dim)
    train_model(model, train_gen, validation_gen, 100, weights_save_path, memory_fraction, title)