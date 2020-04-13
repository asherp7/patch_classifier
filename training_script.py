from train.training_model import train_model
from train.generator import DataGenerator
from train.patch_model import get_model

if __name__ == '__main__':
    train_h5_path = '/mnt/local/aszeskin/asher/liver_data/combined_105_ct_patches_step_3_train.h5'
    validation_h5_path = '/mnt/local/aszeskin/asher/liver_data/combined_105_ct_patches_step_3_validation.h5'
    # train_h5_path = '/mnt/local/aszeskin/asher/liver_data/combined_patches_step_2_train.h5'
    # validation_h5_path = '/mnt/local/aszeskin/asher/liver_data/combined_patches_step_2_validation.h5'
    # train_h5_path = '/mnt/local/aszeskin/asher/liver_data/seperated_26_3/combined_patches_step_2_train.h5'
    # validation_h5_path = '/mnt/local/aszeskin/asher/liver_data/seperated_26_3/combined_patches_step_2_validation.h5'
    # train_h5_path = '/mnt/local/aszeskin/asher/liver_data/combined_patches_step_2_train.h5'
    # validation_h5_path = '/mnt/local/aszeskin/asher/liver_data/combined_patches_step_2_validation.h5'
    weights_save_path = '/mnt/local/aszeskin/asher/weights'
    title = 'f1_loss_105_cases_step_3'
    batch_size = 64
    memory_fraction = 0.3
    do_augmentations = True
    dim = (35, 35, 1)
    model = get_model()
    model.summary()
    train_gen = DataGenerator(train_h5_path, 'patches', 'labels', do_augmentations, batch_size=batch_size, dim=dim)
    validation_gen = DataGenerator(validation_h5_path, 'patches', 'labels', False, batch_size=batch_size,
                                   dim=dim)
    train_model(model, train_gen, validation_gen, 100, weights_save_path, memory_fraction, title)