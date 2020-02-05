from train.training_model import train_model

if __name__ == '__main__':
    # train_h5_path = '/mnt/local/aszeskin/asher/liver_data/BL_patches.h5'
    # validation_h5_path = '/mnt/local/aszeskin/asher/liver_data/FU_patches.h5'
    # train_h5_path = '/mnt/local/aszeskin/asher/liver_data/split_BL/patches_train.h5'
    # validation_h5_path = '/mnt/local/aszeskin/asher/liver_data/split_BL/patches_validation.h5'
    train_h5_path = '/mnt/local/aszeskin/asher/liver_data/sample_step_2_split_BL/patches_train.h5'
    validation_h5_path = '/mnt/local/aszeskin/asher/liver_data/sample_step_2_split_BL/patches_validation.h5'
    weights_save_path = '/mnt/local/aszeskin/asher/weights'
    title = 'train_with_augmentations_sampling_step-2_batch-64'
    batch_size = 64
    memory_fraction = 0.2
    do_augmentations = True
    train_model(train_h5_path, validation_h5_path, 100, batch_size, weights_save_path, memory_fraction,
                do_augmentations, title)