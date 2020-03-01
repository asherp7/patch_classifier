from train.training_model import train_model
from train.unet_generator import DataGenerator
from train.attention_unet_model import get_model

if __name__ == '__main__':
    train_h5_path = '/mnt/local/aszeskin/asher/liver_data/unet_BL_all_patches.h5'
    validation_h5_path = '/mnt/local/aszeskin/asher/liver_data/unet_FU_patches.h5'
    weights_save_path = '/mnt/local/aszeskin/asher/weights'
    title = 'unet_train_all_BL'
    batch_size = 16
    memory_fraction = 0.5
    do_augmentations = False
    dim = (64, 64, 1)
    model = get_model()
    model.summary()
    train_gen = DataGenerator(train_h5_path, 'patches', 'mask_patches', do_augmentations, batch_size=batch_size, dim=dim)
    validation_gen = DataGenerator(validation_h5_path, 'patches', 'mask_patches', do_augmentations, batch_size=batch_size, dim=dim)
    train_model(model, train_gen, validation_gen, 100, weights_save_path, memory_fraction, title)