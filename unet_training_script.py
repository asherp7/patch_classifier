from train.training_model import train_model
from train.unet_generator import DataGenerator
from train.attention_unet_model import get_model

if __name__ == '__main__':
    train_h5_path = '/mnt/local/aszeskin/asher/liver_data/unet_26_3/unet_train.h5'
    validation_h5_path = '/mnt/local/aszeskin/asher/liver_data/unet_26_3/unet_validation.h5'
    weights_save_path = '/cs/labs/josko/asherp7/follow_up/weights'
    title = 'unet_96_patches'
    batch_size = 8
    memory_fraction = 0.5
    do_augmentations = False
    dim = (96, 96, 1)
    dropout_rate = 0.2
    batch_normalization = False
    model = get_model(input_shape=dim, drop=dropout_rate, bn=batch_normalization)
    model.summary()
    train_gen = DataGenerator(train_h5_path, 'patches', 'mask_patches', do_augmentations, batch_size=batch_size, dim=dim)
    validation_gen = DataGenerator(validation_h5_path, 'patches', 'mask_patches', do_augmentations, batch_size=batch_size, dim=dim)
    train_model(model, train_gen, validation_gen, 100, weights_save_path, memory_fraction, title)