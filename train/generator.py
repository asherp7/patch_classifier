import numpy as np
import keras
import h5py

from train.training_utils import augment_batch, custom_augment_img

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, h5_filepath, data_name, labels_name, do_augmentations, batch_size=32, dim=(35, 35, 1),
                 n_classes=2, shuffle=True, min_clip_value=-100, max_clip_value=150):
        'Initialization'
        self.hf = h5py.File(h5_filepath, 'r')
        self.data_name = data_name
        self.labels_name = labels_name
        self.num_patches = self.hf[self.labels_name].shape[0]
        self.do_augmentations = do_augmentations
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = range(self.num_patches)
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.min_clip_value = min_clip_value
        self.max_clip_value = max_clip_value
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs), dtype=np.int32)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            patch = self.hf[self.data_name][ID]
            X[i,] = np.expand_dims(patch, axis=-1)

            # Store class
            y[i] = self.hf[self.labels_name][ID]

        # Clip values:
        np.clip(X, self.min_clip_value, self.max_clip_value, out=X)

        # Normalize:
        X = (X - self.min_clip_value) / (self.max_clip_value - self.min_clip_value)

        # do augmentations
        if self.do_augmentations:
            # transform to uint8:
            X = (X * 255).astype(np.uint8)
            # augment batch:
            X = augment_batch(X)
            # transform back to (0,1) interval:
            X = (X / 255.0).astype(np.float32)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


