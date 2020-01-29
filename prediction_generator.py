import nibabel as nib
import numpy as np
import keras


class PredictionGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, scan_filepath, organ_segmentation_filepath, batch_size=32, dim=(35, 35, 1),
                 min_clip_value=-100, max_clip_value=150, sampling_step=1):
        'Initialization'
        self.batch_size = batch_size
        self.dim = dim
        self.patch_size = dim[0]
        self.min_clip_value = min_clip_value
        self.max_clip_value = max_clip_value
        self.sampling_step = sampling_step
        self.scan_data = nib.load(scan_filepath).get_fdata()
        self.organ_segmentation = nib.load(organ_segmentation_filepath).get_fdata()
        self.patch_list = self.create_pathches()
        self.num_patches = len(self.patch_list)
        self.list_IDs = range(self.num_patches)
        self.on_epoch_end()
        self.index_array = np.empty((self.num_patches, 3))

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
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs), dtype=np.int32)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        indices = np.empty((self.batch_size, 3))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            patch, (x,y,z) = self.patch_list[ID]
            X[i,] = np.expand_dims(patch, axis=-1)

            indices[i,] = np.asarray([x, y, z])

        # Clip values:
        np.clip(X, self.min_clip_value, self.max_clip_value, out=X)

        # Normalize:
        min_value = np.min(X)
        max_value = np.max(X)
        X = (X - min_value) / (max_value - min_value)

        # save indices:
        self.index_array[list_IDs_temp] = indices

        return X

    def create_pathches(self):
        patch_list = []
        rows, columns, depth = self.scan_data.shape
        for z in range(depth):
            for y in range(0, rows, self.sampling_step):
                for x in range(0, columns, self.sampling_step):
                    if x + self.patch_size < columns and y + self.patch_size < rows:  # discard border patches
                        if self.is_patch_center_in_mask(x, y, z, self.organ_segmentation):
                            patch = self.scan_data[y:y+self.patch_size, x:x+self.patch_size, z]
                            patch_list.append((patch, (x, y, z)))
        return patch_list

    def is_patch_center_in_mask(self, patch_x, patch_y, patch_z, roi_mask):
        patch_center_x = patch_x + self.patch_size // 2
        patch_center_y = patch_y + self.patch_size // 2
        if roi_mask[patch_center_y, patch_center_x, patch_z] == 0:
            return False
        else:
            return True


