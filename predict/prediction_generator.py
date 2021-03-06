import nibabel as nib
import numpy as np
import keras


class PredictionGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, scan_filepath, organ_segmentation_filepath, batch_size=32, patch_size=(35, 35, 1),
                 min_clip_value=-100, max_clip_value=150, sampling_step=1):
        'Initialization'
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.patch_center_y = self.patch_size[0] // 2
        self.patch_center_x = self.patch_size[1] // 2
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
        X = np.empty((self.batch_size, *self.patch_size))
        indices = np.empty((self.batch_size, 3))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            patch, (y, x, z) = self.patch_list[ID]
            X[i, ] = np.expand_dims(patch, axis=-1)
            indices[i, ] = np.asarray([y,
                                       x,
                                       z])
        # Clip values:
        np.clip(X, self.min_clip_value, self.max_clip_value, out=X)

        # Normalize:
        X = (X - self.min_clip_value) / (self.max_clip_value - self.min_clip_value)

        # save indices:
        # Please notice that x,y,z in numpy array is not the same as nifti. for the generation of the
        # nifti file it should be y , x ,z
        self.index_array[list_IDs_temp] = indices

        return X

    def create_pathches(self):
        patch_list = []
        rows, columns, depth = self.scan_data.shape
        for z in range(depth):
            for y in range(-self.patch_center_y, rows-self.patch_center_y, self.sampling_step):
                for x in range(-self.patch_center_x, columns-self.patch_center_x, self.sampling_step):
                    # Adding to left corner indices: patch_size // 2:
                    # so we will get the middle of the patch instead of the left upper corner of the patch.
                    y_center = y + self.patch_center_y
                    x_center = x + self.patch_center_x
                    if self.is_patch_center_in_mask(x_center, y_center, z, self.organ_segmentation):
                        patch = np.zeros((self.patch_size[0], self.patch_size[1]))
                        y_min = max(0, -y)
                        y_max = patch.shape[0] - max(0, y + self.patch_size[0] - self.scan_data.shape[0])
                        x_min = max(0, -x)
                        x_max = patch.shape[1] - max(0, x + self.patch_size[1] - self.scan_data.shape[1])
                        patch[y_min:y_max, x_min:x_max] = self.scan_data[max(0, y):y + self.patch_size[0],
                                                          max(0, x):x + self.patch_size[1], z]
                        patch_list.append((patch, (y_center, x_center, z)))
        return patch_list

    def is_patch_center_in_mask(self, patch_center_x, patch_center_y, patch_z, roi_mask):
        if roi_mask[patch_center_y, patch_center_x, patch_z] == 0:
            return False
        else:
            return True


