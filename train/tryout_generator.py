import skimage
from skimage.util import montage
from train.generator import DataGenerator
import matplotlib.pyplot as plt

train_h5_path = '/mnt/local/aszeskin/asher/liver_data/BL_patches.h5'
validation_h5_path = '/mnt/local/aszeskin/asher/liver_data/FU_patches.h5'
gen = DataGenerator(train_h5_path, 'patches', 'labels', do_augmentations=True)
a = gen[0]
print(a[0][0].shape)
print(a[0].dtype)
print(a[0])
plt.imshow(montage(a[0].reshape([-1,35,35])), cmap='gray')
# plt.imsave('patch_montage.png', montage(a[0].reshape([-1,35,35])))
plt.show()



# import h5py
# data_path = '/mnt/local/aszeskin/asher/transforfm_h5_output'
# f = h5py.File('/mnt/local/aszeskin/asher/transforfm_h5_output/patches.h5', 'r')
# print(f.keys())
# print(f['patches'])
# print(f['labels'])
# print(f['tumor_idx'])
# print(f['non_tumor_idx'])
# n_patches = f['patches'].shape[0]
