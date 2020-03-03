import skimage
from skimage.util import montage
from train.unet_generator import DataGenerator
import matplotlib.pyplot as plt

train_h5_path = '/mnt/local/aszeskin/asher/liver_data/unet_BL_all_patches.h5'
validation_h5_path = '/mnt/local/aszeskin/asher/liver_data/unet_FU_patches.h5'
batch_size = 16
do_augmentations = False
dim = (64, 64, 1)
gen = DataGenerator(train_h5_path, 'patches', 'mask_patches', do_augmentations, batch_size=batch_size, dim=dim)
a = gen[2]
print(a[0][0].shape)
plt.imshow(montage(a[0].reshape([-1, 64, 64])), cmap='gray')
plt.title('patches')
# plt.imsave('patch_montage.png', montage(a[0].reshape([-1,35,35])))
plt.show()

plt.imshow(montage(a[1].reshape([-1, 64, 64])), cmap='gray')
plt.title('mask')
plt.show()
