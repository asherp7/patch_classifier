import skimage
from skimage.util import montage
from train.generator import DataGenerator
import matplotlib.pyplot as plt

train_h5_path = '/mnt/local/aszeskin/asher/liver_data/BL_patches.h5'
validation_h5_path = '/mnt/local/aszeskin/asher/liver_data/FU_patches.h5'
gen = DataGenerator(train_h5_path, 'patches', 'labels', do_augmentations=True)
a = gen[0]
print(a[0][0].shape)
print('labels:')
print(a[1])
plt.imshow(montage(a[0].reshape([-1,35,35])), cmap='gray')
# plt.imsave('patch_montage.png', montage(a[0].reshape([-1,35,35])))
plt.show()

