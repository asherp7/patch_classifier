import cv2
import numpy as np
import imgaug as ia
import tensorflow as tf
import imgaug.augmenters as iaa
from keras.backend.tensorflow_backend import set_session

def limit_gpu_memory(memory_fraction, gpu_serial_number='0'):
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = gpu_serial_number
    config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
    set_session(tf.Session(config=config))


seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Flipud(0.5),  # vertical flips
    # iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
                  iaa.GaussianBlur(sigma=(0, 0.5))
                  ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255)),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2))
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    # iaa.Affine(
    #     # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #     # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    #     rotate=(-25, 25),
    #     # shear=(-8, 8)
    # )
], random_order=True) # apply augmenters in random order


def augment_batch(X):
    return seq(images=X)


def custom_augment_img(img):
    thresholds = {'fliplr': 0.5,
                  'flipud': 0.5,
                  'gaussian_blur': 0.5,
                  'multiply': 0.5,
                  'linear_contrast': 0.5
                  }
    do_aug = np.random.uniform(0, 1, len(thresholds))
    if do_aug[0] > thresholds['fliplr']:
        img = np.fliplr(img)
    if do_aug[1] > thresholds['flipud']:
        img = np.flipud(img)
    if do_aug[2] > thresholds['gaussian_blur']:
        std = np.random.uniform(0, 3)
        img = cv2.GaussianBlur(img, (3, 3), std)
    if do_aug[3] > thresholds['multiply']:
        alpha = np.random.uniform(0.8, 1.2)
        img = alpha * img
    if do_aug[4] > thresholds['linear_contrast']:
        mean = np.mean(img)
        beta = np.random.uniform(0.5, 1.5)
        img = mean + beta * (img - mean)
    return img