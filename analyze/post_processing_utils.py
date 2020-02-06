import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
import random
import cv2


def remove_small_connected_componenets_3D(arr, min_size):
    """
    Remove 3D connected components for components smaller than min_size
    :param arr: 3-dimensional array
    :param min_size: minimum size of pixels we want to keep
    :return: 3d array from which small connected components were removed
    """
    structure = ndimage.morphology.generate_binary_structure(arr.ndim, arr.ndim)
    labeled_array, num_components = ndimage.label(arr, structure)
    filtered_arr = np.zeros(arr.shape, arr.dtype)
    for label in range(num_components):
        component = labeled_array == label + 1
        if np.count_nonzero(component) >= min_size:
            filtered_arr[component] = 1
    return filtered_arr


def remove_small_connected_componenets_from_slice(img, min_size):
    """

    :param img:
    :param min_size: minimum size of pixels we want to keep
    :return:
    """
    # if img has no components, then just return image
    if img.max() == 0:
        return img
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size.
    # the following part is just taking out the background which is also considered a component, but most of the time
    # we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # result clean image
    filtered_img = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            filtered_img[output == i + 1] = 1
    return filtered_img


def remove_small_connected_components_from_all_slices(arr, min_size):
    result_arr = np.zeros(arr.shape, dtype=arr.dtype)
    for z in range(arr.shape[-1]):
        result_arr[:, :, z] = remove_small_connected_componenets_from_slice(arr[:, :, z], min_size)
    return result_arr


def threshold_probability_map(probability_map, threshold):
    return (probability_map >= threshold).astype(probability_map.dtype)


def save_mask_after_removing_small_connected_components(mask_filepath, output_file_path, min_size, spatial=True):
    mask = nib.load(mask_filepath)
    mask_data = mask.get_data().astype(np.uint8)
    if spatial:
        filtered_mask = remove_small_connected_componenets_3D(mask_data, min_size)
    else:
        filtered_mask = remove_small_connected_components_from_all_slices(mask_data, min_size)
    save_data_as_new_nifti_file(mask_filepath, filtered_mask, output_file_path)


def save_probability_map_as_thresholded_mask(path_to_probability_map, mask_output_filepath, threshold):
    probability_map = nib.load(path_to_probability_map)
    mask = threshold_probability_map(probability_map.get_data(), threshold)
    save_data_as_new_nifti_file(path_to_probability_map, mask, mask_output_filepath)


def save_data_as_new_nifti_file(old_filepath, new_data, new_filepath):
    old_file = nib.load(old_filepath)
    new_file = nib.Nifti1Image(new_data, old_file.affine)
    nib.save(new_file, new_filepath)
    print('saved file to:', new_filepath)


# if __name__ == '__main__':
#     output_dir = '/cs/labs/josko/asherp7/follow_up/outputs/'
#     path_to_probability_map = os.path.join(output_dir, 'BL11.nii.gz')
#     mask_output_filepath = os.path.join(output_dir, 'BL11_predicted_mask.nii.gz')
#     threshold = 0.933
#     save_probability_map_as_thresholded_mask(path_to_probability_map, mask_output_filepath, threshold)

# if __name__ == '__main__':
#     output_dir = '/cs/labs/josko/asherp7/follow_up/outputs/'
#     mask_filepath = os.path.join(output_dir, 'BL11_predicted_mask.nii.gz')
#     clean_mask_output_file_path = os.path.join(output_dir, 'BL11_predicted_clean_mask.nii.gz')
#     path_to_tumor_segmentation = '/cs/labs/josko/asherp7/example_cases/case11/BL/BL11_Tumors.nii.gz'
#     # for min_size in np.linspace(25, 500, num=10):
#     for min_size in [400]:
#         save_mask_after_removing_small_connected_components(mask_filepath, clean_mask_output_file_path, min_size)
#         annotation = nib.load(path_to_tumor_segmentation).get_data()
#         prediction = nib.load(clean_mask_output_file_path).get_data()
#         dice_score = segmentations_dice(prediction, annotation)
#         print('min component size:', min_size,  ', Dice score:', dice_score)


if __name__ == '__main__':
    height = 256
    width = 256
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(10):
        color = (255, 255, 255)
        radius = random.randint(1, 30)
        center = (random.randint(0, width), random.randint(0,height))
        cv2.circle(img, center, radius, color, thickness=-1, lineType=8, shift=0)
    binary_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_3d = np.stack((binary_img,binary_img))
    plt.imshow(binary_img)
    plt.show()
    img2 = remove_small_connected_componenets_3D(image_3d, min_size=2000)
    plt.imshow(img2[0])
    plt.show()

