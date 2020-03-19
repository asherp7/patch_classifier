from skimage.filters import threshold_local, threshold_otsu
from scipy import ndimage
import nibabel as nib
import numpy as np
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


def adaptive_threshold_probability_map(probability_map, block_size = 35,  offset=10):
    adaptive_thresh = threshold_local(probability_map, block_size, offset=offset)
    return (probability_map >= adaptive_thresh).astype(probability_map.dtype)


def apply_otsu_threshold_on_probability_map(probability_map):
    # transform to uint8:
    # uint8_probability_map = (255 * probability_map).astype(np.uint8)
    # threshold, prediction = cv2.threshold(uint8_probability_map, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    threshold = threshold_otsu(probability_map)
    prediction = (probability_map >= threshold).astype(probability_map.dtype)
    return threshold, prediction


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


def save_probability_map_as_thresholded_mask(path_to_probability_map, mask_output_filepath, threshold=None):
    probability_map = nib.load(path_to_probability_map)
    if threshold:
        mask = threshold_probability_map(probability_map.get_data(), threshold)
    else:
        mask = apply_otsu_threshold_on_probability_map(probability_map.get_data())
    save_data_as_new_nifti_file(path_to_probability_map, mask, mask_output_filepath)


def save_data_as_new_nifti_file(old_filepath, new_data, new_filepath, verbose=False):
    old_file = nib.load(old_filepath)
    new_file = nib.Nifti1Image(new_data, old_file.affine)
    nib.save(new_file, new_filepath)
    if verbose:
        print('saved file to:', new_filepath)




