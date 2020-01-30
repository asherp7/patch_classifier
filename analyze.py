from post_processing import remove_small_connected_componenets_3D
import nibabel as nib
import numpy as np
import os

def segmentations_dice(segmentation_1, segmentation_2):
    n1 = np.count_nonzero(segmentation_1)
    n2 = np.count_nonzero(segmentation_2)
    intersection = np.logical_and(segmentation_1, segmentation_2)
    n_intersection = np.count_nonzero(intersection)
    # print('n1:', n1)
    # print('n2:', n2)
    # print('n_intersection:', n_intersection)
    dice = 2*n_intersection / (n1 + n2)
    return dice


def segmentations_assd(segmentation_1, segmentation_2):
    assd = None
    return assd


def segmentations_voe(segmentation_1, segmentation_2):
    voe = None
    return voe


def get_ct_liver_tumor_filepaths_list(ct_dir_path, roi_dir_path, tumor_dir_path, prediction_dir_path,
                                      tumor_suffix='_Tumors', roi_suffix='_liverseg'):
    file_names_list = []
    for filename in os.listdir(ct_dir_path):
        tumor_file_path = os.path.join(tumor_dir_path, filename.replace('.nii.gz', tumor_suffix+'.nii.gz'))
        if filename.startswith('BL'):
            extension = '.nii.gz'
        else:
            extension = '.nii'
        roi_file_path = os.path.join(roi_dir_path, filename.replace('.nii.gz', roi_suffix+extension))
        if not os.path.isfile(roi_file_path):
            print(roi_file_path, 'is missing!')
            continue
        if not os.path.isfile(tumor_file_path):
            print(tumor_file_path, 'is missing!')
            continue
        scan_file_path = os.path.join(ct_dir_path, filename)
        prediction_file_path = os.path.join(prediction_dir_path, filename)
        file_names_list.append((scan_file_path, roi_file_path, tumor_file_path, prediction_file_path))
    return file_names_list


def analyze_dataset(ct_dir_path, roi_dir_path, tumor_dir_path, prediction_dir_path, threshold, min_size):
    file_paths = get_ct_liver_tumor_filepaths_list(ct_dir_path, roi_dir_path, tumor_dir_path, prediction_dir_path)
    for idx, (ct_path, roi_path, tumor_path, pred_path) in enumerate(file_paths, 1):
        annotation = nib.load(tumor_path).get_data()
        probabilty_map = nib.load(pred_path).get_data()
        prediction = (probabilty_map >= threshold)
        remove_small_connected_componenets_3D(prediction, min_size)
        prediction = (probabilty_map >= thresh)
        case_name = os.path.basename(ct_path)
        print(idx, '/', len(file_paths), case_name, ', threshold:', thresh, 'dice: ', segmentations_dice(prediction, annotation))


if __name__ == '__main__':
    output_path = '/cs/labs/josko/asherp7/follow_up/outputs'
    path_to_prediction = os.path.join(output_path, 'BL11.nii.gz')
    path_to_tumor_segmentation = '/cs/labs/josko/asherp7/example_cases/case11/BL/BL11_Tumors.nii.gz'
    probabilty_map = nib.load(path_to_prediction).get_data()
    annotation = nib.load(path_to_tumor_segmentation).get_data()
    for thresh in np.linspace(0.7, 1, num=10):
        prediction = (probabilty_map >= thresh)
        print('threshold:', thresh, 'dice: ', segmentations_dice(prediction, annotation))

        