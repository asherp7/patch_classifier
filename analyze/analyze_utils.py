from analyze.post_processing_utils import remove_small_connected_componenets_3D, \
    save_mask_after_removing_small_connected_components, save_probability_map_as_thresholded_mask,\
    adaptive_threshold_probability_map, apply_otsu_threshold_on_probability_map, threshold_probability_map, \
    save_data_as_new_nifti_file
from scipy.ndimage import morphology
import nibabel as nib
import numpy as np
import json
import os

# def segmentations_dice(segmentation_1, segmentation_2):
#     n1 = np.count_nonzero(segmentation_1)
#     n2 = np.count_nonzero(segmentation_2)
#     intersection = np.logical_and(segmentation_1, segmentation_2)
#     n_intersection = np.count_nonzero(intersection)
#     dice = 2*n_intersection / (n1 + n2)
#     return dice


def segmentations_dice(gt_seg, estimated_seg):
    """
    compute dice coefficient
    :param gt_seg:
    :param estimated_seg:
    :return:
    """
    seg1 = np.asarray(gt_seg).astype(np.bool)
    seg2 = np.asarray(estimated_seg).astype(np.bool)

    # Compute Dice coefficient
    intersection = np.logical_and(seg1, seg2)

    return 2. * intersection.sum() / (seg1.sum() + seg2.sum())


def segmentations_assd(segmentation_1, segmentation_2):
    assd = None
    return assd


def segmentations_voe(segmentation_1, segmentation_2):
    voe = None
    return voe


def get_ct_liver_tumor_filepaths_list(ct_dir_path, roi_dir_path, tumor_dir_path, prediction_dir_path,
                                      tumor_suffix='_newTumors_copy', roi_suffix='_LiverSeg'):
    file_names_list = []
    for filename in os.listdir(ct_dir_path):
        tumor_file_path = os.path.join(tumor_dir_path, filename.replace('.nii', tumor_suffix+'.nii'))
        if filename.startswith('BL'):
            extension = '.nii'
        else:
            extension = '.nii'
        # roi_file_path = os.path.join(roi_dir_path, filename.replace('.nii.gz', roi_suffix+extension))
        roi_file_path = os.path.join(roi_dir_path, filename.replace('.nii', roi_suffix+extension))
        if not os.path.isfile(roi_file_path):
            print(roi_file_path, 'is missing!')
            continue
        if not os.path.isfile(tumor_file_path):
            print(tumor_file_path, 'is missing!')
            continue
        scan_file_path = os.path.join(ct_dir_path, filename)
        prediction_file_path = os.path.join(prediction_dir_path, filename).replace('.nii', '_chanvese_seg_expand.nii')
        file_names_list.append((scan_file_path, roi_file_path, tumor_file_path, prediction_file_path))
    return file_names_list


def analyze_dataset(ct_dir_path, roi_dir_path, tumor_dir_path, prediction_dir_path):
    dice_loss_dict = {}
    file_paths = get_ct_liver_tumor_filepaths_list(ct_dir_path, roi_dir_path, tumor_dir_path, prediction_dir_path)
    for idx, (ct_path, roi_path, tumor_path, pred_path) in enumerate(file_paths, 1):
        filename = os.path.basename(ct_path)
        gt = nib.load(tumor_path).get_data()
        prediction = nib.load(pred_path).get_data()
        dice_loss_dict[filename] = segmentations_dice(gt, prediction)
        print(idx, '/', len(file_paths), filename, ', dice:', dice_loss_dict[filename])
    return dice_loss_dict


def analyze_dataset_after_threshold_and_filter_small_components(ct_dir_path,
                                                                 roi_dir_path,
                                                                 tumor_dir_path,
                                                                 prediction_dir_path,
                                                                 min_size,
                                                                 threshold=None,
                                                                 save_path=None):
    dice_loss_dict = {}
    if threshold:
        apply_otsu = False
    else:
        apply_otsu = True
    file_paths = get_ct_liver_tumor_filepaths_list(ct_dir_path, roi_dir_path, tumor_dir_path, prediction_dir_path, roi_suffix='_liverseg')
    for idx, (ct_path, roi_path, tumor_path, pred_path) in enumerate(file_paths, 1):
        filename = os.path.basename(ct_path)
        annotation = nib.load(tumor_path).get_data()
        probabilty_map = nib.load(pred_path).get_data()
        if apply_otsu:
            threshold, prediction = apply_otsu_threshold_on_probability_map(probabilty_map)
        else:
            prediction = threshold_probability_map(probabilty_map, threshold)
        filtered_prediction = remove_small_connected_componenets_3D(prediction, min_size)
        fill_holes = morphology.binary_fill_holes(filtered_prediction, np.ones((3, 3, 3)))
        case_name = os.path.basename(ct_path)
        threshold_dice_loss = segmentations_dice(prediction, annotation)
        filtered_dice_loss = segmentations_dice(filtered_prediction, annotation)
        fill_holes_dice_loss = segmentations_dice(fill_holes, annotation)
        dice_loss_dict[filename] = {"threshold": threshold_dice_loss,
                                    "filtered": filtered_dice_loss,
                                    "fill": fill_holes_dice_loss}
        print(idx, '/', len(file_paths), case_name, ', threshold:', round(threshold, 2), ', threshold dice: ',
              round(threshold_dice_loss, 2), ", filtered dice loss", round(filtered_dice_loss, 2),
              ', fill holes dice:', round(fill_holes_dice_loss, 2))
        if save_path:
            filtered_output_file_path = os.path.join(save_path, 'filtered_'+case_name)
            mask_output_filepath = os.path.join(save_path, 'threshold_'+case_name)
            save_data_as_new_nifti_file(pred_path, filtered_prediction, filtered_output_file_path)
            save_data_as_new_nifti_file(pred_path, prediction, mask_output_filepath)
    print('threshold mean dice:', round(sum([x["threshold"] for x in dice_loss_dict.values()]) / len(dice_loss_dict), 3))
    print('filtered mean dice:', round(sum([x["filtered"] for x in dice_loss_dict.values()]) / len(dice_loss_dict), 3))
    print('fill holes mean dice:', round(sum([x["fill"] for x in dice_loss_dict.values()]) / len(dice_loss_dict), 3))
    return dice_loss_dict


def get_data_split(output_dir_path):
    data_split_filepath = os.path.join(output_dir_path, 'data_split.json')
    with open(data_split_filepath, 'r') as fp:
        data_split = json.load(fp)
        return data_split



        