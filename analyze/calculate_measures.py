import os
import csv
import numpy as np
from scipy import ndimage
import nibabel as nib


def dice(gt_seg, estimated_seg):
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


def compute_TP_detections(pred, gt, min_diameter=0):
    tp_mask = np.logical_and(pred, gt)
    structure = ndimage.morphology.generate_binary_structure(tp_mask.ndim, tp_mask.ndim)
    labeled_array, num_components = ndimage.label(tp_mask, structure)
    tp_counter = 0
    tp_arr = np.zeros(pred.shape)
    for label in range(num_components):
        component = labeled_array == label + 1
        tumor_volume = np.count_nonzero(component)
        if approximate_diameter(tumor_volume) > min_diameter:
            tp_counter += 1
            tp_arr[component] = 1
    return tp_counter, tp_arr


def compute_FP_detections(pred, gt, min_diameter=0):
    structure = ndimage.morphology.generate_binary_structure(pred.ndim, pred.ndim)
    labeled_array, num_components = ndimage.label(pred, structure)
    fp_counter = 0
    fp_arr = np.zeros(pred.shape)
    for label in range(num_components):
        component = labeled_array == label + 1
        tumor_volume = np.count_nonzero(component)
        if approximate_diameter(tumor_volume) > min_diameter:
            if not np.any(np.logical_and(component, gt)):
                fp_counter += 1
                fp_arr[component] = 1
    return fp_counter, fp_arr


def approximate_diameter(tumor_volume):
    r = ((.75 * tumor_volume) / np.pi) ** (1/3)
    diameter = 2 * r
    return diameter


def bbox_3D(arr):
    """

    :param arr:
    :return:
    """
    r = np.any(arr, axis=(1, 2))
    c = np.any(arr, axis=(0, 2))
    z = np.any(arr, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax


def get_bbox_mean_diameter(arr):
    rmin, rmax, cmin, cmax, zmin, zmax = bbox_3D(arr)
    return ((rmax - rmin) + (cmax - cmin) + (zmax - zmin)) / 3


def compute_tumors_sizes_and_FN(pred, gt, voxel_volume, min_diameter, min_gt_voxels):
    structure = ndimage.morphology.generate_binary_structure(gt.ndim, gt.ndim)
    labeled_array, num_components = ndimage.label(gt, structure)
    tumor_sizes = []
    FN_counter = 0
    fn_arr = np.zeros(pred.shape)
    tumor_filtered_arr = np.zeros(pred.shape)
    for label in range(num_components):
        component = labeled_array == label + 1
        num_voxels = np.count_nonzero(component)
        tumor_volume = num_voxels * voxel_volume
        approx_diameter = approximate_diameter(tumor_volume)
        tumor_sizes.append({"voxels": num_voxels,
                            "volume": tumor_volume,
                            "bbox_diameter": get_bbox_mean_diameter(component),
                            "approximate_diameter": approx_diameter})
        if approx_diameter > min_diameter and num_voxels > min_gt_voxels:
            tumor_filtered_arr[component] = 1
            if not np.any(np.logical_and(pred, component)):
                FN_counter += 1
                fn_arr[component] = 1

    return tumor_sizes, FN_counter, fn_arr, tumor_filtered_arr


def compute_detection_measures(pred, gt, voxel_volume, min_diameter, min_gt_voxels=10):
    print('computing TP')
    TP_num, TP_pred = compute_TP_detections(pred, gt, min_diameter)
    print("TP_num:", TP_num)
    print('\ncomputing FP')
    FP_num, FP_pred = compute_FP_detections(pred, gt, min_diameter)
    print("FP_num:", FP_num)
    print('\ncomputing FN')
    tumor_sizes, FN_num, FN_pred, tumor_filtered_arr = \
        compute_tumors_sizes_and_FN(pred, gt, voxel_volume, min_diameter, min_gt_voxels)
    print("\nFN_num", FN_num)
    num_tumors = len([x for x in tumor_sizes if x["approximate_diameter"] > min_diameter and x["volume"] > min_gt_voxels])
    print("\nTumors", len(tumor_sizes), ', Tumors with more than', min_gt_voxels, 'voxels:', num_tumors)
    if num_tumors == 0:
        recall = np.nan
        precision = np.nan
    elif TP_num == 0:
        recall = 0
        precision = 0
    else:
        recall = TP_num / (TP_num + FN_num)
        precision = TP_num / (TP_num + FP_num)
    # print(tumor_sizes)
    print("\nrecall", recall)
    print("\nprecision", precision)
    TP_dice = np.count_nonzero(TP_pred) * voxel_volume
    FP_dice = np.count_nonzero(FP_pred) * voxel_volume
    FN_dice = np.count_nonzero(FN_pred) * voxel_volume
    if num_tumors == 0:
        dice_score = np.nan
    else:
        dice_score = dice(pred, tumor_filtered_arr)
    return [num_tumors, FN_num, TP_num, FP_num, recall, precision, TP_dice, FP_dice, FN_dice, dice_score]


def compute_folder_detection_measures(pred_folder, data_root_path):
    min_diameter_lists, num_tumors_lists, FN_num_lists, TP_num_lists, FP_num_lists, recall_lists, precision_lists = {}, {}, {}, {}, {}, {}, {}
    TP_volume_lists, FP_volume_lists, FN_volume_lists, dice_score_lists = {}, {}, {}, {}
    for min_diameter in [0, 5, 10]:
        min_diameter_lists[str(min_diameter)] = []
        num_tumors_lists[str(min_diameter)] = []
        FN_num_lists[str(min_diameter)] = []
        TP_num_lists[str(min_diameter)] = []
        FP_num_lists[str(min_diameter)] = []
        recall_lists[str(min_diameter)] = []
        precision_lists[str(min_diameter)] = []
        # dice scores:
        TP_volume_lists[str(min_diameter)] = []
        FP_volume_lists[str(min_diameter)] = []
        FN_volume_lists[str(min_diameter)] = []
        dice_score_lists[str(min_diameter)] = []
    with open('detection_measures.csv', mode='w') as measures_file:
        file_writer = csv.writer(measures_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(['CASE', 'MIN DIAMETER', 'TUMOR NUMBER', 'FN', 'TP', 'FP', 'recall', 'precision'])
        for pred_filename in [x for x in os.listdir(pred_folder) if x.endswith('nii.gz')]:
            if not pred_filename.endswith('.nii.gz'):
                continue
            print('*' * 60)
            print(pred_filename)
            print('*' * 60)
            pred_filepath = os.path.join(pred_folder, pred_filename)
            liver_seg_path = os.path.join(data_root_path, 'liver_seg', pred_filename.replace('.nii.gz', '_liverseg.nii.gz'))
            tumor_seg_path = os.path.join(data_root_path, 'tumors', pred_filename.replace('.nii.gz', '_Tumors.nii.gz'))
            pred_file = nib.load(pred_filepath)
            pix_dims = pred_file.header.get_zooms()
            voxel_volume = pix_dims[0] * pix_dims[1] * pix_dims[2]
            pred = pred_file.get_data()
            liver_seg = nib.load(liver_seg_path).get_data()
            tumor_seg = nib.load(tumor_seg_path).get_data()
            fixed_tumor_seg = np.logical_and(tumor_seg, liver_seg)

            for min_diameter in [0, 5, 10]:
                print('*' * 30)
                print('DIAMETER:', min_diameter)
                print('*' * 30)
                [num_tumors, FN_num, TP_num, FP_num, recall, precision, TP_vol, FP_vol, FN_vol, dice_score] = \
                    compute_detection_measures(pred, fixed_tumor_seg, voxel_volume, min_diameter)
                print('\n\n')
                file_writer.writerow([pred_filename, min_diameter, num_tumors, FN_num, TP_num, FP_num, recall, precision])
                min_diameter_lists[str(min_diameter)].append(min_diameter)
                num_tumors_lists[str(min_diameter)].append(num_tumors)
                FN_num_lists[str(min_diameter)].append(FN_num)
                TP_num_lists[str(min_diameter)].append(TP_num)
                FP_num_lists[str(min_diameter)].append(FP_num)
                recall_lists[str(min_diameter)].append(recall)
                precision_lists[str(min_diameter)].append(precision)
                # append dice lists:
                TP_volume_lists[str(min_diameter)].append(TP_vol)
                FP_volume_lists[str(min_diameter)].append(FP_vol)
                FN_volume_lists[str(min_diameter)].append(FN_vol)
                dice_score_lists[str(min_diameter)].append(dice_score)
        for min_diameter in [0, 5, 10]:
            file_writer.writerow(['Mean:',
                                  min_diameter,
                                  np.nanmean(num_tumors_lists[str(min_diameter)]),
                                  np.nanmean(FN_num_lists[str(min_diameter)]),
                                  np.nanmean(TP_num_lists[str(min_diameter)]),
                                  np.nanmean(FP_num_lists[str(min_diameter)]),
                                  np.nanmean(recall_lists[str(min_diameter)]),
                                  np.nanmean(precision_lists[str(min_diameter)])])

    with open('segmentation_measures.csv', mode='w') as measures_file:
        file_writer = csv.writer(measures_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(['CASE', 'MIN DIAMETER', 'NUM TUMORS', 'DICE', 'TP VOLUME', 'FN VOLUME', 'FP VOLUME'])
        for idx, pred_filename in enumerate([x for x in os.listdir(pred_folder) if x.endswith('nii.gz')]):
            for min_diameter in [0, 5, 10]:
                file_writer.writerow([pred_filename,
                                      min_diameter,
                                      num_tumors_lists[str(min_diameter)][idx],
                                      dice_score_lists[str(min_diameter)][idx],
                                      TP_volume_lists[str(min_diameter)][idx],
                                      FN_volume_lists[str(min_diameter)][idx],
                                      FP_volume_lists[str(min_diameter)][idx]
                                      ])

        for min_diameter in [0, 5, 10]:
            file_writer.writerow(['Mean:',
                                  min_diameter,
                                  np.nanmean(num_tumors_lists[str(min_diameter)]),
                                  np.nanmean(dice_score_lists[str(min_diameter)]),
                                  np.nanmean(TP_volume_lists[str(min_diameter)]),
                                  np.nanmean(FN_volume_lists[str(min_diameter)]),
                                  np.nanmean(FP_volume_lists[str(min_diameter)])])


if __name__ == '__main__':
    data_path = '/cs/labs/josko/asherp7/follow_up/data_3_4_2020'
    pred_path = '/cs/labs/josko/asherp7/follow_up/outputs/validation_cnn_predictions_5_4_2020_2020-04-05_11-13-12/selection'
    # data_path = '/cs/labs/josko/asherp7/follow_up/data_31_3_2020'
    # pred_path = '/cs/labs/josko/asherp7/follow_up/outputs/validation_cnn_predictions_1_4_2020_2020-04-01_01-57-47/selection'
    # pred_path = '/cs/labs/josko/asherp7/follow_up/outputs/validation_cnn_predictions_1_4_2020_2020-04-01_01-57-47/threshold_cnn_predictions'
    # data_path = '/cs/labs/josko/asherp7/follow_up/validation_combined_data'
    # pred_path = '/cs/labs/josko/asherp7/follow_up/outputs/pred_2020-03-26_10-20-24/selection'
    compute_folder_detection_measures(pred_path, data_path)
