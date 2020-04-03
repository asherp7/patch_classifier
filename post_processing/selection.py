import os
import sys
if __name__ == '__main__':
    sys.path.insert(0, os.getcwd())

from analyze.post_processing_utils import remove_small_connected_componenets_3D, save_data_as_new_nifti_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_multilabel_classification
from scipy import ndimage
import nibabel as nib
import numpy.ma as ma
import numpy as np
from analyze.analyze_utils import get_data_split



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


def get_scan_connected_componenets_features_and_gt(ct_data,
                                                   cnn_pred_data,
                                                   thresholded_cnn_data,
                                                   tumor_data,
                                                   cc_gt_intersection_thresh=0.5):
    feature_list = []
    gt_list = []
    structure = ndimage.morphology.generate_binary_structure(thresholded_cnn_data.ndim, thresholded_cnn_data.ndim)
    labeled_array, num_components = ndimage.label(thresholded_cnn_data, structure)
    print('num_components', num_components)
    for label in range(num_components):
        component = labeled_array == label + 1
        volume = component.sum()
        rmin, rmax, cmin, cmax, zmin, zmax = bbox_3D(component)
        row_len = rmax - rmin
        col_len = cmax - cmin
        depth_len = zmax - zmin
        bb_volume = row_len * col_len * depth_len
        if bb_volume == 0:
            bb_occupation = 10000
        else:
            bb_occupation = volume / bb_volume
        ct_component = ct_data[component]
        ct_mean = np.sum(ct_component)
        ct_min = np.min(ct_component)
        ct_max = np.max(ct_component)
        ct_std = np.std(ct_component)
        cnn_component = cnn_pred_data[component]
        cnn_mean = np.mean(cnn_component)
        cnn_min = np.min(cnn_component)
        cnn_max = np.max(cnn_component)
        cnn_std = np.std(cnn_component)
        features = [volume, bb_volume, bb_occupation, ct_mean, ct_min, ct_max, ct_std, cnn_mean, cnn_min, cnn_max, cnn_std]
        names = ['volume', 'bb_volume', 'bb_occupation', 'ct_mean', 'ct_min', 'ct_max', 'ct_std', 'cnn_mean', 'cnn_min', 'cnn_max', 'cnn_std']

        cc_gt_intersection = np.sum(np.logical_and(tumor_data, component)) / volume
        if cc_gt_intersection == 0:
            cc_label = 0
            gt_list.append(cc_label)
            feature_list.append(features)
            # print(0, list(zip(names, features)))
        elif cc_gt_intersection > cc_gt_intersection_thresh:
            cc_label = 1
            gt_list.append(cc_label)
            feature_list.append(features)
            # print(1, list(zip(names, features)))
        else:
            print('skipped:', cc_gt_intersection, 'intersection')
    return feature_list, gt_list


def get_scan_connected_componenets_gt(chan_vase_data,
                                      tumor_data,
                                      min_cc_size=10,
                                      cc_gt_intersection_thresh=0.2):
    gt_list = []
    structure = ndimage.morphology.generate_binary_structure(chan_vase_data.ndim, chan_vase_data.ndim)
    labeled_array, num_components = ndimage.label(chan_vase_data, structure)
    for label in range(num_components):
        component = labeled_array == label + 1
        volume = component.sum()
        if volume < min_cc_size:
            continue
        cc_gt_intersection = np.sum(np.logical_and(tumor_data, component)) / volume
        if cc_gt_intersection > cc_gt_intersection_thresh:
            cc_label = 1
        else:
            cc_label = 0
        gt_list.append(cc_label)
    return gt_list


def get_scan_connected_componenets_features_and_labeled_array(ct_data,
                                                              cnn_pred_data,
                                                              chan_vase_data):
    feature_list = []
    structure = ndimage.morphology.generate_binary_structure(chan_vase_data.ndim, chan_vase_data.ndim)
    labeled_array, num_components = ndimage.label(chan_vase_data, structure)
    print('num_components', num_components)
    for label in range(num_components):
        component = labeled_array == label + 1
        volume = component.sum()
        rmin, rmax, cmin, cmax, zmin, zmax = bbox_3D(component)
        row_len = rmax - rmin
        col_len = cmax - cmin
        depth_len = zmax - zmin
        bb_volume = row_len * col_len * depth_len
        if bb_volume == 0:
            bb_occupation = 10000
        else:
            bb_occupation = volume / bb_volume
        ct_component = ct_data[component]
        ct_mean = np.sum(ct_component)
        ct_min = np.min(ct_component)
        ct_max = np.max(ct_component)
        ct_std = np.std(ct_component)
        cnn_component = cnn_pred_data[component]
        cnn_mean = np.mean(cnn_component)
        cnn_min = np.min(cnn_component)
        cnn_max = np.max(cnn_component)
        cnn_std = np.std(cnn_component)
        features = [volume, bb_volume, bb_occupation, ct_mean, ct_min, ct_max, ct_std, cnn_mean, cnn_min, cnn_max, cnn_std]
        names = ['volume', 'bb_volume', 'bb_occupation', 'ct_mean', 'ct_min', 'ct_max', 'ct_std', 'cnn_mean', 'cnn_min', 'cnn_max', 'cnn_std']
        feature_list.append(features)
        # print(list(zip(names, features)))
    return feature_list, labeled_array, num_components


def transform_nifti_to_vector(nifti_filepath):
    data = nib.load(nifti_filepath).get_fdata()
    return data.reshape((data.size, 1))


def get_filenames(ct_dir_path, tumor_dir_path, cnn_pred_dir_path, chan_vase_dir_path):
    filenames = []
    for ct_filename in os.listdir(cnn_pred_dir_path):
        ct_path = os.path.join(ct_dir_path, ct_filename)
        tumor_path = os.path.join(tumor_dir_path, ct_filename.replace('.nii.gz', '_Tumors.nii.gz'))
        cnn_pred_path = os.path.join(cnn_pred_dir_path, ct_filename)
        chan_vese_path = os.path.join(chan_vase_dir_path, 'chanvese_seg_expand_'+ct_filename)
        filenames.append((ct_path, tumor_path, cnn_pred_path, chan_vese_path))
    return filenames


def create_random_forest_features(filenames):
    all_features_list = []
    for idx, (ct_filename, tumor_filename, cnn_pred_filename, chan_vase_filename) in enumerate(filenames):
        print('(', idx, '/', len(filenames), ') processing', os.path.basename(ct_filename))
        ct_data = nib.load(ct_filename).get_fdata()
        cnn_pred_data = nib.load(cnn_pred_filename).get_fdata()
        chan_vase_data = nib.load(chan_vase_filename).get_fdata()
        feature_list, _, _ = get_scan_connected_componenets_features_and_labeled_array(ct_data,
                                                                                       cnn_pred_data,
                                                                                       chan_vase_data)
        all_features_list.extend(feature_list)
    return np.array(all_features_list)


def create_random_forest_features_and_gt(filenames):
    all_features_list = []
    all_gt_list = []
    for idx, (ct_filename, roi_filepath, tumor_filename, cnn_pred_filename, threshold_cnn_filepath) in \
            enumerate(filenames, 1):
        print('(', idx, '/', len(filenames), ') processing', os.path.basename(ct_filename))
        ct_data = nib.load(ct_filename).get_fdata()
        cnn_pred_data = nib.load(cnn_pred_filename).get_fdata()
        thresholded_cnn_data = nib.load(threshold_cnn_filepath).get_fdata()
        roi = nib.load(roi_filepath).get_fdata()
        tumor_data = nib.load(tumor_filename).get_fdata()
        validated_tumor_data = np.logical_and(roi, tumor_data)
        feature_list, gt_list = get_scan_connected_componenets_features_and_gt(ct_data,
                                                                               cnn_pred_data,
                                                                               thresholded_cnn_data,
                                                                               validated_tumor_data)
        all_features_list.extend(feature_list)
        all_gt_list.extend(gt_list)
    return np.array(all_features_list), np.array(all_gt_list)


def create_random_forest_labels(filenames):
    all_gt_list = []
    for idx, (ct_filename, tumor_filename, _, chan_vase_filename) in enumerate(filenames):
        print('(', idx, '/', len(filenames), ') processing', os.path.basename(ct_filename))
        chan_vase_data = nib.load(chan_vase_filename).get_fdata()
        tumor_data = nib.load(tumor_filename).get_fdata()
        labels = get_scan_connected_componenets_gt(chan_vase_data, tumor_data)
        all_gt_list.append(labels)
    return np.array(all_gt_list)


def train_random_forest(num_estimatores, features, gt, class_weight=None):
    clf = RandomForestClassifier(n_estimators=num_estimatores, class_weight=class_weight, verbose=1)
    clf.fit(features, gt)
    return clf


def apply_selection_on_cnn_and_chan_vase(classifier, ct_data, cnn_pred_data, chan_vase_data):
    """

    :param classifier:
    :param cnn_pred_data:
    :param chan_vase_data:
    :return:
    """
    features, labeled_array, num_components = get_scan_connected_componenets_features_and_labeled_array(ct_data,
                                                                                                        cnn_pred_data,
                                                                                                        chan_vase_data)
    selected_arr = np.zeros(chan_vase_data.shape)
    predictions = classifier.predict(features)
    for label_idx in range(num_components):
        if predictions[label_idx]:
            component = labeled_array == label_idx + 1
            selected_arr[component] = 1
    return selected_arr


def apply_selection_on_dir(classifier, filenames, output_dir):
    for idx, (ct_filename, __, tumor_filename, cnn_pred_filename, chan_vase_filename) in enumerate(filenames, 1):
        print('(', idx, '/', len(filenames), ') processing', os.path.basename(ct_filename))
        ct_data = nib.load(ct_filename).get_fdata()
        cnn_pred_data = nib.load(cnn_pred_filename).get_fdata()
        chan_vase_data = nib.load(chan_vase_filename).get_fdata()
        selected = apply_selection_on_cnn_and_chan_vase(classifier, ct_data, cnn_pred_data, chan_vase_data)
        new_filepath = os.path.join(output_dir, os.path.basename(ct_filename))
        save_data_as_new_nifti_file(chan_vase_filename, selected, new_filepath)


def remove_small_components_and_save(old_dirpath, new_dirpath, cc_min_size=10):
    for filename in os.listdir(old_dirpath):
        filepath = os.path.join(old_dirpath, filename)
        noisy_data = nib.load(filepath).get_fdata()
        clean_mask = remove_small_connected_componenets_3D(noisy_data, cc_min_size)
        new_filepath = os.path.join(new_dirpath, filename)
        old_filepath = os.path.join(old_dirpath, filename)
        save_data_as_new_nifti_file(old_filepath, clean_mask, new_filepath)


def get_filepaths_from_data_split(data_dir_path, split, pred_path):
    data_split = get_data_split(data_dir_path)
    file_names_list = []
    for ct_filepath, roi_filepath, tumor_seg_filepath in data_split[split]:
        pred_filepath = os.path.join(pred_path, 'cnn_predictions', os.path.basename(ct_filepath))
        threshold_pred_filepath = os.path.join(pred_path, 'threshold_cnn_predictions', 'threshold_'+os.path.basename(ct_filepath))
        # chan_vese_filepath = os.path.join(pred_path, 'chan_vese_results', 'chanvese_seg_expand_' + os.path.basename(pred_filepath))
        file_names_list.append((ct_filepath, roi_filepath, tumor_seg_filepath, pred_filepath, threshold_pred_filepath))
    return file_names_list


def augment_feature(features, std, multiplier):
    """
    Augment features usin Gaussian distribution with std, stack with original features and return.
    :param features:
    :param std:
    :param multiplier:
    :return:
    """
    augmentations = np.random.normal(loc=features, scale=std, size=(multiplier, features.shape[0]))
    return np.vstack((augmentations, features))


def balance_data_by_augmentation(path_to_labels, path_to_features):
    labels = np.load(path_to_labels)
    feature_arr = np.load(path_to_features)
    std = np.std(feature_arr, axis=0)
    multiplier = np.count_nonzero(labels == 0) // np.count_nonzero(labels)
    num_augmeted_features = np.count_nonzero(labels) * (multiplier+1) + np.count_nonzero(labels == 0)
    augmented_features = np.empty((num_augmeted_features, feature_arr.shape[1]))
    augmented_labels = np.empty((num_augmeted_features,))
    idx = 0
    for i in range(labels.shape[0]):
        feature_vec = feature_arr[i]
        label = labels[i]
        # for the small group - the positive group, add the augmentations:
        if label == 1:
            new_features = augment_feature(feature_vec, std, multiplier)
            augmented_features[idx: idx+multiplier+1] = new_features
            augmented_labels[idx: idx+multiplier+1] = label
            idx += multiplier + 1
        else:
            # Add original feature and label:
            augmented_features[idx] = feature_vec
            augmented_labels[idx] = label
            idx += 1

    np.save('augmented_features', augmented_features)
    np.save('augmented_gt', augmented_labels)


if __name__ == '__main__':
    # # Create Selection random forest training features and labels:
    # cnn_pred_dir_path = '/cs/labs/josko/asherp7/follow_up/outputs/train_cnn_predictions_2020-03-26_13-46-08'
    # selection_output_folder = os.path.join(cnn_pred_dir_path, 'selection')
    # if not os.path.isdir(selection_output_folder):
    #     os.mkdir(selection_output_folder)
    #
    # data_dir_path = '/mnt/local/aszeskin/asher/liver_data/seperated_26_3'
    # split = 'train'
    # filenames = get_filepaths_from_data_split(data_dir_path, split, cnn_pred_dir_path)
    #
    # # create gt and features:
    # features, gt = create_random_forest_features_and_gt(filenames)
    # np.save('features', features)
    # np.save('gt', gt)
    path_to_labels, path_to_features = 'gt.npy', 'features.npy'
    balance_data_by_augmentation(path_to_labels, path_to_features)

    # load gt and features:
    gt = np.load('augmented_gt.npy')
    features = np.load('augmented_features.npy')


    num_estimatores = 100
    classifier = train_random_forest(num_estimatores, features, gt)
    # # apply selection to training set (just for testing selection)
    # apply_selection_on_dir(classifier, filenames, selection_output_folder)

    # apply selection to validation set - data the selection algorithm didn't see:
    split = 'validation'
    # data_dir_path = '/mnt/local/aszeskin/asher/liver_data/seperated_26_3'
    # validation_cnn_pred_dir_path = '/cs/labs/josko/asherp7/follow_up/outputs/pred_2020-03-26_10-20-24'
    data_dir_path = '/cs/labs/josko/asherp7/follow_up/data_31_3_2020'
    validation_cnn_pred_dir_path = '/cs/labs/josko/asherp7/follow_up/outputs/validation_cnn_predictions_1_4_2020_2020-04-01_01-57-47'
    validation_selection_output_folder = os.path.join(validation_cnn_pred_dir_path, 'selection')
    if not os.path.isdir(validation_selection_output_folder):
        os.mkdir(validation_selection_output_folder)
    validation_filenames = get_filepaths_from_data_split(data_dir_path, split, validation_cnn_pred_dir_path)
    apply_selection_on_dir(classifier, validation_filenames, validation_selection_output_folder)