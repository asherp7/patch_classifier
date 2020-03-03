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
                                                   chan_vase_data,
                                                   tumor_data,
                                                   cc_gt_intersection_thresh=0.2):
    feature_list = []
    gt_list = []
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

        cc_gt_intersection = np.sum(np.logical_and(tumor_data, component)) / volume
        if cc_gt_intersection > cc_gt_intersection_thresh:
            cc_label = 1
        else:
            cc_label = 0
        gt_list.append(cc_label)
        # print(list(zip(names, features)))
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
        print(list(zip(names, features)))
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
    for idx, (ct_filename, tumor_filename, cnn_pred_filename, chan_vase_filename) in enumerate(filenames):
        print('(', idx, '/', len(filenames), ') processing', os.path.basename(ct_filename))
        ct_data = nib.load(ct_filename).get_fdata()
        cnn_pred_data = nib.load(cnn_pred_filename).get_fdata()
        chan_vase_data = nib.load(chan_vase_filename).get_fdata()
        tumor_data = nib.load(tumor_filename).get_fdata()
        feature_list, gt_list = get_scan_connected_componenets_features_and_gt(ct_data,
                                                                               cnn_pred_data,
                                                                               chan_vase_data,
                                                                               tumor_data)
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


def train_random_forest(num_estimatores, features, gt):
    clf = RandomForestClassifier(n_estimators=num_estimatores, verbose=1)
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
    for idx, (ct_filename, tumor_filename, cnn_pred_filename, chan_vase_filename) in enumerate(filenames):
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


# if __name__ == '__main__':
#     # X, y = make_multilabel_classification(n_samples=1000, n_features=4)
#     # print(X.shape)
#     # print(y.shape)
#     # clf = RandomForestClassifier(max_depth=2, random_state=0)
#     # clf.fit(X, y)
#     # print(clf.feature_importances_)
#     # print(clf.predict([[0, 0, 0, 0]]))
#
#     # ct_path = '/cs/labs/josko/asherp7/Cases_for_training/cropped_with_liver_segmentation/A_A_03_10_2019/DICOM_Abdomen+_Abdomen_20191003182817_501.nii.gz'
#     # tumor_path = '/cs/labs/josko/asherp7/Cases_for_training/cropped_with_liver_segmentation/A_A_03_10_2019/AA_03102019_R1.nii.gz'
#     # liver_path = '/cs/labs/josko/asherp7/Cases_for_training/cropped_with_liver_segmentation/A_A_03_10_2019/Liver.nii.gz'
#
#
#     # ct_vec = transform_nifti_to_vector(ct_path)
#     # tumor_path = transform_nifti_to_vector(tumor_path)
#     # liver_vec = transform_nifti_to_vector(liver_path)
#     # print(ct_vec.shape, tumor_path.shape, liver_vec.shape)
#     # classifier = train_random_forest(num_estimatores, ct_vec, tumor_path)
#     # pred = classifier.predict(ct_vec)
#
#     ct_path = '/cs/labs/josko/asherp7/follow_up/Chanvese/allFU'
#     ct15_path = os.path.join(ct_path, 'FU15.nii.gz')
#
#     tumor_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allFU_newAndOldTumors/'
#     tumor15_path = os.path.join(tumor_path, 'FU15_newTumors_copy.nii.gz')
#
#     chan_vase_path = '/cs/labs/josko/asherp7/follow_up/Chanvese/allFU_segmentation_CNN_and_CV'
#     chan_vase_15_path = os.path.join(chan_vase_path, 'FU15_chanvese_seg_expand.nii.gz')
#
#     cnn_pred_path = '/cs/labs/josko/asherp7/follow_up/Chanvese/allFU_cnnRes'
#     cnn_pred_15_path = os.path.join(cnn_pred_path, 'FU15_cnnRes.nii.gz')
#
#     num_estimatores = 30
#     cc_min_size = 10
#
#     ct_data = nib.load(ct15_path).get_fdata()
#     cnn_pred_data = nib.load(cnn_pred_15_path).get_fdata()
#     print('removing connected components smaller than', cc_min_size)
#     chan_vase_data = remove_small_connected_componenets_3D(nib.load(chan_vase_15_path).get_fdata(), cc_min_size)
#     tumor_data = nib.load(tumor15_path).get_fdata()
#
#     features, labeled_array, num_components = get_scan_connected_componenets_features_and_labeled_array(ct_data,
#                                                                                                         cnn_pred_data,
#                                                                                                         chan_vase_data)
#     gt = get_scan_connected_componenets_gt(chan_vase_data, tumor_data)
#     classifier = train_random_forest(num_estimatores, features, gt)
#
#     selected = apply_selection_on_cnn_and_chan_vase(classifier, cnn_pred_data, chan_vase_data)
#     old_file_path = chan_vase_15_path
#     new_filepath = 'selected.nii.gz'
#     save_data_as_new_nifti_file(old_file_path, selected, new_filepath)

if __name__ == '__main__':
    ct_dir_path = '/cs/labs/josko/asherp7/follow_up/combined_data/ct_scans'
    tumor_dir_path = '/cs/labs/josko/asherp7/follow_up/combined_data/tumors'
    cnn_pred_dir_path = '/cs/labs/josko/asherp7/follow_up/outputs/results_27_2/predictions'
    chan_vase_remove_small_cc_dir_path = '/cs/labs/josko/asherp7/follow_up/outputs/results_27_2/chan_vese_remove_small_cc'
    # old_dir_path= '/cs/labs/josko/asherp7/follow_up/outputs/results_27_2/chan_vese'
    # remove_small_components_and_save(old_dir_path, chan_vase_dir_path)
    selection_output_folder = '/cs/labs/josko/asherp7/follow_up/outputs/results_27_2/selection'


    filenames = get_filenames(ct_dir_path, tumor_dir_path, cnn_pred_dir_path, chan_vase_remove_small_cc_dir_path)

    # create gt and features:
    # features, gt = create_random_forest_features_and_gt(filenames)
    # np.save('features', features)
    # np.save('gt', gt)

    # load gt and features:
    gt = np.load('gt.npy')
    features = np.load('features.npy')

    num_estimatores = 30
    classifier = train_random_forest(num_estimatores, features, gt)
    apply_selection_on_dir(classifier, filenames, selection_output_folder)