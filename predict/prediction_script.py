from train.model import get_model
from predict.prediction_generator import PredictionGenerator
# from training_utils import limit_gpu_memory
import nibabel as nib
import numpy as np
import os


def predict_nifti(model, path_to_weights, path_to_ct_scan, path_to_liver_segmentation, output_path):
    """
    Use model and weights to predict tumor probability map on ct nifti in the ROI given by the liver segmentation.
    :param model: a Keras model used for predict patches.
    :param path_to_weights: path to weights that are loaded into model.
    :param path_to_ct_scan: Path to nifti CT scan.
    :param path_to_liver_segmentation: path to nifti file containing liver segmentation.
    :param output_path: path into which to save the output probability map
    :param save_array: flag to save the array for debug mode.
    """
    model.load_weights(path_to_weights)
    scan_name = os.path.basename(path_to_ct_scan)
    # Use model for prediction:
    prediction_generator = PredictionGenerator(path_to_ct_scan, path_to_liver_segmentation)
    predictions = model.predict_generator(prediction_generator, verbose=1)
    index_array = prediction_generator.index_array
    # create tumor probability map:
    ct_scan = nib.load(path_to_ct_scan)
    prediction_probability_map = construct_3d_arry(predictions[:, 1], index_array, ct_scan.get_fdata().shape)
    output_filepath = os.path.join(output_path, scan_name)
    new_img = nib.Nifti1Image(prediction_probability_map, ct_scan.affine)
    # save tumor probability map:
    nib.save(new_img, output_filepath)
    print('saved file to:', output_filepath)


def construct_3d_arry(predictions, indices, arr_shape):
    arr = np.zeros(arr_shape)
    for i in range(predictions.shape[0]):
        y, x, z = int(indices[i, 0]), int(indices[i, 1]), int(indices[i, 2])
        arr[y, x, z] = predictions[i]
    return arr


def get_ct_and_liver_segmentation_filepaths(ct_dir_path, liver_seg_path, roi_suffix='_liverseg'):
    file_names_list = []
    for filename in os.listdir(ct_dir_path):
        if filename.startswith('BL'):
            extension = '.nii.gz'
        else:
            extension = '.nii'
        roi_file_path = os.path.join(liver_seg_path, filename.replace('.nii.gz', roi_suffix+extension))
        if not os.path.isfile(roi_file_path):
            print(roi_file_path, 'is missing!')
            continue
        scan_file_path = os.path.join(ct_dir_path, filename)
        file_names_list.append((scan_file_path, roi_file_path))
    return file_names_list


def predict_on_all_scans(ct_dir_path, liver_seg_path, model, path_to_weights, output_dir_path):
    model.load_weights(path_to_weights)
    file_list = get_ct_and_liver_segmentation_filepaths(ct_dir_path, liver_seg_path)
    for idx, (ct_path, roi_path) in enumerate(file_list, 1):
        print(idx, '/', len(file_list), 'predict:', os.path.basename(ct_path))
        predict_nifti(model, path_to_weights, ct_path, roi_path, output_dir_path)

# if __name__ == '__main__':
#     path_to_ct_scan =  '/cs/labs/josko/asherp7/example_cases/case11/BL/BL11.nii.gz'
#     path_to_liver_segmentation = '/cs/labs/josko/asherp7/example_cases/case11/BL/BL11_liverseg.nii.gz'
#     path_to_weights = '/mnt/local/aszeskin/asher/weights/weights-01-0.93.hdf5'
#     output_path = '/cs/labs/josko/asherp7/follow_up/outputs'
#
#     # Uncomment to enable limitation of gpu memory.
#     # memory_fraction = 0.2
#     # limit_gpu_memory(memory_fraction)
#     model = get_model()
#     model.summary()
#     predict_nifti(model, path_to_weights, path_to_ct_scan, path_to_liver_segmentation, output_path)


if __name__ == '__main__':
    ct_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL'
    liver_seg_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL_liverSeg'
    path_to_weights = '/mnt/local/aszeskin/asher/weights/weights-01-0.93.hdf5'
    output_path = '/cs/labs/josko/asherp7/follow_up/outputs/all_predictions'
    model = get_model()
    model.summary()
    predict_on_all_scans(ct_dir_path, liver_seg_path, model, path_to_weights, output_path)

