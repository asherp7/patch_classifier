from model import get_model
from prediction_generator import PredictionGenerator
from training_utils import limit_gpu_memory
import nibabel as nib
import numpy as np
import os


def predict_nifti(model, path_to_weights, path_to_ct_scan, path_to_liver_segmentation, output_path):
    """
    Predict
    :param model: a Keras model used for predicting patches.
    :param path_to_weights: path to weights that are loaded into model.
    :param path_to_ct_scan: Path to nifti CT scan.
    :param path_to_liver_segmentation: path to nifti file containing liver segmentation.
    :return: tumor probability map
    """
    scan_name = os.path.basename(path_to_ct_scan)
    model.load_weights(path_to_weights)
    prediction_generator = PredictionGenerator(path_to_ct_scan, path_to_liver_segmentation)
    predictions = model.predict_generator(prediction_generator, verbose=1)
    # np.savetxt(os.path.join(output_path, 'predictions.txt'), predictions)
    # np.savetxt(os.path.join(output_path, 'indices.txt'), prediction_generator.index_array)
    # predictions = np.loadtxt(os.path.join(output_path, 'predictions.txt'))
    # index_array = np.loadtxt(os.path.join(output_path, 'indices.txt'))
    index_array = prediction_generator.index_array
    ct_scan = nib.load(path_to_ct_scan)
    prediction_probability_map = ct_scan.get_data()
    arr = construct_3d_arry(predictions[:, 1], index_array, ct_scan.get_fdata().shape)
    arr = (arr * 1024).astype(np.int16)
    prediction_probability_map[:, :, :] = arr
    output_filepath = os.path.join(output_path, scan_name)
    nib.save(ct_scan, output_filepath)
    print('saved file to:', output_filepath)


def construct_3d_arry(predictions, indices, arr_shape):
    arr = np.zeros(arr_shape)
    for i in range(predictions.shape[0]):
        x, y, z = int(indices[i, 0]), int(indices[i, 1]), int(indices[i, 2])
        arr[x, y, z] = predictions[i]
    return arr


if __name__ == '__main__':
    path_to_ct_scan = '/cs/labs/josko/asherp7/example_cases/case11/BL/BL11.nii.gz'
    path_to_liver_segmentation = '/cs/labs/josko/asherp7/example_cases/case11/BL/BL11_liverseg.nii.gz'
    path_to_weights = '/mnt/local/aszeskin/asher/weights/weights-01-0.93.hdf5'
    output_path = '/cs/labs/josko/asherp7/follow_up/outputs'
    memory_fraction = 0.2
    limit_gpu_memory(memory_fraction)
    model = get_model()
    model.summary()
    predict_nifti(model, path_to_weights, path_to_ct_scan, path_to_liver_segmentation, output_path)

    # prediction_generator = PredictionGenerator(path_to_ct_scan, path_to_liver_segmentation)
    # batch = prediction_generator[0]
    # print(batch[0].shape)
    # print(batch[1].shape)

