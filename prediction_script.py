from train.training_utils import limit_gpu_memory
from train.patch_model import get_model
from predict.prediction_utils import *
from datetime import datetime

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


def create_preditcions_save_dir(save_path, title):
    now = datetime.now() # current date and time
    date_time = now.strftime("_%Y-%m-%d_%H-%M-%S")
    prediction_path = os.path.join(save_path, title+date_time)
    os.mkdir(prediction_path)
    return prediction_path


if __name__ == '__main__':
    memory_fraction = 0.2

    ct_dir_path = '/cs/labs/josko/asherp7/follow_up/data_31_3_2020/ct_scans'
    liver_seg_path = '/cs/labs/josko/asherp7/follow_up/data_31_3_2020/liver_seg'
    path_to_weights = '/mnt/local/aszeskin/asher/weights/dataset_75_cases_2020-03-31_17-57-05/weights-06-0.87.hdf5'
    data_split_path = '/mnt/local/aszeskin/asher/weights/dataset_75_cases_2020-03-31_17-57-05/data_split.json'

    save_path = '/cs/labs/josko/asherp7/follow_up/outputs/'
    output_path = create_preditcions_save_dir(save_path, 'train_cnn_predictions_1_4_2020')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    limit_gpu_memory(memory_fraction)
    model = get_model()
    model.summary()
    # predict_on_all_scans(ct_dir_path, liver_seg_path, model, path_to_weights, output_path)
    split = 'train'
    predict_on_data_split(data_split_path, split, model, path_to_weights, output_path)

    # # predict on single file:
    # ct_path = '/cs/labs/josko/asherp7/Cases_for_training/FollowUps/Cropped/A_A_15_12_2019/cropped_ct.nii.gz'
    # roi_path = '/cs/labs/josko/asherp7/Cases_for_training/FollowUps/Cropped/A_A_15_12_2019/Liver.nii.gz'
    # predict_nifti(model, path_to_weights, ct_path, roi_path, output_path)


