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
    # ct_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL'
    # liver_seg_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL_liverSeg'
    # ct_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allFU'
    # liver_seg_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allFU_liverSegFixed'
    # ct_dir_path = '/cs/labs/josko/asherp7/follow_up/combined_data/ct_scans'
    # ct_dir_path = '/cs/labs/josko/asherp7/follow_up/combined_data_2/validation/ct_scans'
    ct_dir_path = ''
    # ct_dir_path = '/cs/labs/josko/asherp7/follow_up/combined_data/ct_FU'
    # liver_seg_path = '/cs/labs/josko/asherp7/follow_up/combined_data/liver_seg'
    # liver_seg_path = '/cs/labs/josko/asherp7/follow_up/combined_data_2/validation/liver_seg'
    liver_seg_path = ''
    # path_to_weights = '/mnt/local/aszeskin/asher/weights/weights-01-0.93.hdf5'
    # path_to_weights = '/cs/labs/josko/asherp7/follow_up/weights-01-0.93.hdf5'
    # path_to_weights = '/mnt/local/aszeskin/asher/weights/unet_train_all_BL_2020-02-20_13-56-49/weights-05-0.96.hdf5'
    # path_to_weights = '/mnt/local/aszeskin/asher/weights/unet_train_all_BL_2020-02-27_15-01-39/weights-01-0.91.hdf5'
    # path_to_weights = '/mnt/local/aszeskin/asher/weights/combined_data_fixed_liver_seg_2020-03-03_14-12-47/weights-01-0.92.hdf5'
    # path_to_weights = '/mnt/local/aszeskin/asher/weights/fixed_normalization_2020-03-05_12-24-54/weights-16-0.96.hdf5'
    # path_to_weights = '/mnt/local/aszeskin/asher/weights/fixed_liver_segmentation_full_validation_set_with_augs_2020-03-12_14-34-18/weights-07-0.77.hdf5'
    path_to_weights = '/mnt/local/aszeskin/asher/weights/fixed_liver_segmentation_with_augs_random_split_2020-03-12_17-11-39/weights-13-0.97.hdf5'
    save_path = '/cs/labs/josko/asherp7/follow_up/outputs/'
    output_path = create_preditcions_save_dir(save_path, 'cnn_predictions')
    limit_gpu_memory(memory_fraction)
    model = get_model()
    model.summary()
    # predict_on_all_scans(ct_dir_path, liver_seg_path, model, path_to_weights, output_path)
    data_split_path = '/mnt/local/aszeskin/asher/liver_data/data_split.json'
    split = 'validation'
    predict_on_data_split(data_split_path, split, model, path_to_weights, output_path)

    # # predict on single file:
    # ct_path = '/cs/labs/josko/asherp7/Cases_for_training/FollowUps/Cropped/A_A_15_12_2019/cropped_ct.nii.gz'
    # roi_path = '/cs/labs/josko/asherp7/Cases_for_training/FollowUps/Cropped/A_A_15_12_2019/Liver.nii.gz'
    # predict_nifti(model, path_to_weights, ct_path, roi_path, output_path)


