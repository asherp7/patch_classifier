from predict.prediction_utils import *

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
    # ct_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL'
    # liver_seg_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL_liverSeg'
    # ct_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allFU'
    # liver_seg_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allFU_liverSegFixed'
    # ct_dir_path = '/cs/labs/josko/asherp7/follow_up/combined_data/ct_scans'
    ct_dir_path = '/cs/labs/josko/asherp7/follow_up/combined_data/ct_FU'
    liver_seg_path = '/cs/labs/josko/asherp7/follow_up/combined_data/liver_seg'
    # path_to_weights = '/mnt/local/aszeskin/asher/weights/weights-01-0.93.hdf5'
    # path_to_weights = '/cs/labs/josko/asherp7/follow_up/weights-01-0.93.hdf5'
    # path_to_weights = '/mnt/local/aszeskin/asher/weights/unet_train_all_BL_2020-02-20_13-56-49/weights-05-0.96.hdf5'
    path_to_weights = '/mnt/local/aszeskin/asher/weights/unet_train_all_BL_2020-02-27_15-01-39/weights-01-0.91.hdf5'
    output_path = '/cs/labs/josko/asherp7/follow_up/outputs/new_predictions'
    model = get_model()
    model.summary()
    model.load_weights(path_to_weights)
    predict_on_all_scans(ct_dir_path, liver_seg_path, model, path_to_weights, output_path)


    # # predict on single file:
    # ct_path = '/cs/labs/josko/asherp7/Cases_for_training/FollowUps/Cropped/A_A_15_12_2019/cropped_ct.nii.gz'
    # roi_path = '/cs/labs/josko/asherp7/Cases_for_training/FollowUps/Cropped/A_A_15_12_2019/Liver.nii.gz'
    # predict_nifti(model, path_to_weights, ct_path, roi_path, output_path)


