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
    ct_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL'
    liver_seg_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL_liverSeg'
    path_to_weights = '/mnt/local/aszeskin/asher/weights/weights-01-0.93.hdf5'
    output_path = '/cs/labs/josko/asherp7/follow_up/outputs/all_predictions'
    model = get_model()
    model.summary()
    predict_on_all_scans(ct_dir_path, liver_seg_path, model, path_to_weights, output_path)