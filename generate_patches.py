from patch_generation.transform2h5 import Transform2h5
import h5py
import os


def create_training_set():
    nifti_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL'
    roi_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL_liverSeg'
    tumor_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL_onlytumors'
    output_path = '/mnt/local/aszeskin/asher/liver_data'
    output_filename = 'BL_all_patches.h5'
    roi_suffix = '_liverseg'
    tumor_suffix = '_Tumors'
    patch_size = 35
    sampling_step = 2
    transform = Transform2h5(nifti_dir_path, output_path,output_filename, ('L', 'P', 'S'), patch_size, sampling_step,
                             roi_dir_path, roi_suffix, tumor_dir_path, tumor_suffix)
    transform.save_all_nifti_patches()
    check_file(output_path, output_filename)


def create_validation_set():
    nifti_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allFU'
    roi_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allFU_liverSeg/'
    tumor_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allFU_newAndOldTumors/'
    output_path = '/mnt/local/aszeskin/asher/liver_data'
    output_filename = 'FU_patches.h5'
    roi_suffix = '_liverseg'
    tumor_suffix = '_newTumors_copy'
    patch_size = 35
    sampling_step = 6
    transform = Transform2h5(nifti_dir_path, output_path, output_filename, ('L', 'P', 'S'), patch_size, sampling_step,
                             roi_dir_path, roi_suffix, tumor_dir_path, tumor_suffix)
    transform.save_all_nifti_patches()
    check_file(output_path, output_filename)


def create_training_and_validation_from_BL():
    nifti_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL'
    roi_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL_liverSeg'
    tumor_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL_onlytumors'
    output_path = '/mnt/local/aszeskin/asher/liver_data/'
    output_filename = 'patches.h5'
    roi_suffix = '_liverseg'
    tumor_suffix = '_Tumors'
    patch_size = 35
    sampling_step = 3
    transform = Transform2h5(nifti_dir_path, output_path,output_filename, ('L', 'P', 'S'), patch_size, sampling_step,
                             roi_dir_path, roi_suffix, tumor_dir_path, tumor_suffix)
    transform.save_all_patches_split_train_validation()
    check_file(output_path, 'patches_train.h5')
    check_file(output_path, 'patches_validation.h5')


def create_unet_training_set():
    nifti_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL'
    roi_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL_liverSeg'
    tumor_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL_onlytumors'
    output_path = '/mnt/local/aszeskin/asher/liver_data'
    output_filename = 'unet_BL_all_patches.h5'
    roi_suffix = '_liverseg'
    tumor_suffix = '_Tumors'
    patch_size = 64
    sampling_step = 2
    save_tumor_segmentation = True
    transform = Transform2h5(nifti_dir_path, output_path,output_filename, ('L', 'P', 'S'), patch_size, sampling_step,
                             roi_dir_path, roi_suffix, tumor_dir_path, tumor_suffix,
                             save_tumor_segmentation=save_tumor_segmentation)
    transform.save_all_nifti_patches()
    check_file(output_path, output_filename)


def create_unet_validation_set():
    nifti_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allFU'
    roi_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allFU_liverSeg/'
    tumor_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allFU_newAndOldTumors/'
    output_path = '/mnt/local/aszeskin/asher/liver_data'
    output_filename = 'unet_FU_patches.h5'
    roi_suffix = '_liverseg'
    tumor_suffix = '_newTumors_copy'
    patch_size = 64
    sampling_step = 6
    save_tumor_segmentation = True
    transform = Transform2h5(nifti_dir_path, output_path, output_filename, ('L', 'P', 'S'), patch_size, sampling_step,
                             roi_dir_path, roi_suffix, tumor_dir_path, tumor_suffix,
                             save_tumor_segmentation=save_tumor_segmentation)
    transform.save_all_nifti_patches()
    check_file(output_path, output_filename, check_mask_patches=True)


def create_training_set_combined_data():
    nifti_dir_path = '/cs/labs/josko/asherp7/combined_data/ct_scans'
    roi_dir_path = '/cs/labs/josko/asherp7/combined_data/liver_seg'
    tumor_dir_path = '/cs/labs/josko/asherp7/combined_data/tumors'
    output_path = '/mnt/local/aszeskin/asher/liver_data'
    output_filename = 'combined_patches_step_2.h5'
    roi_suffix = '_liverseg'
    tumor_suffix = '_Tumors'
    patch_size = 35
    sampling_step = 2
    transform = Transform2h5(nifti_dir_path, output_path,output_filename, ('L', 'P', 'S'), patch_size, sampling_step,
                             roi_dir_path, roi_suffix, tumor_dir_path, tumor_suffix)
    transform.save_all_nifti_patches()
    check_file(output_path, output_filename)


def check_file(output_path, file_name, check_mask_patches=False):
    file_path = os.path.join(output_path, file_name)
    f = h5py.File(file_path, 'r')
    print('\ndatasets in ', file_path, ':\n')
    print(f.keys())
    print(f['patches'])
    print(f['labels'])
    print(f['tumor_idx'])
    print(f['non_tumor_idx'])
    if check_mask_patches:
        print(f['mask_patches'])


if __name__ == '__main__':
    # create_training_set()
    # create_validation_set()

    # create_training_and_validation_from_BL()

    # create_unet_training_set()
    # create_unet_validation_set()

    create_training_set_combined_data()