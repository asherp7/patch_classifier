import os
import glob
import shutil


    # dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allFU_liverSeg'
    # # rename files for easier handling:
    # for filename in os.listdir(dir_path):
    #     new_filename = filename.replace('allFU_', 'FU')
    #     new_filename = new_filename.replace('liverSeg', 'liverseg')
    #     old_file_path = os.path.join(dir_path, filename)
    #     new_file_path = os.path.join(dir_path, new_filename)
    #     os.rename(old_file_path, new_file_path)
    #     print(old_file_path, new_file_path)

    # # rename files for easier handling:
    # dir_path = '/cs/labs/josko/asherp7/follow_up/outputs/all_FU_predictions'
    # for filename in os.listdir(dir_path):
    #     new_filename = filename.replace('.nii', '_cnnRes.nii')
    #     old_file_path = os.path.join(dir_path, filename)
    #     new_file_path = os.path.join(dir_path, new_filename)
    #     os.rename(old_file_path, new_file_path)
    #     print(old_file_path, new_file_path)
#     data_from_aviv = '/cs/labs/josko/asherp7/Cases_for_training'


def combine_Rafi_data(cropped_data, ct_dir, liver_seg_dir, tumor_dir):

    cases_with_BL_and_FU = ['29', '30', '32', '35', '36']
    for case in cases_with_BL_and_FU:
        for case_type in ['BL', 'FU']:
            if case == '30' and case_type=='FU':
                continue
            # copy tumor segmentation:
            file_path = os.path.join(cropped_data, 'case_'+case, case_type+case+'_R1.nii.gz')
            filename = case_type+case+'_Tumors.nii.gz'
            copy_filepath = os.path.join(tumor_dir, filename)
            print(file_path)
            print(copy_filepath)
            assert os.path.isfile(file_path)
            shutil.copy(file_path, copy_filepath)

            # copy ct scan:
            filename = case_type+case+'.nii.gz'
            copy_filepath = os.path.join(ct_dir, filename)
            ct_filepath = os.path.join(cropped_data, 'case_'+case, filename)
            print(ct_filepath)
            print(copy_filepath)
            assert os.path.isfile(ct_filepath)
            shutil.copy(ct_filepath, copy_filepath)

            # copy liver segmentation:
            if case_type == 'FU':
                liver_dir = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allFU_liverSeg'
                liver_file = os.path.join(liver_dir, case_type+case+'_liverseg.nii')
                copy_liver_path = os.path.join(liver_seg_dir, case_type+case+'_liverseg.nii')
            elif case_type == 'BL':
                liver_dir = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL_liverSeg'
                liver_file = os.path.join(liver_dir, case_type+case+'_liverseg.nii.gz')
                copy_liver_path = os.path.join(liver_seg_dir, case_type+case+'_liverseg.nii.gz')

            print(liver_file)
            assert os.path.isfile(liver_file)
            print(copy_liver_path)
            shutil.copy(liver_file, copy_liver_path)
            print()


def combine_Richard_data(cropped_data, ct_dir, liver_seg_dir, tumor_dir):
    # copy New ct scans to combined folder structure
    for dir_name in os.listdir(cropped_data):
        if dir_name == 'Rafi_folder':
            print('skipping ', dir_name, '...')
            continue
        dir_path = os.path.join(cropped_data, dir_name)
        ct_path = os.path.join(dir_path, 'cropped_ct.nii.gz')
        ct_copy_path = os.path.join(ct_dir, dir_name+'.nii.gz')
        assert os.path.isfile(ct_path)
        print(ct_path)
        print(ct_copy_path)
        shutil.copy(ct_path, ct_copy_path)

        liver_path = os.path.join(dir_path, 'Liver.nii.gz')
        liver_copy_path = os.path.join(liver_seg_dir, dir_name+'_liverseg.nii.gz')
        assert os.path.isfile(liver_path)
        print(liver_path)
        print(liver_copy_path)
        shutil.copy(liver_path, liver_copy_path)

        seg_path = os.path.join(dir_path, 'cropped_tumors.nii.gz')
        copy_seg_path = os.path.join(tumor_dir, dir_name+'_Tumors.nii.gz')
        assert os.path.isfile(seg_path)
        print(seg_path)
        print(copy_seg_path)
        shutil.copy(seg_path, copy_seg_path)


if __name__ == '__main__':
    combined_data_root = '/cs/labs/josko/asherp7/follow_up/combined_data'
    ct_dir = os.path.join(combined_data_root, 'ct_scans')
    liver_seg_dir = os.path.join(combined_data_root, 'liver_seg')
    tumor_dir = os.path.join(combined_data_root, 'tumors')
    rafi_data = '/cs/labs/josko/public/for_aviv/TrainingSet/Rafi_folder'
    cropped_data = '/cs/labs/josko/public/for_aviv/TrainingSet'

    combine_Rafi_data(rafi_data, ct_dir, liver_seg_dir, tumor_dir)
    combine_Richard_data(cropped_data, ct_dir, liver_seg_dir, tumor_dir)