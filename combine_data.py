import os
import glob
import shutil


def combine_data(cropped_data,
                 ct_dir,
                 liver_seg_dir,
                 tumor_dir,
                 ct_filename='Cropped_CT_Canonical.nii.gz',
                 liver_filename='Cropped_Liver_Canonical.nii.gz',
                 tumor_filename='Cropped_Tumors_Canonical.nii.gz',
                 prefix=''
                 ):
    ct_filename = prefix + ct_filename
    liver_filename = prefix + liver_filename
    tumor_filename = prefix + tumor_filename
    # copy New ct scans to combined folder structure
    good_case_count = 0
    for idx, dir_name in enumerate(os.listdir(cropped_data), 1):
        if (dir_name == 'case_09' or dir_name == 'case_10' or dir_name == 'case_16') and prefix == 'FU_':
            print('skipping ', dir_name, prefix, ', since CT has no contrast agent!')
            continue
        print("case", idx, ":")

        dir_path = os.path.join(cropped_data, dir_name)
        ct_path = os.path.join(dir_path, ct_filename)
        ct_copy_path = os.path.join(ct_dir, prefix + dir_name+'.nii.gz')
        print(ct_path)
        print(ct_copy_path)
        assert os.path.isfile(ct_path)
        shutil.copy(ct_path, ct_copy_path)

        liver_path = os.path.join(dir_path, liver_filename)
        liver_copy_path = os.path.join(liver_seg_dir, prefix + dir_name + '_liverseg.nii.gz')
        print(liver_path)
        print(liver_copy_path)
        assert os.path.isfile(liver_path)
        shutil.copy(liver_path, liver_copy_path)

        seg_path = os.path.join(dir_path, tumor_filename)
        copy_seg_path = os.path.join(tumor_dir, prefix + dir_name + '_Tumors.nii.gz')
        print(seg_path)
        print(copy_seg_path)
        assert os.path.isfile(seg_path)
        shutil.copy(seg_path, copy_seg_path)
        print()
        good_case_count += 1

    return good_case_count


if __name__ == '__main__':
    # data source
    rafi_data = '/cs/labs/josko/public/for_aviv/TrainingSet/Rafi_folder/'
    new_data = '/cs/labs/josko/public/for_aviv/TrainingSet/new_cases/'

    # data destination:
    combined_data_root = '/cs/labs/josko/asherp7/follow_up/data_3_4_2020/'

    ct_dir = os.path.join(combined_data_root, 'ct_scans')
    liver_seg_dir = os.path.join(combined_data_root, 'liver_seg')
    tumor_dir = os.path.join(combined_data_root, 'tumors')

    # create folders:
    if not os.path.isdir(ct_dir):
        os.mkdir(ct_dir)
    if not os.path.isdir(liver_seg_dir):
        os.mkdir(liver_seg_dir)
    if not os.path.isdir(tumor_dir):
        os.mkdir(tumor_dir)

    num_cases = {}
    # Combine Richard's data:
    prefix = ''
    ct_filename = 'Cropped_CT_Canonical.nii.gz'
    liver_filename = 'Cropped_Liver_Canonical.nii.gz'
    tumor_filename = 'Cropped_Tumors_Canonical.nii.gz'
    num_cases["New"] = combine_data(new_data, ct_dir, liver_seg_dir, tumor_dir, ct_filename, liver_filename, tumor_filename, prefix)

    # Combine old Base-Line data:
    prefix = 'BL_'
    num_cases["BL"] = combine_data(rafi_data, ct_dir, liver_seg_dir, tumor_dir, ct_filename, liver_filename, tumor_filename, prefix)

    # Combine old Follow-Up data:
    prefix = 'FU_'
    num_cases["FU"] = combine_data(rafi_data, ct_dir, liver_seg_dir, tumor_dir, ct_filename, liver_filename, tumor_filename, prefix)

    print("case summary:")
    print(num_cases)
    print("total number of cases:", sum(num_cases.values()))
