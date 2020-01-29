import os
import argparse
import dicom2nifti


def convert_dicom_directory(dicom_directory, output_folder):
    print('converting dicom files from:', dicom_directory)
    # dicom2nifti.convert_directory(dicom_directory, output_folder, compression=True, reorient=True)
    print('completed dicom to nifti conversion.')
    print('output path:', output_folder)


if __name__ == '__main__':
    home_folder = '/cs/labs/josko/asherp7'
    dicom_directory = os.path.join(home_folder, 'dicom_data')
    output_folder = os.path.join(home_folder, 'nifti_data')
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default=dicom_directory, help="directory which contains")
    parser.add_argument("-o", "--output_dir", default=output_folder)
    args = parser.parse_args()
    convert_dicom_directory(args.input_dir, args.output_dir)