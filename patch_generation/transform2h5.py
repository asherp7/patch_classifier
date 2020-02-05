import nibabel as nib
import numpy as np
import random
import h5py
import math
import json
import os


class Transform2h5:
    def __init__(self, nifti_dir_path, output_dir_path, output_filename, orientation, patch_size, sampling_step,
                 roi_path, roi_suffix, tumor_data_path, tumor_suffix, padding=None, augmetations=None, balance=True):
        self.nifti_dir_path = nifti_dir_path
        self.output_dir_path = output_dir_path
        self.orientation = orientation
        self.patch_size = patch_size
        self.sampling_step = sampling_step
        self.roi_path = roi_path
        self.padding = padding
        self.augmetations = augmetations
        self.output_file_path = os.path.join(self.output_dir_path, output_filename)
        self.roi_suffix = roi_suffix
        self.tumor_data_path = tumor_data_path
        self.tumor_suffix = tumor_suffix
        self.balance_patches = balance
        self.patches_datset_name = 'patches'
        self.label_dataset_name = 'labels'
        self.tumor_indices_name = 'tumor_idx'
        self.patch_dict_name = 'patch_dict'
        self.non_tumor_indices_name = 'non_tumor_idx'
        self.num_saved_patches = 0

    def read_nifti(self, nifti_filepath):
        return nib.load(nifti_filepath)

    def get_nifti_data(self, nifti):
        return nifti.get_fdata()

    def get_ct_liver_tumor_filepaths_list(self):
        file_names_list = []
        for filename in os.listdir(self.nifti_dir_path):
            tumor_file_path = os.path.join(self.tumor_data_path, filename.replace('.nii.gz', self.tumor_suffix+'.nii.gz'))
            if filename.startswith('BL'):
                extension = '.nii.gz'
            else:
                extension = '.nii'
            roi_file_path = os.path.join(self.roi_path, filename.replace('.nii.gz', self.roi_suffix+extension))
            if not os.path.isfile(roi_file_path):
                print(roi_file_path, 'is missing!')
                continue
            if not os.path.isfile(tumor_file_path):
                print(tumor_file_path, 'is missing!')
                continue
            scan_file_path = os.path.join(self.nifti_dir_path, filename)
            file_names_list.append((scan_file_path, roi_file_path, tumor_file_path))
        return file_names_list

    def transform_to_global_indices(self, tumor_patches_indices, non_tumor_patches_indices):
        tumor_patches_indices = [self.num_saved_patches + x for x in tumor_patches_indices]
        non_tumor_patches_indices = [self.num_saved_patches + x for x in non_tumor_patches_indices]
        return tumor_patches_indices, non_tumor_patches_indices

    def save_single_nifti_patches(self, nifti, roi, tumor):
        data = self.get_nifti_data(nifti)
        roi_data = self.get_nifti_data(roi)
        tumor_data = self.get_nifti_data(tumor)
        patch_list, patch_labels, tumor_patches_indices, non_tumor_patches_indices = \
            self.split_arr_into_patches(data, roi_data, tumor_data)
        if self.balance_patches:
            # balance out nifti patches:
            patch_list, patch_labels, tumor_patches_indices, non_tumor_patches_indices = \
                self.balance_scan_patches(patch_list, patch_labels, tumor_patches_indices, non_tumor_patches_indices)

        # move from single nifti indices, to global indices:
        tumor_patches_indices, non_tumor_patches_indices = \
            self.transform_to_global_indices(tumor_patches_indices, non_tumor_patches_indices)
        self.save_patches(patch_list)
        self.save_labels(patch_labels)
        self.save_tumor_patches(tumor_patches_indices)
        self.save_non_tumor_patches(non_tumor_patches_indices)

    def balance_scan_patches(self, patch_list, patch_labels, tumor_patches_indices, non_tumor_patches_indices):
        num_tumer_patches = len(tumor_patches_indices)
        num_healthy_patches = len(non_tumor_patches_indices)
        if num_tumer_patches < num_healthy_patches:
            non_tumor_patches_indices = random.sample(non_tumor_patches_indices, num_tumer_patches)
        elif num_tumer_patches > num_healthy_patches:  # not a likely case
            tumor_patches_indices = random.sample(tumor_patches_indices, num_healthy_patches)
        all_patches_indices = tumor_patches_indices + non_tumor_patches_indices
        if all_patches_indices:
            patch_list = np.asarray(patch_list)[all_patches_indices]
            patch_labels = np.asarray(patch_labels)[all_patches_indices]
        return patch_list, patch_labels, tumor_patches_indices, non_tumor_patches_indices

    def save_patches(self, patch_list):
        if isinstance(patch_list, list):
            if not patch_list:
                return
        elif isinstance(patch_list, np.ndarray):
            if patch_list.size == 0:
                return
        patch_stack = np.stack(patch_list)
        self.hf[self.patches_datset_name].resize(self.hf[self.patches_datset_name].shape[0] + patch_stack.shape[0], axis=0)
        self.hf[self.patches_datset_name][-patch_stack.shape[0]:] = patch_stack
        self.num_saved_patches += patch_stack.shape[0]

    def save_labels(self, patch_labels):
        if len(patch_labels) == 0:
            return
        label_stack = np.array(patch_labels, dtype=np.uint8)
        self.hf[self.label_dataset_name].resize(self.hf[self.label_dataset_name].shape[0] + label_stack.shape[0], axis=0)
        self.hf[self.label_dataset_name][-label_stack.shape[0]:] = label_stack

    def save_tumor_patches(self, tumor_patches_indices):
        if len(tumor_patches_indices) == 0:
            return
        arr_indices = np.array(tumor_patches_indices, dtype=np.uint8)
        self.hf[self.tumor_indices_name].resize(self.hf[self.tumor_indices_name].shape[0] + arr_indices.shape[0], axis=0)
        self.hf[self.tumor_indices_name][-arr_indices.shape[0]:] = arr_indices

    def save_non_tumor_patches(self, non_tumor_patches_indices):
        if len(non_tumor_patches_indices) == 0:
            return
        arr_indices = np.array(non_tumor_patches_indices, dtype=np.uint8)
        self.hf[self.non_tumor_indices_name].resize(self.hf[self.non_tumor_indices_name].shape[0] + arr_indices.shape[0], axis=0)
        self.hf[self.non_tumor_indices_name][-arr_indices.shape[0]:] = arr_indices

    def split_arr_into_patches(self, arr, organ_segmentation, tumor_segmentation):
        patch_list = []
        rows, columns, depth = arr.shape
        tumor_patches_indices = []
        non_tumor_patches_indices = []
        patch_labels = []
        patch_idx = 0
        for z in range(depth):
            for y in range(0, rows, self.sampling_step):
                for x in range(0, columns, self.sampling_step):
                    if x + self.patch_size < columns and y + self.patch_size < rows:  # discard border patches
                        if self.is_patch_center_in_mask(x, y, z, organ_segmentation):
                            if self.is_patch_center_in_mask(x, y, z, tumor_segmentation):
                                patch_labels.append(1)  # center of patch belongs to tumor
                                tumor_patches_indices.append(patch_idx)
                            else:
                                patch_labels.append(0)  # center of patch DOES NOT belong to tumor
                                non_tumor_patches_indices.append(patch_idx)
                            patch = arr[y:y+self.patch_size, x:x+self.patch_size, z]
                            patch_list.append(patch)
                            patch_idx += 1
        return patch_list, patch_labels, tumor_patches_indices, non_tumor_patches_indices

    def create_h5_datasets(self, split=''):
        print('creating', split, 'data at:')
        print(self.output_file_path.replace('.h5', '_'+split+'.h5'))
        self.hf = h5py.File(self.output_file_path.replace('.h5', '_'+split+'.h5'), 'w')
        self.hf.create_dataset(self.patches_datset_name,shape=(0, self.patch_size, self.patch_size), chunks=True,
                               maxshape=(None,self.patch_size, self.patch_size))
        self.hf.create_dataset(self.label_dataset_name, shape=(0,), chunks=True, maxshape=(None,))
        self.hf.create_dataset(self.tumor_indices_name, shape=(0,), chunks=True, maxshape=(None,))
        self.hf.create_dataset(self.non_tumor_indices_name, shape=(0,), chunks=True, maxshape=(None,))

    def save_all_nifti_patches(self):
        self.create_h5_datasets()
        file_paths = self.get_ct_liver_tumor_filepaths_list()
        for idx, (scan_filepath, roi_filepath, tumor_filepath) in enumerate(file_paths):
            print('(', idx+1, '/', len(file_paths), ')', 'processing', scan_filepath, '...')
            scan = self.read_nifti(scan_filepath)
            roi = self.read_nifti(roi_filepath)
            tumor = self.read_nifti(tumor_filepath)
            self.standardize_orientation(scan)
            self.standardize_orientation(roi)
            self.standardize_orientation(tumor)
            self.save_single_nifti_patches(scan, roi, tumor)
        self.hf.close()

    def save_all_patches_split_train_validation(self, validation_ratio=0.2):
        print('creating train set:')
        self.create_h5_datasets('train')
        file_paths = self.get_ct_liver_tumor_filepaths_list()
        num_validation_files = math.floor(len(file_paths) * validation_ratio)
        validation_file_indices = random.sample(range(len(file_paths)), num_validation_files)
        train_file_indices = list(set(range(len(file_paths))) - set(validation_file_indices))
        train_file_paths = [file_paths[i] for i in train_file_indices]
        validation_file_paths = [file_paths[i] for i in validation_file_indices]
        for idx, (scan_filepath, roi_filepath, tumor_filepath) in enumerate(train_file_paths):
            print('(', idx+1, '/', len(train_file_paths), ')', 'processing', scan_filepath, '...')
            scan = self.read_nifti(scan_filepath)
            roi = self.read_nifti(roi_filepath)
            tumor = self.read_nifti(tumor_filepath)
            self.standardize_orientation(scan)
            self.standardize_orientation(roi)
            self.standardize_orientation(tumor)
            self.save_single_nifti_patches(scan, roi, tumor)
        self.hf.close()
        print('creating validation set:')
        self.create_h5_datasets('validation')
        for idx, (scan_filepath, roi_filepath, tumor_filepath) in enumerate(validation_file_paths):
            print('(', idx+1, '/', len(validation_file_paths), ')', 'processing', scan_filepath, '...')
            scan = self.read_nifti(scan_filepath)
            roi = self.read_nifti(roi_filepath)
            tumor = self.read_nifti(tumor_filepath)
            self.standardize_orientation(scan)
            self.standardize_orientation(roi)
            self.standardize_orientation(tumor)
            self.save_single_nifti_patches(scan, roi, tumor)
        self.hf.close()

        data_split = {"train": train_file_paths, "validation": validation_file_paths}
        data_split_filepath = os.path.join(self.output_dir_path, 'data_split.json')
        with open(data_split_filepath, 'w') as fp:
            json.dump(data_split, fp, sort_keys=True, indent=4)
        print('saved training - validation split to:', data_split_filepath)



    def standardize_orientation(self, nifti):
        for i in range(3):
            if nib.aff2axcodes(nifti.affine)[i] != self.orientation[i]:
                print('flipping axis', i)
                nifti = nib.orientations.flip_axis(nifti, axis=i)
        return nifti

    def is_patch_center_in_mask(self, patch_x, patch_y, patch_z, roi_mask):
        patch_center_x = patch_x + self.patch_size // 2
        patch_center_y = patch_y + self.patch_size // 2
        if roi_mask[patch_center_y, patch_center_x, patch_z] == 0:
            return False
        else:
            return True

    @staticmethod
    def get_roi_bbox(arr):
        """
        Find bbox surrounding mask in 3d data
        :param arr:
        :return:
        """
        r = np.any(arr, axis=(1, 2))
        c = np.any(arr, axis=(0, 2))
        z = np.any(arr, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return arr[rmin:rmax+1, cmin:cmax+1, zmin:zmax+1]
