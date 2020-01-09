import nibabel as nib
import numpy as np
import h5py
import os


class Transform2h5:
    def __init__(self, nifti_dir_path, output_dir_path, orientation, patch_size, overlapping, roi=None, padding=None,
                 augmetations=None):
        self.nifti_dir_path = nifti_dir_path
        self.output_dir_path = output_dir_path
        self.orientation = orientation
        self.patch_size = patch_size
        self.overlapping = overlapping
        self.roi = roi
        self.padding = padding
        self.augmetations = augmetations
        self.output_file_path = os.path.join(self.output_dir_path, 'patches.h5')
        # self.hf = h5py.File(self.output_file_path, 'w')

    def read_nifti(self, nifti_filename):
        nifti_filepath = os.path.join(self.nifti_dir_path, nifti_filename)
        return nib.load(nifti_filepath)

    def get_nifti_data(self, nifti):
        return nifti.get_fdata()

    def get_nifti_filenames(self):
        return os.listdir(self.nifti_dir_path)

    def save_single_nifti_patches(self, nifti):
        self.hf = h5py.File(self.output_file_path, 'w')
        data = self.get_nifti_data(nifti)
        patch_list = self.split_arr_into_patches(data)
        patch_stack = np.stack(patch_list)
        self.hf.create_dataset('patches', data=patch_stack)
        self.hf.close()

    def split_arr_into_patches(self, arr):
        patch_list = []
        rows, columns, depth = arr.shape
        for z in range(depth):
            for y in range(0, rows, self.overlapping):
                for x in range(0, columns, self.overlapping):
                    if x + self.patch_size < columns and y + self.patch_size < rows:  # discard border patches
                        patch = arr[y:y+self.patch_size, x:x+self.patch_size, z]
                        patch_list.append(patch)
        return patch_list

    def save_all_nifti_patches(self):
        for nifti_filename in self.get_nifti_filenames():
            nifti = self.read_nifti(nifti_filename)
            self.standardize_orientation(nifti)
            # self.save_single_nifti_patches(nifti)

    def standardize_orientation(self, nifti):
        for i in range(3):
            if nib.aff2axcodes(nifti.affine)[i] != self.orientation[i]:
                print('flipping axis', i)
                nifti = nib.orientations.flip_axis(nifti, axis=i)
        return nifti

    def does_patch_intersect_roi(self, patch_x, patch_y, roi_mask):
        patch_center_x = patch_x + self.patch_size // 2
        patch_center_y = patch_y + self.patch_size // 2
        if roi_mask[patch_center_y, patch_center_x]:
            return True
        else:
            return False

    @staticmethod
    def get_roi_bbox(arr):
        r = np.any(arr, axis=(1, 2))
        c = np.any(arr, axis=(0, 2))
        z = np.any(arr, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return arr[rmin:rmax+1, cmin:cmax+1, zmin:zmax+1]

    def __del__(self):
        self.hf.close()
        print('closed h5 at:', self.output_file_path)
