from transform2h5 import Transform2h5
import h5py
import numpy as np
import nibabel as nib

nifti_dir_path = '/cs/labs/josko/aszeskin/Rafi_Tumor_data/allBL'
output_path = '/mnt/local/aszeskin/asher/transforfm_h5_output'
patch_size = 35
overlapping = 3
transform = Transform2h5(nifti_dir_path, output_path, ('L', 'P', 'S'), patch_size, overlapping)
nifti = transform.read_nifti('BL01.nii.gz')
np.set_printoptions(precision=2, suppress=True)
# print(nifti.affine)
# print(nib.aff2axcodes(nifti.affine))
# print('\n\n')
# canonical_img = nib.as_closest_canonical(nifti)
# print(canonical_img.affine)
# print(nib.aff2axcodes(canonical_img.affine))

# transform.save_all_nifti_patches()

nifti_list = transform.get_nifti_filenames()
for filename in nifti_list:
    nifti = transform.read_nifti(filename)
    canonical_img = nib.as_closest_canonical(nifti)
    print(filename, nib.aff2axcodes(nifti.affine), nib.aff2axcodes(canonical_img.affine))

# print('processing...')
# patches = transform.save_single_nifti_patches(nifti)
# del transform
# hf = h5py.File("/mnt/local/aszeskin/asher/transforfm_h5_output/patches.h5", 'r')


# example_filename = os.path.join(data_path, 'example4d.nii.gz')
# img = nib.load(example_filename)
# print(img.shape)
# print(img.get_data_dtype())
# data = img.get_fdata()
# hdr = img.header
# hdr.get_xyzt_units()

arr = np.zeros([100,100,100])
arr[10:90,20:80,30:70] = 1
transform.get_coarse_roi_bbox(arr)