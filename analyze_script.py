from analyze.analyze_utils import *

if __name__ == '__main__':
    output_path = '/cs/labs/josko/asherp7/follow_up/outputs'
    path_to_prediction = os.path.join(output_path, 'BL11.nii.gz')
    path_to_tumor_segmentation = '/cs/labs/josko/asherp7/example_cases/case11/BL/BL11_Tumors.nii.gz'
    probabilty_map = nib.load(path_to_prediction).get_data()
    annotation = nib.load(path_to_tumor_segmentation).get_data()
    for thresh in np.linspace(0.7, 1, num=10):
        prediction = (probabilty_map >= thresh)
        print('threshold:', thresh, 'dice: ', segmentations_dice(prediction, annotation))
