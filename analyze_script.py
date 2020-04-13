from analyze.analyze_utils import *
from analyze.post_processing_utils import *
import matplotlib.pyplot as plt
import random
import json
import cv2


# # save probability map as thresholded mask:
# if __name__ == '__main__':
#     output_dir = '/cs/labs/josko/asherp7/follow_up/outputs/'
#     path_to_probability_map = os.path.join(output_dir, 'BL11.nii.gz')
#     mask_output_filepath = os.path.join(output_dir, 'BL11_predicted_mask.nii.gz')
#     threshold = 0.933
#     save_probability_map_as_thresholded_mask(path_to_probability_map, mask_output_filepath, threshold)


# # compute dice score for all predictions after applying Chan Vesse algorithm:
# if __name__ == '__main__':
#     prediction_dir_path = '/cs/labs/josko/asherp7/follow_up/outputs/validation_cnn_predictions_5_4_2020_2020-04-05_11-13-12/'
#     data_dir_path = '/cs/labs/josko/asherp7/follow_up/data_3_4_2020'
#     split = 'validation'
#     # tumor_seg_dir_path = '/cs/labs/josko/asherp7/follow_up/all_combined_data/tumors'
#     # print_tumor_burden_per_ct(tumor_seg_dir_path)
#     dice_dict = analyze_dataset(data_dir_path, split, prediction_dir_path)
#     print('mean dice:', round(sum([x for x in dice_dict.values()]) / len(dice_dict), 4))
#
#     # save dice dict:
#     output_filename = os.path.join(prediction_dir_path, 'prediction_dice.json')
#     with open(output_filename, 'w') as f:
#         json.dump(dice_dict, f)


# compute dice score for all predictions after thresholding and removing small connected components:
if __name__ == '__main__':
    split = 'train'
    # prediction_path = '/cs/labs/josko/asherp7/follow_up/outputs/pred_2020-03-26_10-20-24'
    # data_dir_path = '/mnt/local/aszeskin/asher/liver_data/seperated_26_3'
    # prediction_path = '/cs/labs/josko/asherp7/follow_up/outputs/validation_cnn_predictions_1_4_2020_2020-04-01_01-57-47'
    # prediction_path = '/cs/labs/josko/asherp7/follow_up/outputs/train_cnn_predictions_1_4_2020_2020-04-01_02-21-07/'
    # data_dir_path = '/cs/labs/josko/asherp7/follow_up/data_31_3_2020'
    prediction_path = '/cs/labs/josko/asherp7/follow_up/outputs/train_cnn_predictions_5_4_2020_2020-04-05_12-10-11'
    data_dir_path = '/cs/labs/josko/asherp7/follow_up/data_3_4_2020'
    # prediction_path = '/cs/labs/josko/asherp7/follow_up/outputs/validation_cnn_predictions_5_4_2020_2020-04-05_11-13-12/'
    min_size = 80
    # min_size = 100
    threshold = 933
    # threshold = 900
    save_path = os.path.join(prediction_path, 'threshold_'+str(threshold)+'_cnn_predictions')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    dice_dict = analyze_dataset_after_threshold_and_filter_small_components(prediction_path,
                                                                            data_dir_path,
                                                                            min_size,
                                                                            threshold=threshold,
                                                                            save_path=save_path,
                                                                            split=split)
    print(dice_dict)
    output_filename = os.path.join(prediction_path, 'prediction_dice.json')
    with open(output_filename, 'w') as f:
        json.dump(dice_dict, f)


# # check dice coefficient of prediction:
# if __name__ == '__main__':
#     output_path = '/cs/labs/josko/asherp7/follow_up/outputs'
#     path_to_prediction = os.path.join(output_path, 'BL11.nii.gz')
#     path_to_tumor_segmentation = '/cs/labs/josko/asherp7/example_cases/case11/BL/BL11_Tumors.nii.gz'
#     probabilty_map = nib.load(path_to_prediction).get_data()
#     annotation = nib.load(path_to_tumor_segmentation).get_data()
#     for thresh in np.linspace(0.7, 1, num=10):
#         prediction = (probabilty_map >= thresh)
#         print('threshold:', thresh, 'dice: ', segmentations_dice(prediction, annotation))



# # test small components removal on synthetic data:
# if __name__ == '__main__':
#     height = 256
#     width = 256
#     img = np.zeros((height, width, 3), dtype=np.uint8)
#     for i in range(10):
#         color = (255, 255, 255)
#         radius = random.randint(1, 30)
#         center = (random.randint(0, width), random.randint(0,height))
#         cv2.circle(img, center, radius, color, thickness=-1, lineType=8, shift=0)
#     binary_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     image_3d = np.stack((binary_img,binary_img))
#     plt.imshow(binary_img)
#     plt.show()
#     img2 = remove_small_connected_componenets_3D(image_3d, min_size=2000)
#     plt.imshow(img2[0])
#     plt.show()

