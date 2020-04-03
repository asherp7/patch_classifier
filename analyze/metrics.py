import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import morphology
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from skimage.segmentation import find_boundaries, mark_boundaries
from scipy.ndimage.measurements import center_of_mass
import logging
from skimage.morphology import remove_small_objects
from scipy.ndimage.measurements import label
# from postprocess import save_difference_volume


def dice(gt_seg, estimated_seg):
    """
    compute dice coefficient
    :param gt_seg:
    :param estimated_seg:
    :return:
    """
    seg1 = np.asarray(gt_seg).astype(np.bool)
    seg2 = np.asarray(estimated_seg).astype(np.bool)

    # Compute Dice coefficient
    intersection = np.logical_and(seg1, seg2)

    return 2. * intersection.sum() / (seg1.sum() + seg2.sum())


def IoU(gt_seg, estimated_seg):
    """
    compute Intersection over Union
    :param gt_seg:
    :param estimated_seg:
    :return:
    """
    seg1 = np.asarray(gt_seg).astype(np.bool)
    seg2 = np.asarray(estimated_seg).astype(np.bool)

    # Compute IOU
    intersection = np.logical_and(seg1, seg2)
    union = np.logical_or(seg1, seg2)

    return intersection.sum() / union.sum()


def VOE(gt_seg, pred_seg):
    """
    compute volumetric overlap error (in percent) = 1 - intersection/union
    :param gt_seg:
    :param pred_seg:
    :return:
    """
    return 1 - IoU(gt_seg, pred_seg)


def seg_ROI_overlap(gt_seg, roi_pred):
    """
    compare ground truth segmentation to predicted ROI, return number of voxels from gt seg that aren't contained in
    the predicted ROI
    :param gt_seg: segmentation
    :param roi_pred: ROI represented as a binary segmentation
    :return:
    """
    seg = np.asarray(gt_seg).astype(np.bool)
    seg_roi = np.asarray(roi_pred).astype(np.bool)

    # if segmentation is bigger than intersection seg_roi, we are out of bounds
    intersection = np.logical_and(seg, seg_roi)
    return np.sum(seg ^ intersection)


def load_img(dir_path, img_name='data_CT.nii.gz.gz'):
    img_path = os.path.join(dir_path, img_name)
    img.nii.gz = nib.load(img_path)
    img_seg = img.nii.gz.get_data()
    return img_seg


def load_segs(dir_path, pred_path, pred_name='prediction.nii.gz.gz', truth_name='Tumors.nii.gz.gz', load_pix_dims=False,
              truth_dir_path=None):
    """
    load ground truth segmentation and predicted segmentation from a prediction directory
    :param dir_path: directory where the files of prediction and gt are saved
    :param pred_name: filename of the predicted segmentation
    :param truth_name: filename of the ground truth segmentation
    :param load_pix_dims: if True, will also return the pixel spacing data of this case.
    :param truth_dir_path: directory where the gt file is located, if None, assumed to be the same directory where the
    prediction file is
    :return: (gt_seg, predicted_seg) or (gt_seg, predicted_seg, pix_dims)
    """
    pred_full_path = os.path.join(dir_path, pred_path, pred_name)
    pred = nib.load(pred_full_path)
    pred_seg = pred.get_data()

    if not truth_dir_path:
        truth_dir_path = dir_path
        truth_name = "Tumors.nii.gz"
    gt_path = os.path.join(truth_dir_path, pred_path, truth_name)
    gt = nib.load(gt_path)
    gt_seg = gt.get_data()

    if load_pix_dims:
        pix_dims = gt.header.get_zooms()
        return gt_seg, pred_seg, pix_dims

    return gt_seg, pred_seg


def print_dice_case(case_name, dice_score):
    print(str(dice_score) + "\t" + case_name + "\\\\")
    print("--------------------------------------")


def print_dice_dict(dice_dict):
    print("Dice coefficient\tcase_name")
    print("=======================================")
    sorted_dict = sorted(dice_dict.items(), key=lambda kv: kv[1], reverse=True)
    scores = []
    for case_name, dice_score in sorted_dict:
        print_dice_case(case_name, dice_score)
        scores.append(dice_score)
    print_dice_case("mean score", np.mean(scores))
    print_dice_case("mean w.o. worst case", np.mean(scores[:-1]))


def print_overlap_case(case_name, overlap):
    print(str(overlap) + "\t" + case_name + "\\\\")
    print("--------------------------------------")


def pring_overlap_dict(overlap_dict):
    print("Overlap\tcase_name")
    print("=======================================")
    for case_name, overlap in overlap_dict.items():
        print_dice_case(case_name, overlap)


def print_surfd_case(case_name, mean_surfd, hausdorf):
    print('%.2f\t%.2f\t' % (mean_surfd, hausdorf) + case_name + "\\\\")
    print("--------------------------------------")


def print_worst_msd_case(case_name, msd, msd_slice):
    print(case_name + "&%.1f&%d" % (msd, msd_slice) + "\\\\")
    print("\hline")


def print_surfd_and_dice_case(case_name, dice_score, mean_surfd, hausdorf):
    print(case_name+'&%.2f&%.1f&%.1f' % (dice_score, mean_surfd, hausdorf) + "\\\\")
    print("\hline")
    # print("--------------------------------------")


def print_surfd_and_dice_case_with_mean(case_name, dice_score, mean_surfd, max_surfd, mean_max_surfd, mean_big_msd,
                                        logger=None):
    print(case_name + '&%.2f&%.1f&%.1f&%.1f&%.1f' % (dice_score, mean_surfd, max_surfd, mean_max_surfd, mean_big_msd)
          + "\\\\")
    print("\hline")

    if logger:
        logger.debug(case_name + '&%.2f&%.1f&%.1f&%.1f&%.1f' % (dice_score, mean_surfd, max_surfd, mean_max_surfd,
                                                                mean_big_msd) + "\\\\")
        logger.debug("\hline")


def print_surfd_and_dice_case_2d(case_name, mean_dice_score, min_dice, max_dice, mean_surfd, max_surfd, slice_num,
                                 mean_max_surfd, msd_bigobj, msd_bigobj_slice, underseg_rate, logger=None):
    print(case_name + '&%.4f&%.2f&%.2f&%.1f&%.1f&%d&%.1f&%.1f&%d&%.2f' %
          (mean_dice_score, min_dice, max_dice, mean_surfd, max_surfd, slice_num, mean_max_surfd, msd_bigobj,
           msd_bigobj_slice, underseg_rate)
          + "\\\\")
    print("\hline")
    if logger:
        logger.debug(case_name + '&%.4f&%.2f&%.2f&%.1f&%.1f&%d&%.1f&%.1f&%d&%.2f' %
                     (mean_dice_score, min_dice, max_dice, mean_surfd, max_surfd, slice_num, mean_max_surfd, msd_bigobj,
                      msd_bigobj_slice, underseg_rate) + "\\\\")
        logger.debug("\hline")


def print_dice_surfd_voe_case(case_name, dice_score, mean_surfd, hausdorf, voe, logger=None):
    print(case_name+'&%.4f&%.3f&%.3f&%.3f' % (dice_score, mean_surfd, hausdorf, voe) + "\\\\")
    print("\hline")

    if logger:
        logger.debug(case_name + '&%.4f&%.3f&%.3f&%.3f' % (dice_score, mean_surfd, hausdorf, voe) + "\\\\")
        logger.debug("\hline")

    # print("--------------------------------------")


def print_surfd_dict(surfd_dict):
    print("Mean SD\tMax SD\tcase_name")
    print("=======================================")
    sorted_dict = sorted(surfd_dict.items(), key=lambda kv: kv[1].mean(), reverse=True)
    scores = []
    for case_name, surfd in sorted_dict:
        print_surfd_case(case_name, surfd.mean(), surfd.max())
        scores.append([surfd.mean(), surfd.max()])
    scores = np.asarray(scores)
    print_surfd_case("mean score", np.mean(scores[:, 0]), np.mean(scores[:, 1]))
    print_surfd_case("mean w.o. worst case", np.mean(scores[:-1, 0]), np.mean(scores[:-1, 1]))


def print_surfd_and_dice_dicts(surfd_dict, dice_dict):
    print("case_name\t\tDice\tMean SD\tMax SD")
    print("=======================================")
    sorted_dict = sorted(dice_dict.items(), key=lambda kv: kv[1], reverse=True)
    scores = []
    for case_name, dice_score in sorted_dict:
        surfd = surfd_dict[case_name]
        print_surfd_and_dice_case(case_name, dice_score, surfd.mean(), surfd.max())
        scores.append([dice_score, surfd.mean(), surfd.max()])
    scores = np.asarray(scores)
    print_surfd_and_dice_case("mean score", np.mean(scores[:, 0]), np.mean(scores[:, 1]), np.mean(scores[:, 2]))
    print_surfd_and_dice_case("stds", np.std(scores[:, 0]), np.std(scores[:, 1]), np.std(scores[:, 2]))
    print_surfd_and_dice_case("mean w.o. worst case", np.mean(scores[:-1, 0]),
                              np.mean(scores[:-1, 1]), np.mean(scores[:-1, 2]))
    print_surfd_and_dice_case("stds w.o. worst case", np.std(scores[:-1, 0]),
                              np.std(scores[:-1, 1]), np.std(scores[:-1, 2]))


def print_dice_voe_surfd_dicts(dice_dict, voe_dict, surfd_dict, logger=None):
    print("case_name\t\tDice\tMean SD\tMax SD\tVOE")
    print("=======================================")

    if logger:
        logger.debug("case_name\t\tDice\tMean SD\tMax SD\tVOE")
        logger.debug("=======================================")

    sorted_dict = sorted(dice_dict.items(), key=lambda kv: kv[1], reverse=True)
    scores = []
    for case_name, dice_score in sorted_dict:
        surfd = surfd_dict[case_name]
        voe = voe_dict[case_name]
        print_dice_surfd_voe_case(case_name, dice_score, surfd.mean(), surfd.max(), voe, logger=logger)
        scores.append([dice_score, surfd.mean(), surfd.max(), voe])
    scores = np.asarray(scores)
    print_dice_surfd_voe_case("mean score", np.mean(scores[:, 0]), np.mean(scores[:, 1]), np.mean(scores[:, 2]),
                              np.mean(scores[:, 3]), logger=logger)
    print_dice_surfd_voe_case("stds", np.std(scores[:, 0]), np.std(scores[:, 1]), np.std(scores[:, 2]),
                              np.std(scores[:, 3]), logger=logger)
    print_dice_surfd_voe_case("mean w.o. worst case", np.mean(scores[:-1, 0]),
                              np.mean(scores[:-1, 1]), np.mean(scores[:-1, 2]), np.mean(scores[:-1, 3]), logger=logger)
    print_dice_surfd_voe_case("stds w.o. worst case", np.std(scores[:-1, 0]),
                              np.std(scores[:-1, 1]), np.std(scores[:-1, 2]), np.std(scores[:-1, 3]), logger=logger)


def print_2d_dice_surfd_dicts(sorted_dict, surfd_dict, slices_dict, underseg_dict, msd_bigobj_dict, logger=None):
    print("case_name  Mean Dice  Min Dice  Max Dice  Mean ASSD (mm)  Max MSD (mm)  Max MSD Slice  Mean MSD  BigObjMSD  "
          "BMSD slice  % Under Seg.")
    print("=====================================================")

    if logger:
        logger.debug("case_name  Mean Dice  Min Dice  Max Dice  Mean ASSD (mm)  Max MSD (mm)  Max MSD Slice  Mean MSD  "
                     "BigObjMSD  BMSD slice  % Under Seg.")
        logger.debug("=====================================================")

    scores = []
    for case_name, dice_list in sorted_dict:
        surfd_list = surfd_dict[case_name]
        msd_bigobj_list = msd_bigobj_dict[case_name]
        slices_list = slices_dict[case_name]
        dice_array = np.asarray(dice_list)

        # calculating mean and max from each list
        mean_surfd_per_slice = np.asarray([surfd.mean() for surfd in surfd_list])
        max_surfd_per_slice = np.asarray([surfd.max() for surfd in surfd_list])
        msd_bigobj_per_slice = np.asarray([surfd.max() for surfd in msd_bigobj_list])

        dice_mean = dice_array.mean()
        dice_min = dice_array[dice_array.nonzero()].min()
        dice_max = dice_array.max()

        argmax_surfd = max_surfd_per_slice.argmax()
        max_surfd_slice = slices_list[argmax_surfd]
        max_surfd = max_surfd_per_slice[argmax_surfd]
        mean_max_surfd = max_surfd_per_slice.mean()

        argmax_bigobj_surfd = msd_bigobj_per_slice.argmax()
        msd_bigobj_slice = slices_list[argmax_bigobj_surfd]
        msd_bigobj = msd_bigobj_per_slice[argmax_bigobj_surfd]

        underseg_rate = underseg_dict[case_name] / len(slices_list)

        print_surfd_and_dice_case_2d(case_name, dice_mean, dice_min, dice_max, mean_surfd_per_slice.mean(),
                                     max_surfd, max_surfd_slice, mean_max_surfd, msd_bigobj, msd_bigobj_slice,
                                     underseg_rate, logger=logger)
        scores.append([dice_mean, mean_surfd_per_slice.mean(), max_surfd, mean_max_surfd, msd_bigobj, underseg_rate])

    scores = np.asarray(scores)
    print_surfd_and_dice_case_with_mean("mean score", np.mean(scores[:, 0]), np.mean(scores[:, 1]),
                                        np.mean(scores[:, 2]), np.mean(scores[:, 3]), np.mean(scores[:, 4]),
                                        logger=logger)
    print_surfd_and_dice_case_with_mean("stds", np.std(scores[:, 0]), np.std(scores[:, 1]), np.std(scores[:, 2]),
                                        np.std(scores[:, 3]), np.mean(scores[:, 4]), logger=logger)
    print_surfd_and_dice_case_with_mean("mean w.o. worst case", np.mean(scores[:-1, 0]),
                                        np.mean(scores[:-1, 1]), np.mean(scores[:-1, 2]), np.mean(scores[:-1, 3]),
                                        np.mean(scores[:-1, 4]), logger=logger)
    print_surfd_and_dice_case_with_mean("stds w.o. worst case", np.std(scores[:-1, 0]),
                                        np.std(scores[:-1, 1]), np.std(scores[:-1, 2]), np.std(scores[:-1, 3]),
                                        np.mean(scores[:-1, 4]), logger=logger)

    print("mean undersegmentation rate: %.2f" % np.mean(scores[:, -1]))
    print("mean undersegmentation rate w.o. worst: %.2f" % np.mean(scores[:-1, -1]))

    if logger:
        logger.debug("mean undersegmentation rate: %.2f" % np.mean(scores[:, -1]))
        logger.debug("mean undersegmentation rate w.o. worst: %.2f" % np.mean(scores[:-1, -1]))


def display_dice_histograms(sorted_dict, nrows=5, title="Dice per slice histograms", bins=np.arange(0.0, 1.1, 0.1)):
    nhists = len(sorted_dict)
    ncols = int(np.ceil(nhists / nrows))

    if nhists > 10:
        f, a = plt.subplots(nrows, ncols, sharex='col', sharey='row')
    else:
        f, a = plt.subplots(nrows, ncols)

    f.suptitle(title, fontsize=18)
    a = a.ravel()
    for ind, (case_name, dice_list) in enumerate(sorted_dict):
        # subplot
        ax = a[ind]
        # display histogram
        dice_array = np.asarray(dice_list)
        ax.hist(dice_array, bins=bins, rwidth=0.9, align='left')
        ax.set_title(case_name + ": %.2f" % (dice_array.mean()))
        ax.set_xlabel('Dice')
        ax.set_ylabel('Slice Count')

    plt.show()


def slice_distances(seg):
    """
    measure distances between center of mass in segmentation in slice (2D) and boundary of the segmentation.
    :param seg: 2D numpy binary array of a segmented object
    :return: list of all lengths of rays (list length = number of points on segmentation contour)
    """
    # find center of mass of this slice
    center = np.asarray(center_of_mass(seg))[np.newaxis]
    # find boundaries of this slice
    boundaries = np.nonzero(find_boundaries(seg, mode='inner'))
    boundaries_coords = np.asarray([np.asarray((boundaries[0][x], boundaries[1][x]))
                                    for x in range(len(boundaries[0]))])
    distances = cdist(center, boundaries_coords)
    return distances[0]


def display_slice_distances_hist(seg_slice, case_name=''):
    distances = slice_distances(seg_slice)
    plt.hist(distances, rwidth=0.9, align='left')
    plt.title("distances from center of mass, %s" % (case_name))
    plt.show()


def display_multiple_distances_hist(seg_3d, slices_list, nrows=3, title='2D Distance from Center Of Mass Histograms',
                                    case_name='', save=False, path='./'):
    """
    display a figure with multiple histograms of distances between segmentation border and center of mass in given 3D
    segmentation at the given slices (everything works in 2D)
    :param seg_3d: the segmentation volume
    :param slices_list: list of slices we want to display the distances for
    :param nrows: number of rows in the figure
    :param title: figure title
    :param case_name: case name, will be added to figure title
    :param save: if True, will save figure to given path instead of displaying it to the screen
    :param path: path to save figure
    :return:
    """
    nhists = len(slices_list)
    ncols = int(np.ceil(nhists / nrows))
    f, a = plt.subplots(nrows, ncols)

    f.suptitle(title + ': ' + case_name, fontsize=18)
    a = a.ravel()
    for ind, slice_num in enumerate(slices_list):
        # subplot
        ax = a[ind]
        # calculate histogram
        distances = slice_distances(seg_3d[:, :, slice_num])
        # display histogram
        ax.hist(distances, rwidth=0.9, align='left')
        ax.set_title("slice: %d" % (slice_num))
        ax.set_xlabel('Distance')
        ax.set_ylabel('Point Count')

    if save:
        f.set_size_inches((20, 15), forward=False)
        plt.savefig(path+case_name+'.png')
    else:
        plt.show()


def display_slice(vol_slice, gt_seg_slice, pred_seg_slice, savefig_name=None):
    norm_vol_slice = vol_slice.copy()
    norm_vol_slice -= norm_vol_slice.min()
    norm_vol_slice /= norm_vol_slice.max()
    border_gt_img = mark_boundaries(norm_vol_slice, gt_seg_slice)
    border_pred_img = mark_boundaries(norm_vol_slice, pred_seg_slice)
    f, a = plt.subplots(1, 2)
    a[0].imshow(border_gt_img)
    a[0].set_title("Ground Truth")
    a[1].imshow(border_pred_img)
    a[1].set_title("Predicted Segmentation")
    if savefig_name:
        f.set_size_inches((20, 15), forward=False)
        plt.savefig(savefig_name)
    else:
        plt.show()


def evaluate_ROI_prediction(pred_dir, pred_name="predicted_ROI.nii.gz.gz"):
    pred_paths = os.listdir(pred_dir)
    score_dict = {}
    for pred_path in pred_paths:
        gt_seg, _ = load_segs(os.path.join(pred_dir, pred_path),
                              pred_name="prediction.nii.gz.gz")
        roi_pred = load_img(os.path.join(pred_dir, pred_path), pred_name)

        score_dict[pred_path] = seg_ROI_overlap(gt_seg, roi_pred)

    pring_overlap_dict(score_dict)


def surf_dist(pred_seg, gt_seg, sampling=1, connectivity=1):
    """
    from https://mlnotebook.github.io/post/surface-distance-function/
    Calculates and returns the surface distance between the Ground Truth segmentation and the predicted one.
    The surface distance is a vector with length as len(contour(pred_seg)) that indicates for every pixel on the contour,
    its distance from the closest pixel on the contour of the ground truth in euclidean norm.
    :param pred_seg: the segmentation that has been created
    :param gt_seg: the GT segmentation against which we wish to compare pred_seg
    :param sampling: the pixel resolution or pixel size. This is entered as an n-vector where n is equal to the number
    of dimensions in the segmentation i.e. 2D or 3D. The default value is 1 which means pixels (or rather voxels)
    are 1 x 1 x 1 mm in size
    :param connectivity: creates either a 2D (3 x 3) or 3D (3 x 3 x 3) matrix defining the neighbourhood around which
    the function looks for neighbouring pixels. Typically, this is defined as a six-neighbour kernel which is the
    default behaviour of this function.
    :return: surface distance vector
    """
    pred_seg = np.atleast_1d(pred_seg.astype(np.bool))
    gt_seg = np.atleast_1d(gt_seg.astype(np.bool))

    conn = morphology.generate_binary_structure(pred_seg.ndim, connectivity)

    S = pred_seg ^ morphology.binary_erosion(pred_seg, conn)
    Sprime = gt_seg ^ morphology.binary_erosion(gt_seg, conn)

    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])
    return sds


def get_msd_without_small_objects(pred_slice, gt_slice, min_obj_size, pix_dims=1):
    """
    return min{msd(pred_seg_big_objects, gt_seg), msd(pred_seg, gt_seg_big_objects)}, meaning, the maximum surface
    distance of this slice, excluding small objects with size less than min_obj_size pixels.
    :param pred_slice:
    :param gt_slice:
    :param min_obj_size:
    :param pix_dims:
    :return:
    """
    pred_slice_labels = label(pred_slice)[0]
    pred_bigobj = remove_small_objects(pred_slice_labels, min_size=min_obj_size, connectivity=2)
    if pred_bigobj.sum():
        msd1 = surf_dist(pred_bigobj, gt_slice, sampling=pix_dims[:-1]).max()
    else:
        msd1 = surf_dist(pred_slice, gt_slice, sampling=pix_dims[:-1]).max()

    gt_slice_labels = label(gt_slice)[0]
    gt_bigobj = remove_small_objects(gt_slice_labels, min_size=min_obj_size, connectivity=2)
    if gt_bigobj.sum():
        msd2 = surf_dist(gt_bigobj, pred_slice, sampling=pix_dims[:-1]).max()
    else:
        msd2 = surf_dist(gt_slice, pred_slice, sampling=pix_dims[:-1]).max()

    if pred_bigobj.sum() and gt_bigobj.sum():
        msd3 = surf_dist(pred_bigobj, gt_bigobj, sampling=pix_dims[:-1]).max()
    else:
        msd3 = msd2

    return np.min([msd1, msd2, msd3])


def evaluate_segmentation_prediction_2d_measures(pred_dir, pred_name="prediction_postprocessed_3d.nii.gz.gz",
                                                 distance_hist_savefig_dir=None, worst_slice_savefig_dir=None,
                                                 truth_dir_path=None, logger=None, min_obj_size=None):
    """
    function for evaluating the predicted segmentation using 2d measures: per slice dice, per slice ASSD, per slice MSD,
    etc.
    prints the results to the screen.
    :param pred_dir: path for the directory where the prediction is saved
    :param pred_name: name of the file we want to evaluate
    """
    if logger:
        logger.debug("starting evaluation for %s" % pred_dir)
    pred_paths = os.listdir(pred_dir)

    dice_dict = {}
    surfd_dict = {}
    slices_surfd_dict = {}  # index for slices
    under_seg_dict = {}  # per slice count of over segmented slices
    msd_bigobj_dict = {}  # surface distances excluding small objects

    for pred_path in pred_paths:
        try:
            gt_seg, pred_seg, pix_dims = load_segs(pred_dir, pred_path=pred_path, pred_name=pred_name,
                                                   load_pix_dims=True, truth_dir_path=truth_dir_path)

            dice_dict[pred_path] = []
            surfd_dict[pred_path] = []
            slices_surfd_dict[pred_path] = []
            msd_bigobj_dict[pred_path] = []
            under_seg_dict[pred_path] = 0
            for slice_num in np.unique(gt_seg.nonzero()[-1]):
                gt_slice, pred_slice = gt_seg[:, :, slice_num], pred_seg[:, :, slice_num]
                dice_dict[pred_path].append(dice(gt_slice, pred_slice))
                if not gt_slice.any() or not pred_slice.any():
                    continue
                surfd_dict[pred_path].append(surf_dist(pred_slice, gt_slice, sampling=pix_dims[:-1]))
                msd_bigobj_dict[pred_path].append(get_msd_without_small_objects(pred_slice,
                                                                                gt_slice,
                                                                                min_obj_size=min_obj_size,
                                                                                pix_dims=pix_dims))

                slices_surfd_dict[pred_path].append(slice_num)
                if gt_slice.sum() > pred_slice.sum():
                    under_seg_dict[pred_path] += 1

            # display distance histograms
            slices_uni = np.unique(pred_seg.nonzero()[-1])
            spacing = len(slices_uni) // 12
            slices_list = slices_uni[::spacing]
            if distance_hist_savefig_dir:
                display_multiple_distances_hist(pred_seg, slices_list, case_name=pred_path, save=True,
                                                path=distance_hist_savefig_dir)
            if worst_slice_savefig_dir:
                vol = load_img(os.path.join(pred_dir, pred_path))
                surfd_list = surfd_dict[pred_path]
                slices_list = slices_surfd_dict[pred_path]
                max_surfd_per_slice = np.asarray([surfd.max() for surfd in surfd_list])
                argmax_surfd = max_surfd_per_slice.argmax()
                max_surfd_slice = slices_list[argmax_surfd]
                display_slice(vol[:, :, max_surfd_slice], gt_seg[:, :, max_surfd_slice],
                              pred_seg[:, :, max_surfd_slice],
                              savefig_name=os.path.join(worst_slice_savefig_dir, pred_path)+'.png')

        except Exception as e:
            print("exception %s occured" % e)

    # sort dictionary according to mean dice per case
    sorted_dict = sorted(dice_dict.items(), key=lambda kv: np.mean(kv[1]), reverse=True)

    # print dice and surface distances table
    print_2d_dice_surfd_dicts(sorted_dict, surfd_dict, slices_surfd_dict, under_seg_dict, msd_bigobj_dict,
                              logger=logger)

    # # display all the dice per slice histograms
    # display_dice_histograms(sorted_dict)
    #
    # # display dice histograms of worst histograms
    # d = 6
    # display_dice_histograms(sorted_dict[-d:], nrows=2)


def evaluate_segmentation_prediction(pred_dir, pred_name="prediction_postprocessed_3d.nii.gz.gz", truth_dir_path=None,
                                     logger=None):
    """
    function for evaluating the predicted segmentation using 3d measures.
    prints the results to the screen.
    :param pred_dir: path for the directory where the prediction is saved
    :param pred_name: name of the file we want to evaluate
    """
    print("starting evaluation for %s with pred_name=%s" % (pred_dir, pred_name))
    # test_cases_contrast_outliers = ['FU4', 'FU24', 'FU3', 'FU29', 'FU16', 'FU26', 'FU10', 'FU37', 'FU9']
    if logger:
        logger.debug("starting evaluation for %s" % pred_dir)
    pred_paths = os.listdir(pred_dir)
    dice_dict = {}
    surfd_dict = {}
    voe_dict = {}

    # dice_outliers_dict = {}
    # surfd_outliers_dict = {}
    # voe_outliers_dict = {}
    try:
        for pred_path in pred_paths:
            gt_seg, pred_seg, pix_dims = load_segs(pred_dir, pred_path=pred_path, pred_name=pred_name,
                                                   load_pix_dims=True, truth_dir_path=truth_dir_path)
            # if pred_path in test_cases_contrast_outliers:
            #     dice_outliers_dict[pred_path] = dice(gt_seg, pred_seg)
            #     surfd_outliers_dict[pred_path] = surf_dist(pred_seg, gt_seg)  # , sampling=pix_dims)
            #     voe_outliers_dict[pred_path] = VOE(gt_seg, pred_seg)
            #     continue

            dice_dict[pred_path] = dice(gt_seg, pred_seg)
            surfd_dict[pred_path] = surf_dist(pred_seg, gt_seg, sampling=pix_dims)
            voe_dict[pred_path] = VOE(gt_seg, pred_seg)

            # # isolate the case where post processing is bad
            # _, pred_seg_unprocessed, _ = load_segs(os.path.join(prediction_dir, pred_path),
            #                                        pred_name="prediction.nii.gz.gz", load_pix_dims=True)
            # dice_unprocessed = dice(gt_seg, pred_seg_unprocessed)
            # if dice_unprocessed > dice_dict[pred_path]:
            #     print("Case %s is worsened by postprocessing by %.2f" %
            #           (pred_path, dice_unprocessed - dice_dict[pred_path]))
    except Exception as e:
        if logger:
            logger.critical(e)
        print(e)

    print_dice_voe_surfd_dicts(dice_dict, voe_dict, surfd_dict, logger=logger)


def main(logger, prediction_dirs, orig_data_dir, pred_name="prediction_post_reshaped.nii.gz.gz"):
    try:
        for prediction_dir in prediction_dirs:
            evaluate_segmentation_prediction(prediction_dir, pred_name=pred_name,
                                             truth_dir_path=orig_data_dir, logger=logger)
            evaluate_segmentation_prediction_2d_measures(prediction_dir,
                                                         pred_name=pred_name,
                                                         truth_dir_path=orig_data_dir, min_obj_size=64, logger=logger)
    except Exception as e:
        logger.critical(e)
        pass


if __name__ == "__main__":
    # orig_data_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/data/preprocessed_all_cases_cropped_with_prior/"
    pred_dirs = "/cs/labs/josko/asherp7/follow_up/outputs/pred_2020-03-26_10-20-24/grouped_results/"
    orig_data_dir = ''
    parser = argparse.ArgumentParser(description="arguments for training process")
    parser.add_argument("--pred_dirs", dest='pred_dirs', help="list of prediction directories to evaluate",
                        nargs='+', default=[pred_dirs])
    parser.add_argument("--orig_dir", dest="orig_dir", help="name of original data directory", default=orig_data_dir)
    parser.add_argument("--log", dest="log_name", help="name of the logger file", default="logs/evaluate.log")
    parser.add_argument("--pred_name", dest="pred_name", help="name of the prediction file to evaluate",
                        default="prediction.nii.gz")

    # parse args
    args = parser.parse_args()

    ################
    # SEGMENTATION #
    ################

    # prediction_dir = os.path.abspath("prediction_small_training/")
    # prediction_dir = "/cs/casmip/clara.herscu/git/3DUnet/train_prediction"
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/cropping_prediction/"
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/prediction_all_cases_training/"

    # with subject ids
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/prediction_all_cases_training_with_sids_2/"  # best model!
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/prediction_small_training_with_sids/"
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/prediction_all_cases_training_50_epochs_with_sids/"
    # prediction_dirs = []
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/prediction_segmentation_all_cases_like_ROI/"
    # prediction_dirs.append(prediction_dir)
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/prediction_segmentation_all_cases_like_ROI_focal_loss/"  # which post? e3d5?
    # prediction_dirs.append(prediction_dir)
    # # savefig_hists_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/distance_hists_all_like_ROI/"
    # # savefig_worst_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/worst_slice_all_like_ROI/"
    #
    # # with augmentations
    # # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/prediction_segmentation_all_cases_like_ROI_augmentation/"
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/prediction_segmentation_all_cases_like_ROI_rot_aug/"
    # prediction_dirs.append(prediction_dir)
    # # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/prediction_segmentation_all_cases_like_ROI_deform_aug_10_eps/"
    # # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/prediction_segmentation_all_deform_rot_aug/"
    #
    # # other architectures
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/predict_rrunet_rot_aug/"  # e3d3 best
    # prediction_dirs.append(prediction_dir)
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/predict_rrunet_rot_aug_focal/"  # e3d3 best
    # prediction_dirs.append(prediction_dir)
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/predict_rrunet_no_aug/"  # e3d3 best
    # prediction_dirs.append(prediction_dir)
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/predict_rrunet_no_aug_focal/" # e3d3 best
    # prediction_dirs.append(prediction_dir)
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/predict_unet_rot_aug_focal/"  # e3d5 best
    # prediction_dirs.append(prediction_dir)
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/predict_resunet_no_aug/"  # e3d3 best
    # prediction_dirs.append(prediction_dir)
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/predict_resunet_rot_aug/"  # e3d3 best
    # prediction_dirs.append(prediction_dir)
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/predict_resunet_rot_aug_focal/"  # e3d3 best
    # prediction_dirs.append(prediction_dir)
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/predict_resunet_no_aug_focal/"  # e3d3 best
    # prediction_dirs.append(prediction_dir)
    #
    # # model with prior
    # # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/predict_rrunet_rot_aug_with_prior_50_epochs/"
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/predict_rrunet_rot_aug_with_prior/"
    # prediction_dirs.append(prediction_dir)
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/predict_rrunet_rot_aug_with_prior_focal/"
    # prediction_dirs.append(prediction_dir)

    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/predict_rrunet_rot_aug_weak_prior/"
    # prediction_dirs.append(prediction_dir)

    # test prediction
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/prediction_test_FU/all_cases_train/"  # best model!
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/prediction_test_FU/small_train/" # small model

    # create logger
    logger = logging.getLogger('predict_script')
    logger.setLevel(logging.DEBUG)
    # fh = logging.FileHandler("/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/lits_metrics_evaluation_reshaped.log")
    fh = logging.FileHandler("/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/"+args.log_name)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    logger.debug('starting run with directories:')
    logger.debug(args.pred_dirs)

    logger.debug('evaluating prediction named: %s' % args.pred_name)

    main(logger, args.pred_dirs, args.orig_dir, pred_name=args.pred_name)

    #######
    # ROI #
    #######
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/prediction_all_cases_training/"
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/cropping_prediction_all_cases_training/"
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/cropping_prediction_small_training/"

    # with subject ids
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/cropping_prediction_all_cases_training_with_sids/"
    # prediction_dir = "/cs/casmip/clara.herscu/git/AdiUnet/Unet3d/brats/cropping_prediction_small_training_with_sids"

    # evaluate_ROI_prediction(prediction_dir)