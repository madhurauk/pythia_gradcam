"""Get correlation between gradcam maps and human attention maps."""
import skimage.transform as transform
import skimage.filters as filters
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pickle
import math
import json
import cv2

hmap_id_json = '/qids_with_hat.json'
ha_maps_folder = 'vqa_hat_val_new_qid/'
coco_imgs = 'coco/images/val2014/'

gcam_folder_pythia = 'pythia_gcam_maps_npy/'
gcam_folder_squint = 'squint_gcam_maps_npy/'
gcam_folder_sort = 'sort_gcam_maps_npy/'


def save_img(x, name):
    """Save image x with name."""
    plt.imshow(x)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(name)
    plt.close()


def normalize(x):
    """Normalize to range 0-1."""
    normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
    return normalized


def overlay(img_np, att_map, blur=True, overlap=True):
    """Superimpose image on attention map."""
    # Rescale image to 0..1
    img = normalize(img_np)
    att_map = normalize(att_map)
    # att_map = transform.resize(att_map, (img.shape[:2]), order = 3, mode = 'nearest')
    att_map = transform.resize(att_map, (img.shape[:2]), order=3, mode='constant')
    if blur:
        att_map = filters.gaussian(att_map, 0.02 * max(img.shape[:2]))
        att_map -= att_map.min()
        att_map /= att_map.max()
    cmap = plt.get_cmap('jet')
    att_map_v = cmap(att_map)
    att_map_v = np.delete(att_map_v, 3, 2)
    if overlap:
        att_map = 1 * (1 - att_map**0.7).reshape(att_map.shape + (1,)) * img + (att_map**0.7).reshape(att_map.shape + (1,)) * att_map_v
    return att_map


if __name__ == "__main__":

    with open(hmap_id_json) as f_in:
        hids = json.load(f_in)

    corr_pears_pythia, corr_pears_squint, corr_pears_sort = [], [], []
    corr_spear_pythia, corr_spear_squint, corr_spear_sort = [], [], []
    c = 0

    for idx, hid in enumerate(hids):

        print("%d/%d done" % (idx, len(hids)))

        # LOAD GRADCAM IMAGE
        gimg_pythia = np.load(gcam_folder_pythia + str(hid) + '.npy')
        gimg_squint = np.load(gcam_folder_squint + str(hid) + '.npy')
        gimg_sort = np.load(gcam_folder_sort + str(hid) + '.npy')

        # LOAD HUMAN ATTN IMAGE
        himg = cv2.imread(ha_maps_folder + str(hid) + '_1.png', cv2.IMREAD_GRAYSCALE)
        h, w = gimg_pythia.shape[0], gimg_pythia.shape[1]

        # RESIZE (448, 448)
        himg = cv2.resize(himg, (h, w))    # un comment this for (448, 448)

        # DISPLAY IMAGES
        img = cv2.imread(coco_imgs + 'COCO_val2014_' + str(hid)[:-3].zfill(12) + '.jpg')
        SIZE = 14
        himg = cv2.resize(himg, (SIZE, SIZE))
        gimg_pythia = cv2.resize(gimg_pythia, (SIZE, SIZE))
        gimg_squint = cv2.resize(gimg_squint, (SIZE, SIZE))
        gimg_sort = cv2.resize(gimg_sort, (SIZE, SIZE))

        # DISPLAY STUFF
        # overlay with image
        img = cv2.imread(coco_imgs + 'COCO_val2014_' + str(hid)[:-3].zfill(12) + '.jpg')
        save_img(overlay(img, gimg_pythia), 'overlap_gradcam.png')
        save_img(overlay(img, gimg_squint), 'overlap_gradcam.png')
        save_img(overlay(img, gimg_sort), 'overlap_gradcam.png')
        save_img(overlay(img, himg), 'overlap_human.png')

        # COMPUTE CORRELATION
        if not himg.std() or math.isnan(himg.std()) or not gimg_pythia.std() or math.isnan(gimg_pythia.std()) or not gimg_squint.std() or math.isnan(gimg_squint.std()) or not gimg_sort.std() or math.isnan(gimg_sort.std()):
            c += 1
            continue
        sc_py = stats.spearmanr(himg.flatten(), gimg_pythia.flatten(), nan_policy='raise')[0]
        sc_sq = stats.spearmanr(himg.flatten(), gimg_squint.flatten(), nan_policy='raise')[0]
        sc_so = stats.spearmanr(himg.flatten(), gimg_sort.flatten(), nan_policy='raise')[0]
        print("SPEARMAN:: pythia: %f, squint: %f, sort: %f" % (sc_py, sc_sq, sc_so))

        pc_py = stats.pearsonr(himg.flatten(), gimg_pythia.flatten())[0]
        pc_sq = stats.pearsonr(himg.flatten(), gimg_squint.flatten())[0]
        pc_so = stats.pearsonr(himg.flatten(), gimg_sort.flatten())[0]
        print("PEARSON:: pythia: %f, squint: %f, sort: %f" % (pc_py, pc_sq, pc_so))

        corr_pears_pythia.append(pc_py)
        corr_pears_squint.append(pc_sq)
        corr_pears_sort.append(pc_so)
        corr_spear_pythia.append(sc_py)
        corr_spear_squint.append(sc_sq)
        corr_spear_sort.append(sc_so)

    corr_pears_pythia = np.array(corr_pears_pythia)
    corr_pears_squint = np.array(corr_pears_squint)
    corr_pears_sort = np.array(corr_pears_sort)
    corr_spear_pythia = np.array(corr_spear_pythia)
    corr_spear_squint = np.array(corr_spear_squint)
    corr_spear_sort = np.array(corr_spear_sort)

    print("PEARSON ::")
    print("Pythia: ", np.mean(corr_pears_pythia), corr_pears_pythia.std() / math.sqrt(len(corr_pears_pythia)))
    print("Squint: ", np.mean(corr_pears_squint), corr_pears_squint.std() / math.sqrt(len(corr_pears_squint)))
    print("Sort: ", np.mean(corr_pears_sort), corr_pears_sort.std() / math.sqrt(len(corr_pears_sort)))
    print("\n")
    print("SPEARMAN ::")
    print("Pythia: ", np.mean(corr_spear_pythia), corr_spear_pythia.std() / math.sqrt(len(corr_spear_pythia)))
    print("Squint: ", np.mean(corr_spear_squint), corr_spear_squint.std() / math.sqrt(len(corr_spear_squint)))
    print("Sort: ", np.mean(corr_spear_sort), corr_spear_sort.std() / math.sqrt(len(corr_spear_sort)))

    corr_dict = {}
    corr_dict['corr_pears_pythia'] = corr_pears_pythia
    corr_dict['corr_pears_squint'] = corr_pears_squint
    corr_dict['corr_pears_sort'] = corr_pears_squint
    corr_dict['corr_spear_pythia'] = corr_spear_pythia
    corr_dict['corr_spear_squint'] = corr_spear_squint
    corr_dict['corr_spear_sort'] = corr_spear_squint

    with open('CORRS/corrs_448.pickle', 'wb') as handle:
        pickle.dump(corr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("c = ", c)
