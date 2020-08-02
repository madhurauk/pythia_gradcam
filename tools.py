"""Tools."""
import matplotlib.pyplot as plt
import numpy as np
import pdb

# def save_attention_map(attn_map, qid):
#     """Save attn_map image with proper qid extension."""
#     print("Saving attention map...")
#     plt.imshow(attn_map)
#     plt.savefig("./gcam_maps/" + str(qid) + "_1.png")
#     plt.close()


def save_attention_map(attn_map, qid):
    """Save attn_map image with proper qid extension."""
    print("Saving attention map...")
    np.save("./GRADCAM_MAPS/sort_new_gcam_maps_npy/" + str(qid) + ".npy", attn_map)
