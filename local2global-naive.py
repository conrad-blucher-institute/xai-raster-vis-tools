# Combine XAI ouputs
# Very naive technique

import numpy as np
import pickle
from optparse import OptionParser
import matplotlib
import matplotlib.pyplot as plt
import shap
from shap.plots import colors

from sklearn.preprocessing import minmax_scale

def loadPickle(pickleFile, instanceIdx, classIdx):
    shap_values = pickle.load(open(pickleFile, "rb"))
    return (shap_values[instanceIdx, :, :, :, classIdx].values, shap_values[instanceIdx, :, :, :, classIdx].data,
            shap_values[instanceIdx, :, :, :, :].base_values, shap_values.output_names)

def main():
    parser = OptionParser()
    parser.add_option("-p", "--pickle_file",
                      help="Path to pickled SHAP values.",
                      default = None)
    parser.add_option("-i", "--instances",
                      help="Comma-delimited list instances to combine. By default, combine all.",
                      default = None)
    parser.add_option("-c", "--class_index",
                      help="Index of class to visualize.",
                      default = 0, type = "int")


    options, args = parser.parse_args()


    infile = options.pickle_file
    if infile is None:
        print("Must specify input file (-p)\nExiting...")
        exit(-1)

    instances = options.instances
    if instances is not None:
        instances = np.array(instances.split(",")).astype("int")
    else:
        shap_values = pickle.load(open(infile, "rb"))
        instances = range(shap_values.shape[0])

    classIdx = options.class_index

    all_shap_values = [None for i in instances]
    all_data_values = [None for i in instances]
    for i, instanceIdx in enumerate(instances):
        all_shap_values[i], all_data_values[i], _, _ = loadPickle(infile, instanceIdx, classIdx)
    all_shap_values = np.array(all_shap_values)
    all_data_values = np.array(all_data_values)

    posTitle = "Toward class"
    negTitle = "Away from"
    bothTitle = "Combined"

    fig, axs = plt.subplots(3, 2)

    #########################
    # Version 1: Simple sum #
    #########################
    # Positive-only
    pos = np.copy(all_shap_values)
    pos[pos < 0] = 0
    # Negative-only
    neg = np.copy(all_shap_values)
    neg[neg > 0] = 0
    neg = np.abs(neg)
    # Sum all instances into single instance
    pos_sum = np.sum(pos, axis=0)
    neg_sum = np.sum(neg, axis=0)
    # Combine pos, neg into single instance
    output_A = np.add(pos_sum, (-1) * neg_sum)
    # Sum all channels into single channel
    output_A = np.sum(output_A, axis = -1)
    pos_sum = np.sum(pos_sum, axis=-1)
    neg_sum = np.sum(neg_sum, axis=-1)
    pos_max = np.max(pos_sum)
    neg_max = np.max(neg_sum)
    vmax = max(pos_max, neg_max)
    # Plot
    axs[0][0].set_title(posTitle)
    axs[0][0].imshow(pos_sum, cmap="Reds", vmax=vmax)
    axs[0][1].set_title(negTitle)
    axs[0][1].imshow(neg_sum, cmap="Blues", vmax=vmax)
    for i in range(len(axs[0])):
        axs[0][i].invert_yaxis()
        axs[0][i].axis('off')
    axs[0][0].set_ylabel("Sum")
    axs[0][0].text(40, 30, "sum")

    #######################
    # Version 2: Norm sum #
    #######################
    # Positive-only
    pos = np.copy(all_shap_values)
    pos[pos < 0] = 0
    # Negative-only
    neg = np.copy(all_shap_values)
    neg[neg > 0] = 0
    neg = np.abs(neg)
    # Normalize each instance (w.r.t. self)
    for i in range(len(instances)):
        x_min = pos[i].min(axis=(0,1,2), keepdims=True)
        x_max = pos[i].max(axis=(0, 1, 2), keepdims=True)
        pos[i] = (pos[i] - x_min)/(x_max-x_min)
        x_min = neg[i].min(axis=(0,1,2), keepdims=True)
        x_max = neg[i].max(axis=(0, 1, 2), keepdims=True)
        neg[i] = (neg[i] - x_min)/(x_max-x_min)
    # Sum all instances into single instance
    pos_sum = np.sum(pos, axis=0)
    neg_sum = np.sum(neg, axis=0)
    # Combine pos, neg into single instance
    output_A = np.add(pos_sum, (-1) * neg_sum)
    # Sum all channels into single channel
    output_A = np.sum(output_A, axis = -1)
    pos_sum = np.sum(pos_sum, axis=-1)
    neg_sum = np.sum(neg_sum, axis=-1)
    pos_max = np.max(pos_sum)
    neg_max = np.max(neg_sum)
    vmax = max(pos_max, neg_max)
    # Plot
    axs[1][0].set_title(posTitle)
    axs[1][0].imshow(pos_sum, cmap="Reds", vmax=vmax)
    axs[1][1].set_title(negTitle)
    axs[1][1].imshow(neg_sum, cmap="Blues", vmax=vmax)
    for i in range(len(axs[1])):
        axs[1][i].invert_yaxis()
        axs[1][i].axis('off')
    axs[1][0].set_ylabel("Normalized sum")
    axs[1][0].text(40, 30, "norm -> sum")

    #######################
    # Version 3: Only top #
    #######################
    # Positive-only
    pos = np.copy(all_shap_values)
    pos[pos < 0] = 0
    # Negative-only
    neg = np.copy(all_shap_values)
    neg[neg > 0] = 0
    neg = np.abs(neg)
    for i in range(len(instances)):
        # Normalize each instance (w.r.t. self)
        x_min = pos[i].min(axis=(0,1,2), keepdims=True)
        x_max = pos[i].max(axis=(0, 1, 2), keepdims=True)
        pos[i] = (pos[i] - x_min)/(x_max-x_min)
        x_min = neg[i].min(axis=(0,1,2), keepdims=True)
        x_max = neg[i].max(axis=(0, 1, 2), keepdims=True)
        neg[i] = (neg[i] - x_min)/(x_max-x_min)
        # Keep only top
        x_prctile = np.percentile(pos[i], 99, axis=(0, 1, 2))
        pos[i][pos[i] < x_prctile] = 0
        x_prctile = np.percentile(neg[i], 99, axis=(0, 1, 2))
        neg[i][neg[i] < x_prctile] = 0
    # Sum all instances into single instance
    pos_sum = np.sum(pos, axis=0)
    neg_sum = np.sum(neg, axis=0)
    # Combine pos, neg into single instance
    output_A = np.add(pos_sum, (-1) * neg_sum)
    # Sum all channels into single channel
    output_A = np.sum(output_A, axis = -1)
    pos_sum = np.sum(pos_sum, axis=-1)
    neg_sum = np.sum(neg_sum, axis=-1)
    pos_max = np.max(pos_sum)
    neg_max = np.max(neg_sum)
    vmax = max(pos_max, neg_max)
    # Plot
    axs[2][0].set_title(posTitle)
    axs[2][0].imshow(pos_sum, cmap="Reds", vmax=vmax)
    axs[2][1].set_title(negTitle)
    axs[2][1].imshow(neg_sum, cmap="Blues", vmax=vmax)
    for i in range(len(axs[2])):
        axs[2][i].invert_yaxis()
        axs[2][i].axis('off')
    axs[2][0].text(40, 30, "norm -> 99th prctile -> sum")

    plt.tight_layout()
    plt.show()

    exit(0)


    fig, ax = plt.subplots(1)
    ax.imshow(all_data_values[0][:,:,377], cmap = "Greys", alpha=0.25)
    ax.imshow(aggBands, cmap="Purples", alpha=0.75)
    ax.invert_yaxis()
    plt.show()



if __name__ == "__main__":
    main()





