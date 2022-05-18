# Visualize 3D SHAP values as a set of 2D plots

import numpy as np
import pickle
from optparse import OptionParser
import matplotlib
import matplotlib.pyplot as plt
import shap
from shap.plots import colors

def loadPickle(pickleFile, instanceIdx, classIdx):
    shap_values = pickle.load(open(pickleFile, "rb"))
    return (shap_values[instanceIdx, :, :, :, classIdx].values,
            shap_values[instanceIdx, :, :, :, classIdx].data,
            shap_values[instanceIdx, :, :, :, :].base_values,
            shap_values.output_names)


def main():
    parser = OptionParser()
    parser.add_option("-p", "--pickle_file",
                      help="Path to pickled SHAP values.",
                      default = None)
    parser.add_option("-i", "--instance_index",
                      help="Index of instance to visualize.",
                      default = 0, type = "int")
    parser.add_option("-c", "--class_index",
                      help="Index of class to visualize.",
                      default = 0, type = "int")
    parser.add_option("-b", "--bands",
                      help="Comma-delimited list of bands to plot")
    parser.add_option(      "--top-bands",
                      help="Number of top bands to plot. Ignores '-b' option.",
                      type = "int")
    parser.add_option(      "--band_names",
                      help="Comma-delimited list of band names for plot titles")
    parser.add_option(      "--marker",
                      help="Comma-delimited row,col,size for a plot marker")
    parser.add_option("-o", "--output_file",
                      help="Path to save visualization image.")

    options, args = parser.parse_args()

    infile = options.pickle_file
    outfile = options.output_file
    instanceIdx = options.instance_index
    classIdx = options.class_index

    marker = options.marker
    if marker is not None:
        marker = np.array(marker.split(",")).astype(float)

    # Load data from pickle
    shap_values, data_values, base_values, class_labels = loadPickle(infile, instanceIdx, classIdx)
    if data_values is None:
        data_values = np.zeros_like(shap_values)

    # Determine which bands to plot
    bands = options.bands
    nTop = options.top_bands
    if bands is not None:
        # List provided by user
        bands = np.array(bands.split(",")).astype("int")
    elif nTop is not None:
        # Top N bands
        shap_maxes = shap_values.max(axis = (0,1))
        shap_maxes_abs = np.abs(shap_maxes)
        sortIdx = np.argsort(shap_maxes_abs)[::-1]
        bands = sortIdx[:nTop]
    else:
        # Default to all bands
        n = shap_values.shape[-1]
        nMax = 50
        if n > nMax:
            print("Too many bands! ({}). Will plot first {}.".format(n, nMax))
            n = nMax
        bands = np.array(range(n))

    bandNames = options.band_names
    if bandNames is not None:
        bandNames = bandNames.split(",")
    else:
        bandNames = ["band: {}".format(b) for b in bands]

    # If sorting, sort band names
    if nTop:
        bandNames = [bandNames[i] for i in sortIdx]

    fig, ax = plt.subplots(len(bands), 5, squeeze=False, figsize=(12, 4 * len(bands)))

    abs_vals = np.abs(shap_values.sum(-1))
    max_val = np.nanpercentile(abs_vals, 99.9)


    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.50

    for i, band in enumerate(bands):

        shap_b = shap_values[:,:,band]
        data_b = data_values[:,:,band]

        local_abs_vals = np.abs(shap_b[:,:])
        local_max_val = np.nanpercentile(local_abs_vals, 99.9)

        ax[i][0].set_title(bandNames[i])
        ax[i][0].imshow(data_b, cmap="gray")
        ax[i][0].invert_yaxis()
        ax[i][1].set_title("SHAP: scaled by all")
        #ax[i][1].imshow(data_b, cmap="gray", alpha=0.25)
        ax[i][1].imshow(shap_b, cmap="seismic", vmin=-max_val, vmax=max_val)
        ax[i][1].invert_yaxis()
        ax[i][2].set_title("")
        ax[i][2].imshow(shap_b, cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
        contours = ax[i][2].contour(data_b, colors='black', )
        #ax[i][2].clabel(contours, inline=True, fontsize=8)
        ax[i][2].invert_yaxis()
        ax[i][3].set_title("SHAP: scaled by self")
        #ax[i][3].imshow(data_b, cmap="gray", alpha=0.25)
        ax[i][3].imshow(shap_b, cmap=colors.red_transparent_blue, vmin=-local_max_val, vmax=local_max_val)
        ax[i][3].invert_yaxis()
        ax[i][4].set_title("")
        ax[i][4].imshow(shap_b, cmap=colors.red_transparent_blue, vmin=-local_max_val, vmax=local_max_val)
        contours = ax[i][4].contour(data_b, colors='black', alpha=0.5)
        #ax[i][4].clabel(contours, inline=True, fontsize=8)
        ax[i][4].invert_yaxis()


        for j in range(5):
            ax[i][j].set_xticks([])
            ax[i][j].set_xticks([], minor=True)
            ax[i][j].set_yticks([])
            ax[i][j].set_yticks([], minor=True)


        if marker is not None:
            mcolor="#b0e88b"
            for j in range(5):
                ax[i][j].scatter(marker[1], marker[0], marker="*", s=marker[2], c=mcolor, alpha=0.9, edgecolors="#40612b", zorder=100)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    main()
