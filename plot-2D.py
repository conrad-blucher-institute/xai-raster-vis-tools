# Visualize 3D SHAP values as a set of 2D plots

import numpy as np
import pandas as pd
import pickle
from optparse import OptionParser
import matplotlib
import matplotlib.pyplot as plt
import shap
from shap.plots import colors
from scipy.ndimage.filters import gaussian_filter

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
                      help="CSV with a column named 'band' that has the name of each band. May have other optional columns.")
    parser.add_option(      "--marker",
                      help="Comma-delimited row,col,size for a plot marker")
    parser.add_option("-o", "--output_file",
                      help="Path to save visualization image.")
    parser.add_option(      "--output_single",
                      help="Instead of a single large plot, each channel saved as own file. Will use 'BIDX' in filename with channel index.",
                      action = "store_true", default=False)
    parser.add_option(      "--reverse_z_standard",
                      help="Requires the csv from '--band_names', and with columns 'mean', 'std'. For bands with non-null mean & std, will reverse Z standardization back to original values.",
                      action = "store_true", default=False)
    parser.add_option(      "--clabel_fmt",
                      help="Format string for formatting the contour labels. See matplotlib's 'clabel' documentation.",
                      default="%1.2f")

    options, args = parser.parse_args()

    infile = options.pickle_file
    outfile = options.output_file
    instanceIdx = options.instance_index
    classIdx = options.class_index

    output_single = options.output_single

    reverse_z_std = options.reverse_z_standard

    clabel_fmt = options.clabel_fmt

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

    dfBands = None
    bandNames = options.band_names
    if bandNames is not None:
        dfBands = pd.read_csv(bandNames)
        bandNames = dfBands["band"].values
    else:
        bandNames = ["band: {}".format(b) for b in bands]

    # If sorting, sort band names
    if nTop:
        bandNames = [bandNames[i] for i in sortIdx]

    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.50

    abs_vals = np.abs(shap_values.sum(-1))
    max_val = np.nanpercentile(abs_vals, 99.9)

    max_level = 5
    min_level = -5
    step_level = 0.5

    if output_single:
        for i, band in enumerate(bands):
            shap_b = shap_values[:,:,band]
            data_b = data_values[:,:,band]

            # If `reverse_z_std` option,
            # then use the provided mean and std to convert standardized
            # values to original data values
            if dfBands is not None and reverse_z_std:
                mean = dfBands["mean"].iloc[band]
                std = dfBands["std"].iloc[band]
                if np.isnan(mean) == False:
                    data_b = std * data_b + mean

            local_abs_vals = np.abs(shap_b[:,:])
            local_max_val = np.nanpercentile(local_abs_vals, 99.9)

            fig, ax = plt.subplots(figsize=(3,3))
            ax.set_title(bandNames[i])
            ax.imshow(shap_b, cmap=colors.red_transparent_blue, vmin=-local_max_val, vmax=local_max_val, alpha=0.5)
            data_b_ = gaussian_filter(data_b, sigma=0.9)

            contours = ax.contour(data_b_, colors="#383630", alpha=1, linewidths=1)
            clabels = ax.clabel(contours, zorder=200, inline=True, fontsize=16, inline_spacing=6,
                                levels = contours.levels[::2], fmt=clabel_fmt, colors="#030a75")
            for txt in clabels:
                txt.set_weight('bold')

            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_xticks([], minor=True)
            ax.set_yticks([])
            ax.set_yticks([], minor=True)

            if marker is not None:
                mcolor="#b0e88b"
                for j in range(5):
                    ax.scatter(marker[1], marker[0], marker=".", s=marker[2], c=mcolor, alpha=0.5, edgecolors="#40612b", zorder=100)

            if outfile is not None:
                outfile_ = outfile.replace("BIDX", str(i))
                plt.savefig(outfile_, bbox_inches='tight')
            else:
                plt.show()

            plt.close(fig)

    else:
        fig, ax = plt.subplots(len(bands), 5, squeeze=False, figsize=(12, 4 * len(bands)))
        for i, band in enumerate(bands):

            shap_b = shap_values[:,:,band]
            data_b = data_values[:,:,band]

            # If `reverse_z_std` option,
            # then use the provided mean and std to convert standardized
            # values to original data values
            if dfBands is not None and reverse_z_std:
                mean = dfBands["mean"].iloc[band]
                std = dfBands["std"].iloc[band]
                if np.isnan(mean) == False:
                    data_b = std * data_b + mean

            local_abs_vals = np.abs(shap_b[:,:])
            local_max_val = np.nanpercentile(local_abs_vals, 99.9)

            ax[i][0].set_title(bandNames[i])
            ax[i][0].imshow(data_b, cmap="gray")
            ax[i][0].invert_yaxis()
            ax[i][1].set_title("SHAP: scaled by all")
            ax[i][1].imshow(shap_b, cmap="seismic", vmin=-max_val, vmax=max_val)
            ax[i][1].invert_yaxis()
            ax[i][2].set_title("")
            ax[i][2].imshow(shap_b, cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
            contours = ax[i][2].contour(data_b, colors='black', )
            ax[i][2].invert_yaxis()
            ax[i][3].set_title("SHAP: scaled by self")
            ax[i][3].imshow(shap_b, cmap=colors.red_transparent_blue, vmin=-local_max_val, vmax=local_max_val)
            ax[i][3].invert_yaxis()
            ax[i][4].set_title("")
            ax[i][4].imshow(shap_b, cmap=colors.red_transparent_blue, vmin=-local_max_val, vmax=local_max_val)

            data_b_ = gaussian_filter(data_b, sigma=0.6)


            ax[i][4].invert_yaxis()

            for j in range(5):
                ax[i][j].set_xticks([])
                ax[i][j].set_xticks([], minor=True)
                ax[i][j].set_yticks([])
                ax[i][j].set_yticks([], minor=True)


            if marker is not None:
                mcolor="#b0e88b"
                for j in range(5):
                    ax[i][j].scatter(marker[1], marker[0], marker="X", s=marker[2], c=mcolor, alpha=0.8, edgecolors="#40612b", zorder=10)

            contours = ax.contour(data_b_, colors="#383630", alpha=1, linewidths=1)
            clabels = ax.clabel(contours, zorder=200, inline=True, fontsize=16, inline_spacing=6,
                                levels = contours.levels[::2], fmt=clabel_fmt, colors="#030a75")
            for txt in clabels:
                txt.set_weight('bold')

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')
        else:
            plt.show()


if __name__ == "__main__":
    main()
