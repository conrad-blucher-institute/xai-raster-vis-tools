# Calculate & plot the top N bands based on the SHAP values

import numpy as np
import pandas as pd
import pickle
from optparse import OptionParser
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import copy
import shap

def main():
    parser = OptionParser()

    # SHAP values (primary)
    parser.add_option("-p", "--pickled_shap",
                      help="Path to pickled SHAP values.")
    parser.add_option("-c", "--class_index",
                      help="Index of class to use.",
                      default=0, type="int")
    parser.add_option("-n", "--class_name",
                      help="Name of class (for csv, plots).",
                      default="default-class-name")

    # Output files
    parser.add_option("-o", "--output_file",
                      help="Path to save top bands output.")
    parser.add_option("-i", "--image_file",
                      help="Path to save plot image.")

    # Parameters
    parser.add_option("-b", "--num_bands",
                      help="Number of top bands to select.",
                      default=10, type="int")
    parser.add_option(     "--groups",
                      help="Comma-delimited list of bands that separate groups (for plots).")
    parser.add_option(     "--band_descriptions",
                      help="CSV with a column named 'band' that has the name of each band.")
    parser.add_option(     "--no_plot",
                      help="Only calculate top bands, no plotting",
                      default=False, action="store_true")

    options, args = parser.parse_args()

    # SHAP values (primary)
    infile = options.pickled_shap
    if infile is None:
        print("Path to pickled SHAP values required ('-p')\nExiting...")
        exit(1)
    classIdx = options.class_index
    className = options.class_name

    # Output files
    outfile = options.output_file
    plotfile = options.image_file
    if plotfile is None:
        print("Path to save plot required ('-i')\nExiting...")
        exit(1)

    # Parameters
    numBands, groups, bandDescs = None, None, None
    numBands = options.num_bands
    groups = options.groups
    if groups is not None:
        groups = [int(g) for g in groups.split(",")]
    bandDescsFile = options.band_descriptions

    if bandDescsFile is not None:
        bandDescs = pd.read_csv(bandDescsFile)
        bandDescs = bandDescs["band"].values

    noPlot = options.no_plot

    print("")
    print("Get top N SHAP bands (N = {})".format(numBands))
    print("Class index: {}".format(classIdx))
    print("      name: {}".format(className))
    print("SHAP pickle file: {}".format(infile))
    print("CSV output file: {}".format(outfile))
    if noPlot == False:
        print("Plot output file: {}".format(plotfile))
    else:
        print("Selected '--no_plot' --> will not produce plots")
    print("")

    # Load pickled SHAP values
    shap_values = pickle.load(open(infile, "rb"))
    nInstances, rows, cols, bands, nClasses = shap_values.values.shape

    print("Number of instances: {}".format(nInstances))
    print("Image size: ({} rows, {} cols, {} bands)".format(rows, cols, bands))
    print("Number of prediction classes saved: {}".format(nClasses))
    print("")

    fig = plt.figure(figsize=(14,10))
    gs = fig.add_gridspec(2,3)

    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 14}

    matplotlib.rc('font', **font)

    #############
    # Functions #
    #############

    def sortShap(shap_values):
        shapSums = [sv[:, :, :, classIdx].sum(axis=(0, 1)) for sv in shap_values]
        shapSorts = np.array([np.sort(ss)[::-1] for ss in shapSums])
        shapBands = np.array([np.argsort(ss)[::-1] for ss in shapSums])
        return shapSorts, shapBands


    def countBands(shapSorts, shapBands, nBands, n):
        topSums = shapSorts[:,0:n]
        topBands = shapBands[:,0:n]
        topBandsFlat = topBands.flatten()
        topCounts = np.array([np.count_nonzero(topBandsFlat == b) for b in range(nBands)])

        bottomSums = shapSorts[:,(-1)*n:]
        bottomBands = shapBands[:,(-1)*n:]
        bottomBandsFlat = bottomBands.flatten()
        bottomCounts = np.array([np.count_nonzero(bottomBandsFlat == b) for b in range(nBands)])

        return topCounts, topSums, topBands, bottomCounts, bottomSums, bottomBands


    def filterBandsBySign(topBands, topSums, bottomBands, bottomSums, nInstances):
        filteredPos = []
        filteredNeg = []
        for i in range(nInstances):
            for bandidx, band in enumerate(list(topBands[i])):
                if (topSums[i][bandidx] > 0):
                    filteredPos.append(band)
            for bandidx, band in enumerate(list(bottomBands[i])):
                if (bottomSums[i][bandidx] < 0):
                    filteredNeg.append(band)

        return filteredPos, filteredNeg


    def orderBandsByFreq(shapBands, nBands):
        orderedBands = np.zeros(nBands).astype("int")
        for i in range(nBands):
            values, counts = np.unique(shapBands[:,i], return_counts=True)
            ind = np.argmax(counts)
            orderedBands[i] = values[ind]
        return orderedBands


    def orderBandDescs(shapOrderedBands, bandDescs):
        if bandDescs is None:
            bandDescs = ["band {}".format(i) for i in range(bands)]
        bandDescs = np.array(bandDescs)
        bandDescs = bandDescs[shapOrderedBands]
        return bandDescs


    # Extract only the shap values
    shap_values = list(shap_values.values)

    # SHAP absolute values
    shapAbsSorts, shapAbsBands = sortShap(np.absolute(shap_values))
    shapAbsTopCounts, shapAbsTopSums, shapAbsTopBands, \
        shapAbsBottomCounts, shapAbsBottomSums, shapAbsBottomBands = countBands(shapAbsSorts, shapAbsBands, bands, numBands)

    # SHAP values
    shapSorts, shapBands = sortShap(shap_values)
    shapTopCounts, shapTopSums, shapTopBands, \
        shapBottomCounts, shapBottomSums, shapBottomBands = countBands(shapSorts, shapBands, bands, numBands)
    shapTopBandsPos, shapTopBandsNeg = filterBandsBySign(shapTopBands, shapTopSums, shapBottomBands, shapBottomSums, nInstances)

    posCounts = np.array([np.count_nonzero(np.array(shapTopBandsPos) == b) for b in range(bands)])
    negCounts = np.array([np.count_nonzero(np.array(shapTopBandsNeg) == b) for b in range(bands)])

    # Make CSV file
    shapOrderedBands = orderBandsByFreq(shapBands, bands)
    shapOrderedBandDescs = orderBandDescs(shapOrderedBands, bandDescs)

    shapAbsOrderedBands = orderBandsByFreq(shapAbsBands, bands)
    shapAbsOrderedBandDescs = orderBandDescs(shapAbsOrderedBands, bandDescs)

    dfDict = {'{}_shap_band'.format(className) : shapOrderedBands,
              '{}_shap_desc'.format(className) : shapOrderedBandDescs,
              '{}_shap_abs_band'.format(className) : shapAbsOrderedBands,
              '{}_shap_abs_desc'.format(className) : shapAbsOrderedBandDescs}
    df = pd.DataFrame(dfDict)

    if noPlot == False:

        ax0 = fig.add_subplot(gs[0:2, 0])
        ax1A = fig.add_subplot(gs[0, 1])
        ax1B = fig.add_subplot(gs[1, 1])
        ax2A = fig.add_subplot(gs[0, 2])
        ax2B = fig.add_subplot(gs[1, 2])

        maxCount = max(np.max(shapAbsTopCounts), np.max(shapAbsBottomCounts))

        # Plot count of band occurances in top N bands:   SHAP magnitude for top and bottom occurances
        bandLabels = range(bands)
        cmap = matplotlib.cm.get_cmap('Set2').colors
        cmapLen = len(cmap)

        if len(groups) == 4:
            cmap = [
                   ("#6b7f9f"), ("#b19172"), ("#628254"), ("#ab7171"), ("#877d9f"), ("#a4c2f4"),
            ]

        a = 0
        groups_ = groups + [len(shapAbsTopCounts)]
        for i, g in enumerate(groups_):
            b = g
            rgba = cmap[i % cmapLen]
            ax0.bar(x=bandLabels[a:b], height = shapAbsTopCounts[a:b], width=4, color=rgba)
            ax1A.bar(x=bandLabels[a:b], height = shapAbsTopCounts[a:b], width=4, color=rgba)
            ax1B.bar(x=bandLabels[a:b], height = shapAbsBottomCounts[a:b], width=4, color=rgba)
            ax2A.bar(x=bandLabels[a:b], height = posCounts[a:b], width=4, color=rgba)
            ax2B.bar(x=bandLabels[a:b], height = negCounts[a:b], width=4, color=rgba)
            a = b

        ax0.set_xlim(-1, bands+1)
        ax0.set_ylim(0, maxCount)
        ax0.set_title(className)
        ax0.set_xlabel("Band")
        ax0.set_ylabel("Occurrences in top {}".format(numBands))
        for g in groups:
            ax0.axvline(g, color="black", linestyle="dotted", lw=3)

        ax1A.set_xlim(-1, bands+1)
        ax1A.set_ylim(0, maxCount)
        ax1A.set_title("Top {} bands".format(numBands))
        ax1A.set_xlabel(None)
        ax1A.set_ylabel("Occurrences")
        ax1A.set_xticks([])
        a = 0
        for i, g in enumerate(groups_):
            b = float(g)
            ax1A.axvline(g, color="black", linestyle="dotted", lw=3)
            ax1A.text((a + 0.5 * (b - a) - 10) / groups_[-1], -0.05, "G" + str(i + 1), transform=ax1A.transAxes)
            a = b

        ax1B.set_xlim(-1, bands+1)
        ax1B.set_ylim(0, maxCount)
        ax1B.set_ylabel("Occurrences")
        ax1B.set_title("Bottom {} bands".format(numBands), y=-0.075)
        ax1B.set_xticks([])
        for g in groups:
            ax1B.axvline(g, color="black", linestyle="dotted", lw=3)
        a = 0
        ax1B.invert_yaxis()
        for i, g in enumerate(groups_):
            b = float(g)
            ax1B.axvline(g, color="black", linestyle="dotted", lw=3)
            ax1B.text((a + 0.5 * (b - a) - 10) / groups[-1], 1.01, "G" + str(i + 1), transform=ax1B.transAxes)
            a = b

        ax2A.set_xlim(-1, bands+1)
        ax2A.set_ylim(0, maxCount)
        ax2A.set_title("Top positive {} bands".format(numBands))
        ax2A.set_xlabel(None)
        ax2A.set_ylabel("Occurrences")
        ax2A.set_xticks([])
        for g in groups:
            ax2A.axvline(g, color="black", linestyle="dotted", lw=3)
        a = 0
        for i, g in enumerate(groups_):
            b = float(g)
            ax2A.axvline(g, color="black", linestyle="dotted", lw=3)
            ax2A.text((a + 0.5 * (b - a) - 10) / groups_[-1], -0.05, "G" + str(i + 1), transform=ax2A.transAxes)
            a = b

        ax2B.set_xlim(-1, bands+1)
        ax2B.set_ylim(0, maxCount)
        ax2B.set_title("Top negative {} bands".format(numBands), y=-0.075)
        ax2B.set_ylabel("Occurrences")
        ax2B.set_xticks([])
        for g in groups:
            ax2B.axvline(g, color="black", linestyle="dotted", lw=3)
        ax2B.invert_yaxis()
        a = 0
        for i, g in enumerate(groups_):
            b = float(g)
            ax2B.axvline(g, color="black", linestyle="dotted", lw=3)
            ax2B.text((a + 0.5 * (b - a) - 10) / groups[-1], 1.01, "G" + str(i + 1), transform=ax2B.transAxes)
            a = b

    # Write CSV
    if outfile is not None:
        df.to_csv(outfile, index=False)

    # Save plot
    plt.tight_layout()
    plt.savefig(plotfile)


if __name__ == "__main__":
    main()
