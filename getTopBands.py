# Calculate & plot the top N bands based on the SHAP values

import numpy as np
import pandas as pd
import pickle
from optparse import OptionParser
from matplotlib import colors
import matplotlib.pyplot as plt
import copy
import shap

colorRed = (250./255, 150./255, 150./255)
colorBlue = (150./255, 150./255, 250./255)

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
                      help="File where the ith line is a text desciption of the ith band. Starts with band 0.")
    parser.add_option(     "--no_plot",
                      help="Only calculate top bands, no plotting",
                      default=False, action="store_true")

    # SHAP values (optional, secondary)
    parser.add_option(      "--pickled_shap_2",
                      help="Path to pickled SHAP values (optional, comparison SHAP values).")
    parser.add_option(      "--class_index_2",
                      help = "Index of class to use.",
                      default=0, type="int")
    parser.add_option(      "--class_name_2",
                      help="Name of optional comparison SHAP values (see option `--pickled_shap_2`.",
                      default = "default-class-name-2")

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
        with open(bandDescsFile) as f:
            bandDescs = f.read().splitlines()
    noPlot = options.no_plot

    # SHAP values (optional, secondary)
    infile2 = options.pickled_shap_2
    classIdx2 = options.class_index_2
    className2 = options.class_name_2

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

    if infile2 is not None:
        print("  Using secondary SHAP values")
        print("  Class index: {}".format(classIdx2))
        print("        name: {}".format(className2))
        print("  SHAP pickle file: {}".format(infile2))

    if infile2 is None:
        fig = plt.figure(figsize=(12,10))
        gs = fig.add_gridspec(2,3)
    else:
        fig = plt.figure(figsize=(14,10))
        gs = fig.add_gridspec(4,3)


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


    ######################
    # Primary: top bands #
    ######################

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

    ##########################
    # Primary: make CSV file #
    ##########################

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
        #################
        # Primary: plot #
        #################

        ax0 = fig.add_subplot(gs[0:2, 0])
        ax1A = fig.add_subplot(gs[0, 1])
        ax1B = fig.add_subplot(gs[1, 1])
        ax2A = fig.add_subplot(gs[0, 2])
        ax2B = fig.add_subplot(gs[1, 2])

        maxCount = max(np.max(shapAbsTopCounts), np.max(shapAbsBottomCounts))

        # Plot count of band occurances in top N bands:   SHAP magnitude for top and bottom occurances
        mainColor = (105.0/255.0, 225.0/255.0, 105.0/255.0)
        ax0.bar(x=range(bands), height=shapAbsTopCounts, width=4, color=mainColor)
        ax0.set_xlim(-1, bands+1)
        ax0.set_ylim(0, maxCount)
        ax0.set_title(className)
        ax0.set_xlabel("Band")
        ax0.set_ylabel("Occurances in top {}".format(numBands))
        for g in groups:
            ax0.axvline(g, color="black", linestyle="dotted", lw=3)

        ax1A.bar(x=range(bands), height=shapAbsTopCounts, width=4, color=mainColor)
        ax1A.set_xlim(-1, bands+1)
        ax1A.set_ylim(0, maxCount)
        ax1A.set_title("Occurances in top/bottom {} bands".format(numBands))
        ax1A.set_xlabel(None)
        ax1A.set_ylabel("Top bands")
        ax1A.set_xticks([])
        for g in groups:
            ax1A.axvline(g, color="black", linestyle="dotted", lw=3)

        ax1B.bar(x=range(bands), height=shapAbsBottomCounts, width=4, color="tab:gray")
        ax1B.set_xlim(-1, bands+1)
        ax1B.set_ylim(0, maxCount)
        ax1B.set_xlabel("Band")
        ax1B.set_ylabel("Bottom bands")
        for g in groups:
            ax1B.axvline(g, color="black", linestyle="dotted", lw=3)
        ax1B.invert_yaxis()

        # Plot count of band occurances in top N bands:   POS and NEG SHAP values
        ax2A.bar(x=range(bands), height=posCounts, width=4, color=colorRed)
        ax2A.set_xlim(-1, bands+1)
        ax2A.set_ylim(0, maxCount)
        ax2A.set_title("Occurances in top {} bands".format(numBands))
        ax2A.set_xlabel(None)
        ax2A.set_ylabel("Top positive")
        ax2A.set_xticks([])
        for g in groups:
            ax2A.axvline(g, color="black", linestyle="dotted", lw=3)
        ax2B.bar(x=range(bands), height=negCounts, width=4, color=colorBlue)
        ax2B.set_xlim(-1, bands+1)
        ax2B.set_ylim(0, maxCount)
        ax2B.set_xlabel("Band")
        ax2B.set_ylabel("Top negative")
        for g in groups:
            ax2B.axvline(g, color="black", linestyle="dotted", lw=3)
        ax2B.invert_yaxis()


    #########################################
    # [Optional] Add secondary SHAP results #
    #########################################
    if infile2 is not None:

        ######################
        # Secondary: top bands #
        ######################

        # Extract only the shap values
        shap_values = pickle.load(open(infile2, "rb"))
        nInstances, rows, cols, bands, nClasses = shap_values.values.shape
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

        ##########################
        # Primary: make CSV file #
        ##########################

        shapOrderedBands = orderBandsByFreq(shapBands, bands)
        shapOrderedBandDescs = orderBandDescs(shapOrderedBands, bandDescs)

        shapAbsOrderedBands = orderBandsByFreq(shapAbsBands, bands)
        shapAbsOrderedBandDescs = orderBandDescs(shapAbsOrderedBands, bandDescs)

        dfDict = {'{}_shap_band'.format(className2) : shapOrderedBands,
              '{}_shap_desc'.format(className2) : shapOrderedBandDescs,
              '{}_shap_abs_band'.format(className2) : shapAbsOrderedBands,
              '{}_shap_abs_desc'.format(className2) : shapAbsOrderedBandDescs}
        df2 = pd.DataFrame(dfDict)

        # Combine primary and secondary dataframes
        df = pd.concat((df, df2), axis=1)

        if noPlot == False:
            #################
            # Primary: plot #
            #################

            ax02 = fig.add_subplot(gs[2:, 0])
            ax1A2 = fig.add_subplot(gs[2, 1])
            ax1B2 = fig.add_subplot(gs[3, 1])
            ax2A2 = fig.add_subplot(gs[2, 2])
            ax2B2 = fig.add_subplot(gs[3, 2])

            ax0.set_title("")
            ax0.set_ylabel("{}: occurances in top {}".format(className, numBands))

            # Plot count of band occurances in top N bands:   SHAP magnitude for top and bottom occurances
            ax02.bar(x=range(bands), height=shapAbsTopCounts, width=4, color=mainColor)
            ax02.set_xlim(-1, bands+1)
            ax02.set_ylim(0, maxCount)
            ax02.set_xlabel("Band")
            ax02.set_ylabel("{}: occurances in top {}".format(className2, numBands))
            for g in groups:
                ax02.axvline(g, color="black", linestyle="dotted", lw=3)
            ax02.invert_yaxis()

            ax1A2.bar(x=range(bands), height=shapAbsTopCounts, width=4, color=mainColor)
            ax1A2.set_xlim(-1, bands+1)
            ax1A2.set_ylim(0, maxCount)
            ax1A2.set_title("Occurances in top/bottom {} bands".format(numBands))
            ax1A2.set_xlabel(None)
            ax1A2.set_ylabel("Top bands")
            ax1A2.set_xticks([])
            for g in groups:
                ax1A2.axvline(g, color="black", linestyle="dotted", lw=3)

            ax1B2.bar(x=range(bands), height=shapAbsBottomCounts, width=4, color="tab:gray")
            ax1B2.set_xlim(-1, bands+1)
            ax1B2.set_ylim(0, maxCount)
            ax1B2.set_xlabel("Band")
            ax1B2.set_ylabel("Bottom bands")
            for g in groups:
                ax1B2.axvline(g, color="black", linestyle="dotted", lw=3)
            ax1B2.invert_yaxis()

            # Plot count of band occurances in top N bands:   POS and NEG SHAP values
            ax2A2.bar(x=range(bands), height=posCounts, width=4, color=colorRed)
            ax2A2.set_xlim(-1, bands+1)
            ax2A2.set_ylim(0, maxCount)
            ax2A2.set_title("Occurances in top {} bands".format(numBands))
            ax2A2.set_xlabel(None)
            ax2A2.set_ylabel("Top positive")
            ax2A2.set_xticks([])
            for g in groups:
                ax2A2.axvline(g, color="black", linestyle="dotted", lw=3)
            ax2B2.bar(x=range(bands), height=negCounts, width=4, color=colorBlue)
            ax2B2.set_xlim(-1, bands+1)
            ax2B2.set_ylim(0, maxCount)
            ax2B2.set_xlabel("Band")
            ax2B2.set_ylabel("Top negative")
            for g in groups:
                ax2B2.axvline(g, color="black", linestyle="dotted", lw=3)
            ax2B2.invert_yaxis()


            def adjustax(ax, a, b):
                pos = ax.get_position()
                pnts = pos.get_points()
                pnts[0][1] = pnts[0][1] + a
                pnts[1][1] = pnts[1][1] + b
                pos.set_points(pnts)
                ax.set_position(pos)


            plt.tight_layout()

            adjustax(ax0, -0.035, -0.035)
            adjustax(ax02, 0.035, 0.035)
            adjustax(ax1B, 0.07, 0.07)
            adjustax(ax2B, 0.07, 0.07)
            adjustax(ax1A2, -0.07, -0.07)
            adjustax(ax2A2, -0.07, -0.07)


    # Write CSV
    if outfile is not None:
        df.to_csv(outfile, index=False)

    # Save plot
    plt.savefig(plotfile)


if __name__ == "__main__":
    main()
