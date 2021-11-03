# Combine XAI ouputs
# Very naive technique

import numpy as np
import pickle
from optparse import OptionParser
import matplotlib
import matplotlib.pyplot as plt
import shap
from shap.plots import colors

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
                      default=0, type="int")
    parser.add_option("-b", "--background_index",
                      help="Index of data value to use as plot background. By default, nothing.",
                      default=-1, type="int")
    parser.add_option("-e", "--outfile_explanation",
                      help="Path to save aggregate explanation.")
    parser.add_option(      "--explanation_type",
                      help="Type of explanation via numeric code: 0 -> sum, 1 -> normalized sum.",
                      default=0, type="int")
    parser.add_option("-o", "--outfile_plot",
                      help="Path to save aggregate spatial-wise plot.")

    options, args = parser.parse_args()

    explanation_types = ["sum", "normalized sum"]

    outfile_explanation = options.outfile_explanation
    explanation_type = options.explanation_type
    if outfile_explanation is None:
        print("No output file for aggregated explanation specified (-e). Will not save.")
    else:
        print("Will save aggregate explanation to {}".format(outfile_explanation))
        print(" Of type ({} -> {})".format(explanation_type, explanation_types[explanation_type]))
        print(" Available type options (-t):")
        for i in range(len(explanation_types)):
            print("    {} -> {}".format(i, explanation_types[i]))

    outfile_plot = options.outfile_plot
    if outfile_plot is None:
        print("No output file for plot specified (-o). Will display instead.")
    else:
        print("Will save plot to {}".format(outfile_plot))

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

    background_idx = options.background_index

    all_shap_values = [None for i in instances]
    all_data_values = [None for i in instances]

    shap_explanation = pickle.load(open(infile, "rb"))
    shap_explanation = shap_explanation[instances]
    all_shap_values = shap_explanation.values[:,:,:,:,0]
    all_data_values = shap_explanation.data

    posTitle = "Toward class"
    negTitle = "Away from"
    bothTitle = "Combined"


    def splitSum(values, normalize=False):
        instances = values.shape[0]
        values_ = np.copy(values)

        if normalize:
            # Normalize each instance (w.r.t. self)
            for i in range(instances):
                x_min = values_[i].min(axis=(0,1,2), keepdims=True)
                x_max = values_[i].max(axis=(0,1,2), keepdims=True)
                values_[i] = (values_[i] - x_min) / (x_max - x_min)

        # Sum all instances (within each channel)
        aggChannel = np.sum(values_, axis=0)
        # Sum all channels
        aggSpatial = np.sum(aggChannel, axis=-1)

        aggMax = np.max(aggSpatial)
        aggMin = np.min(aggSpatial)

        # Positive values
        pos = np.copy(values)
        pos[pos < 0] = 0 # Set negatives to 0

        # Negative values
        neg = np.copy(values)
        neg[neg > 0] = 0 # Set positives to 0
        neg = np.abs(neg) # Remove neg sign

        if normalize:
            for i in range(instances):
                x_min = pos[i].min(axis=(0,1,2), keepdims=True)
                x_max = pos[i].max(axis=(0,1,2), keepdims=True)
                pos[i] = (pos[i] - x_min) / (x_max - x_min)
                x_min = neg[i].min(axis=(0,1,2), keepdims=True)
                x_max = neg[i].max(axis=(0,1,2), keepdims=True)
                neg[i] = (neg[i] - x_min) / (x_max - x_min)

        # Sum all positives
        aggChannel_pos = np.sum(pos, axis=0)
        # Sum all channels
        aggSpatial_pos = np.sum(aggChannel_pos, axis=-1)
        # Calc max
        posMax = np.max(aggSpatial_pos)

        # Sum all negatives
        aggChannel_neg = np.sum(neg, axis=0)
        # Sum all channels
        aggSpatial_neg = np.sum(aggChannel_neg, axis=-1)
        # Calc max
        negMax = np.max(aggSpatial_neg)

        return aggChannel,     aggSpatial, aggMax, aggMin, \
               aggChannel_pos, aggSpatial_pos, posMax, \
               aggChannel_neg, aggSpatial_neg, negMax


    fig, axs = plt.subplots(2, 4)

    agg_explanations = [None, None]

    # Version 1: simple sum
    aggChannel, aggSpatial, aggMax, aggMin, \
        aggChannel_pos, aggSpatial_pos, posMax, \
        aggChannel_neg, aggSpatial_neg, negMax = splitSum(all_shap_values)
    #agg_explanations[0] = np.copy(aggChannel)
    agg_explanations[0] = aggChannel_pos + (-1)* aggChannel_neg
    # Plot it
    vmax = max(posMax, negMax)
    row = 0
    axs[row][0].set_title("Simple sum")
    vmax_ = np.max(np.abs(aggSpatial))
    axs[row][1].imshow(aggSpatial, cmap=colors.red_transparent_blue)
    axs[row][2].imshow(aggSpatial_pos, cmap=colors.red_transparent_blue, vmin=-vmax, vmax=vmax)
    axs[row][3].imshow(-1*aggSpatial_neg, cmap=colors.red_transparent_blue, vmin=-vmax, vmax=vmax)

    # Version 2: normalized sum
    aggChannel, aggSpatial, aggMax, aggMin, \
        aggChannel_pos, aggSpatial_pos, posMax, \
        aggChannel_neg, aggSpatial_neg, negMax = splitSum(all_shap_values, normalize=True)
    #agg_explanations[1] = np.copy(aggChannel)
    agg_explanations[1] = aggChannel_pos + (-1)* aggChannel_neg
    # Plot it
    vmax = max(posMax, negMax)
    row = 1
    vmax_ = np.max(np.abs(aggSpatial))
    axs[row][0].set_title("Norm sum")
    axs[row][1].imshow(aggSpatial, cmap=colors.red_transparent_blue)
    axs[row][2].imshow(aggSpatial_pos, cmap=colors.red_transparent_blue, vmin=-vmax, vmax=vmax)
    axs[row][3].imshow(-1*aggSpatial_neg, cmap=colors.red_transparent_blue, vmin=-vmax, vmax=vmax)

    # Add background image
    if background_idx >= 0:
        for r in range(len(axs)):
            axs[r][0].imshow(all_data_values[0][:,:,background_idx], cmap = "Greys", alpha=0.25)

    # Other plot details
    for r in range(len(axs)):
        # Titles
        axs[r][1].set_title("SHAP")
        axs[r][2].set_title("Pos SHAP")
        axs[r][3].set_title("Neg SHAP")

        for c in range(len(axs[0])):
            # Spatial data -> invert y axis
            axs[r][c].invert_yaxis()
            # Remove ticks
            axs[r][c].axis('off')
    axs[-1][0].text(0, -5, "(Note: Y-axis inverted)")

    # Write outputs
    # Save aggregate as SHAP explanation
    output_exp = agg_explanations[explanation_type]
    output_exp = np.expand_dims(output_exp, axis = 0)
    output_exp = np.expand_dims(output_exp, axis = -1)
    exp = shap.Explanation(output_exp)
    if outfile_explanation is not None:
        with open(outfile_explanation, 'wb') as f:
            pickle.dump(exp, f)

    # Spatial aggregate plot
    plt.tight_layout()
    if outfile_plot is not None:
        plt.savefig(outfile_plot)
    else:
        plt.show()


if __name__ == "__main__":
    main()
