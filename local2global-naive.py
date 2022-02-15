# Combine XAI ouputs
# Very naive technique

import numpy as np
import pickle
from optparse import OptionParser
import matplotlib
import matplotlib.pyplot as plt
import shap
from shap.plots import colors
import matplotlib.colors as mcolors

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
    parser.add_option("-o", "--outfile_plot",
                      help="Path to save aggregate spatial-wise plot.")

    options, args = parser.parse_args()

    outfile_explanation = options.outfile_explanation
    if outfile_explanation is None:
        print("No output file for aggregated explanation specified (-e). Will not save.")
    else:
        print("Will save aggregate explanation to {}".format(outfile_explanation))

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

    # Get mean of each channel (over all instances,
    # useful for plotting rough idea of underlying data)
    adv = np.array(all_data_values)
    madv = np.mean(adv, axis=0)

    posTitle = "Toward class"
    negTitle = "Away from"
    bothTitle = "Combined"


    def splitSum(values, normalize=False):
        instances = values.shape[0]
        values_ = np.copy(values)

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
        negMax = np.min(aggSpatial_neg)

        return aggChannel,     aggSpatial, aggMax, aggMin, \
               aggChannel_pos, aggSpatial_pos, posMax, \
               aggChannel_neg, aggSpatial_neg, negMax


    fig, axs = plt.subplots(1, 4)
    agg_explanations = [None]

    aggChannel, aggSpatial, aggMax, aggMin, \
        aggChannel_pos, aggSpatial_pos, posMax, \
        aggChannel_neg, aggSpatial_neg, negMax = splitSum(all_shap_values)
    agg_explanations[0] = aggChannel_pos + aggChannel_neg

    # Plot it
    bound=np.max(np.array([posMax, -negMax]))
    axs[1].imshow(aggSpatial, cmap="seismic",  vmin=-bound, vmax=bound)
    axs[2].imshow(aggSpatial_pos, cmap="seismic", vmin=-bound, vmax=bound)
    axs[3].imshow(aggSpatial_neg, cmap="seismic", vmin=-bound, vmax=bound)

    # Add background image
    if background_idx >= 0:
        axs[0].imshow(all_data_values[0][:,:,background_idx], cmap = "Greys", alpha=0.25)

    # Other plot details
    # Titles
    axs[1].set_title("SHAP")
    axs[2].set_title("Pos SHAP")
    axs[3].set_title("Neg SHAP")

    for c in range(len(axs)):
        # Spatial data -> invert y axis
        axs[c].invert_yaxis()
        # Remove ticks
        axs[c].axis('off')
    axs[0].text(0, -5, "(Note: Y-axis inverted)")

    # Write outputs
    # Save aggregate as SHAP explanation
    output_exp = agg_explanations[0]
    output_exp = np.expand_dims(output_exp, axis = 0)
    output_exp = np.expand_dims(output_exp, axis = -1)
    madv_exp = np.expand_dims(madv, axis = 0)
    print(madv_exp.shape)
    exp = shap.Explanation(output_exp, data = madv_exp)
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
