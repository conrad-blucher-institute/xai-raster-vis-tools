# Given a list of pickled SHAP explanations, combine into single pickled SHAP explanation.
# Each SHAP explanation of shape (cases, height, width, channels, classes)
# Here, cases are combined. So all other sizes must be the same
# https://shap.readthedocs.io/en/latest/generated/shap.Explanation.html#shap.Explanation

import shap
import numpy as np
import pickle
from pathlib import Path
from optparse import OptionParser

def merge(explanations):
    # explanations: list of SHAP output explanations, containing SHAP values, data values, etc

    n = len(explanations)

    values              = [None for i in range(n)]
    base_values         = [None for i in range(n)]
    data                = [None for i in range(n)]
    display_data        = [None for i in range(n)]
    instance_names      = [None for i in range(n)]
    feature_names       = [None for i in range(n)]
    output_names        = [None for i in range(n)]
    output_indexes      = [None for i in range(n)]
    lower_bounds        = [None for i in range(n)]
    upper_bounds        = [None for i in range(n)]
    #error_std          = [None for i in range(n)]
    main_effects        = [None for i in range(n)]
    hierarchical_values = [None for i in range(n)]
    clustering          = [None for i in range(n)]

    for i, exp in enumerate(explanations):
        values[i]              = exp.values
        base_values[i]         = exp.base_values
        data[i]                = exp.data
        display_data[i]        = exp.display_data
        instance_names[i]      = exp.instance_names
        feature_names[i]       = exp.feature_names
        output_names[i]        = exp.output_names
        output_indexes[i]      = exp.output_indexes
        lower_bounds[i]        = exp.lower_bounds
        upper_bounds[i]        = exp.upper_bounds
        #error_std[i] = exp.error_std         # <- AttributeError: 'Explanation' object has no attribute 'error_std'
        main_effects[i]        = exp.main_effects
        hierarchical_values[i] = exp.hierarchical_values
        clustering[i]          = exp.clustering

    values = np.concatenate(values)
    base_values         = np.concatenate(base_values)
    data                = np.concatenate(data)
    display_data        = np.concatenate(display_data)        if display_data[0]        is not None else None
    instance_names      = np.concatenate(instance_names)      if instance_names[0]      is not None else None
    feature_names       = np.concatenate(feature_names)       if feature_names[0]       is not None else None
    output_names        = np.concatenate(output_names)        if output_names[0]        is not None else None
    output_indexes      = np.concatenate(output_indexes)      if output_indexes[0]      is not None else None
    lower_bounds        = np.concatenate(lower_bounds)        if lower_bounds[0]        is not None else None
    upper_bounds        = np.concatenate(upper_bounds)        if upper_bounds[0]        is not None else None
    #error_std =
    main_effects        = np.concatenate(main_effects)        if main_effects[0]        is not None else None
    hierarchical_values = np.concatenate(hierarchical_values) if hierarchical_values[0] is not None else None
    clustering          = np.concatenate(clustering)          if clustering[0]          is not None else None

    merged = shap.Explanation(values,
                              base_values         = base_values,
                              data                = data,
                              display_data        = display_data,
                              instance_names      = instance_names,
                              feature_names       = feature_names,
                              output_names        = output_names,
                              output_indexes      = output_indexes,
                              lower_bounds        = lower_bounds,
                              upper_bounds        = upper_bounds,
                              main_effects        = main_effects,
                              hierarchical_values = hierarchical_values,
                              clustering          = clustering,
                              )
    return merged

def main():
    parser = OptionParser()
    parser.add_option("-p", "--pickles",
                      help="Comma-delimited list of pickled SHAP values to merge.")
    parser.add_option("-o", "--outfile",
                      help="Path to save merged SHAP pickle file.")
    parser.add_option("-q", "--quiet",
                      help="Suppress output.",
                      action="store_true", default=False)
    options, args = parser.parse_args()

    quiet = options.quiet
    pickles = options.pickles
    outfile = options.outfile

    try:
        pickles = pickles.split(",")
    except:
        print("Expected (-p) to be a comma-delimited list, like `shap1.pickle,shap2.pickle`.")
        exit(-1)

    try:
        outfile = Path(outfile)
        outfile.touch()
    except:
        print ("Expected (-o) to be a valid path to save merged SHAP pickle file.")
        exit(-2)

    # Load from pickles
    explanations = []
    for pfile in pickles:
        try:
            explanations.append(pickle.load(open(pfile, "rb")))
        except:
            print("Could not load file {}, Skipping...".format(pfile))

    # Merge
    merged = merge(explanations)

    if not quiet:
        print("Merged shape: {}".format(merged.shape))

    # Write
    with open(outfile, 'wb') as f:
        pickle.dump(merged, f)

if __name__ == "__main__":
    main()
