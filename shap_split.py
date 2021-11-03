# Given a pickled SHAP explanation and a list of indices (that start with 0),
# extract only those instances into new SHAP explanation
# https://shap.readthedocs.io/en/latest/generated/shap.Explanation.html#shap.Explanation

import shap
import numpy as np
import pickle
from pathlib import Path
from optparse import OptionParser

def main():
    parser = OptionParser()
    parser.add_option("-p", "--pickle",
                      help="Pickled SHAP explanation.")
    parser.add_option("-i", "--indices",
                      help="Comma-delimited indices to extract, i.e. 4,10,2.")
    parser.add_option("-o", "--outfile",
                      help="Path to save extracted SHAP pickle file.")
    options, args = parser.parse_args()

    picklefile = options.pickle
    indices = options.indices
    outfile = options.outfile

    try:
        inExp = pickle.load(open(picklefile, "rb"))
    except:
        print("Expected picked SHAP explanation (-p).\nExiting...")
        exit(1)
    print("Input pickle file: {}".format(picklefile))
    print("    Shape: {}".format(inExp.shape))

    try:
        indices = np.array(indices.split(",")).astype("int")
    except:
        print("Expected comma-delimited list of indices to extract (-i).\nExiting...")
        exit(1)
    print("Indices to extract: {}".format(indices))
    outExp = inExp[indices]

    try:
        with open(outfile, "wb") as f:
            pickle.dump(outExp, f)
    except:
        print("Expected path to save extracted SHAP explanation (-o).\nExiting...")
        exit(1)
    print("Ouput pickle file: {}".format(outfile))
    print("    Shape: {}".format(outExp.shape))

if __name__ == "__main__":
    main()

