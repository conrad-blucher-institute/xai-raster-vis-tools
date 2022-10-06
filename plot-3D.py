# Visualize 3D SHAP values
#
# PartitionShap, among other programs, may assign SHAP values
# to each (x, y, z) cell in a 3D model input.
# For example, an image classification model may have RGB inputs
# and we are interested in the SHAP contribution of superpixels
# within each color channel

import numpy as np
import pickle
import pyvista as pv
from optparse import OptionParser
from matplotlib.colors import ListedColormap
import shap

def loadPickle(pickleFile, instanceIdx, classIdx):
    shap_values = pickle.load(open(pickleFile, "rb"))

    return (shap_values.values[instanceIdx, :, :, :, classIdx], 
            shap_values.base_values, 
            shap_values.output_names)

def buildGrid(values, origin=(0, 0, 0), spacing=(10, 10, 10)):
    # Spatial reference
    grid = pv.UniformGrid()

    # Grid dimensions (shape + 1)
    grid.dimensions = np.array(values.shape) + 1

    # Spatial reference params
    grid.origin = origin
    grid.spacing = spacing

    # Grid data
    grid.cell_data["values"] = values.flatten(order="F")

    return grid

def printShapInfo(shap_values):
    labels = shap_values.output_names
    print(labels)



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
    options, args = parser.parse_args()

    infile = options.pickle_file
    instanceIdx = options.instance_index
    classIdx = options.class_index

    inNPZ = None  # Numpy archive
    values = None # SHAP values

    # Check: can read data?
    #try:
    shap_values, base_values, class_labels = loadPickle(infile, instanceIdx, classIdx)
    #except:
    #    print("Could not read {} as pickled SHAP values.".format(infile))
    #    exit(1)

    # Check: is data 3D?
    if (len(shap_values.shape) != 3):
        print("Only supports 3 dimensions. Detected shape of {}".format(shap_values.shape))
        exit(1)

    # Calc min, max
    minValue = np.min(shap_values)
    maxValue = np.max(shap_values)

    print("")
    print("SHAP 3D viewer")
    print("--------------")
    print("values file: {}".format(infile))
    print("      shape: {}".format(shap_values.shape))
    print("      range: ({:.4f}, {:.4f})".format(minValue, maxValue))
    print("Prediction class: {}".format(class_labels[classIdx]))
    print("           value: {}".format(base_values[classIdx]))
    print("")

    # Create grid
    grid = buildGrid(shap_values)
    tgrid = grid.threshold_percent([0.4, 0.6], invert = True)

    p = pv.Plotter()
    # Very faint grid mesh
    p.add_mesh(grid,
               style="wireframe",
               opacity=0.075,
               cmap="seismic",
               )

    #p.add_mesh_clip_plane(grid,
    #                      cmap="seismic",
    #                      assign_to_axis='z',
    #                      invert=True)


    from matplotlib.colors import LinearSegmentedColormap
    colors = []
    for l in np.linspace(1, 0, 100):
        colors.append((30./255, 136./255, 229./255,l))
    for l in np.linspace(0, 1, 100):
        colors.append((255./255, 13./255, 87./255,l))
    red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)


    blue_rgb = np.array([0.0, 0.0, 76.0/255])
    red_rgb = np.array([127.0/255, 0.0, 0.0])
    white_rgb = np.array([1.,1.,1.])

    colors = []
    for alpha in np.linspace(1, 0, 100):
        c = blue_rgb * alpha + (1 - alpha) * white_rgb
        colors.append(c)
    for alpha in np.linspace(0, 1, 100):
        c = red_rgb * alpha + (1 - alpha) * white_rgb
        colors.append(c)
    red_white_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

    p.add_mesh_threshold(grid,
                         invert=True,
                         pointa=(0.1, 0.9),
                         pointb=(0.45, 0.9),
                         title = "Lower threshold",
                         cmap=red_white_blue)

    p.add_mesh_threshold(grid,
                         pointa=(0.55, 0.9),
                         pointb=(0.9, 0.9),
                         invert=False,
                         title = "Higher threshold",
                         cmap=red_white_blue)

    p.show()

    return 0

if __name__ == "__main__":
    main()



