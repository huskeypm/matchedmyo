# MatchedMyo

Algorithm for the classification of subcellular features within isolated myocytes and tissue swatches. 

# Package Dependencies and Installation Pages
* OpenCV (cv2). Installed package via pip -> opencv-python (3.4.1.15)
* imutils (https://github.com/jrosebr1/imutils). Installed package via pip -> imutils (0.4.6)
* Numpy. Installed package via pip -> numpy (1.15.4)
* Matplotlib. Installed package via pip -> matplotlib (2.2.3)
* Tkinter. Default Matplotlib Pyplot GUI backend. python-tk (8.6)
* PyYAML. Installed package via pip -> PyYAML (3.12)
* Scipy. Installed package via pip -> scipy (1.1.0)
* Pandas. Installed package via pip -> pandas (0.23.4)
* Pygame (https://www.pygame.org/download.shtml). Installed package via pip -> pygame (1.9.3)
* Python Image Library (PIL). Installed package via pip -> Pillow (5.2.0)
* SciKit Learn. Installed package via pip -> scikit-learn (0.19.1)
* Tifffile. Installed package via pip -> tifffile (2018.10.18)

NOTE: All package dependencies are handled by a full Anaconda install except for imutils and pygame.
If using a linux machine, installation of package dependencies can be handled by running './installation.bash' from within the MatchedMyo repository.

# Upon Pulling a Clean Repo
Initialize the repo by running the following commands:

1. Run `python util.py -genAllMyo` to generate all necessary filters

2. Run `python preprocessing.py -preprocessAll` to process all of the included myocyte images

# Preprocesing User Supplied Images

Before running algorithm on user provided images, be sure to run the included preprocessing routines of the images. To do so, run:

`python preprocessing.py -preprocess <IMGNAME> <FILTERTWOSARCSIZE>`

Where IMGNAME is the path/name of your image and FILTERTWOSARCSIZE the two sarcomere size for the filters used in pixels. Default filter size that the algorithm defaults to if this argument is not specified is 25 pixels. This is for preprocessing of single images.

To preprocess a directory containing user supplied images, run:

`python preprocessing.py -preprocessDirectory <PATH_TO_DIRECTORY>`

# MASTER SCRIPT - MatchedMyo.py

`matchedmyo.py` contains all of the routines needed to analyze user-supplied images with command line functionality. All arguments for analysis routines have default values that can be changed
via the specifed YAML file when calling the `matchedmyo.py` script as such:

`python matchedmyo.py run -yamlFile <YAML_NAME>`

Where -yamlFile indicates the YAML_NAME argument where parameters can be changed from their default.

To run, a YAML file is required (follow YAML_files/template.yml for an example). By default, all parameters used for analysis within the detect.py script are the same as the parameters used within the paper. The YAML (.yml) file contains the functionality for changing parameters the user would like to change from the default settings in the analysis they are running. 


# Miscellaneous 

### validation test
To validate the 2D usage of the software:

`python myocyteFigs.py -validate`

To validate the 3D usage of the software:

`python myocyteFigs.py -validate3D`

To validate the code fully:

`python myocyteFigs.py -fullValidation`

### Generate Paper Figures
To preprocess images:

`python preprocessing.py -preprocessAll`

To generate the paper figures:

`python myocyteFigs.py -allFigs`

### ROC optimization

`python myocyteFigs.py -full_ROC`
