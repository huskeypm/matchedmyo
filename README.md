# MatchedMyo

Algorithm for the classification of subcellular features within isolated myocytes and tissue swatches. 

NOTE: Before running algorithm on user provided images, be sure to run through

<code>
python preprocessing.py -preprocess "IMGNAME" "FILTERTWOSARCSIZE"
</code>

Where IMGNAME is the path/name of your image and FILTERTWOSARCSIZE 
is the default two sarcomere size for the filters used. Default filter size is 25 pixels.

# Package Dependencies and Installation Pages
- OpenCV (cv2)
- imutils (https://github.com/jrosebr1/imutils)
- Numpy
- Matplotlib
- PyYAML
- Scipy
  - Scipy.fftpack
  - Scipy.signal
- Pandas
- Pygame (https://www.pygame.org/download.shtml)
- Python Image Library (PIL)
- SciKit Learn
  - sklearn.decomposition
- TensorFlow

NOTE: All package dependencies are handled by a full Anaconda install except for imutils and pygame.

# Upon Pulling a Clean Repo
Initialize the repo by running the following commands:

1. Run "python util.py -genAllMyo" to generate all necessary filters

2. Run "python preprocessing.py -preprocessAll" to process all of the included myocyte images

# Preprocesing User Supplied Images
To preprocess a directory containing user supplied images, run:

<code>
python preprocessing.py -preprocessDirectory PATH_TO_DIRECTORY
</code>

# MASTER SCRIPT 
## detect.py 
To run with yaml: (follow ex.yml for an example) 
python detect.py -updatedSimpleYaml ex.yml


## GPU-accelerated Detection
GPU-accelerated matched filter detection.

Requires tensorflow cuda (python).

GPU-acceleration implemented in twoDtense.py . This can be turned on and off with the "useGPU" flag in the parameter dictionary.

# Miscellaneous 

### validation test
<code>
python myocyteFigs.py -validate
</code>

### Generate Paper Figures
To preprocess images:
<code>
python preprocessing.py -preprocessAll
</code>
To generate the paper figures:
<code>
python myocyteFigs.py -allFigs 
</code>

### Minor validation test
<code>
python myocyteFigs.py -minorValidate
</code>
Meant to serve as a rapid validation test between commits

### ROC optimization
<code>
python myocyteFigs.py -full_ROC
</code>