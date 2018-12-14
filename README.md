# MatchedMyo

Algorithm for the classification of subcellular features within isolated myocytes and tissue swatches. 

NOTE: Before running algorithm on user provided images, be sure to run through

```
python preprocessing.py -preprocess "IMGNAME" "FILTERTWOSARCSIZE"
```

Where IMGNAME is the path/name of your image and FILTERTWOSARCSIZE 
is the default two sarcomere size for the filters used. Default filter size is 25 pixels.

# Package Dependencies and Installation Pages
- OpenCV (cv2)
  - opencv-python (3.4.1.15)
- imutils (https://github.com/jrosebr1/imutils)
  - imutils (0.4.6)
- Numpy
  - numpy (1.15.4)
- Matplotlib
  - matplotlib (2.2.3)
- PyYAML
  - PyYAML (3.12)
- Scipy
  - scipy (1.1.0)
- Pandas
  - pandas (0.23.4)
- Pygame (https://www.pygame.org/download.shtml)
  - pygame (1.9.3)
- Python Image Library (PIL)
  - Pillow (5.2.0)
- SciKit Learn
  - scikit-learn (0.19.1)
- Tifffile
  - tifffile (2018.10.18)

NOTE: All package dependencies are handled by a full Anaconda install except for imutils and pygame.
If using a linux machine, installation can be handled by running './installation.bash' from within the MatchedMyo repository.

# Upon Pulling a Clean Repo
Initialize the repo by running the following commands:

1. Run `python util.py -genAllMyo` to generate all necessary filters

2. Run `python preprocessing.py -preprocessAll` to process all of the included myocyte images

# Preprocesing User Supplied Images
To preprocess a directory containing user supplied images, run:

`python preprocessing.py -preprocessDirectory PATH_TO_DIRECTORY`

# MASTER SCRIPT 
## detect.py 
To run with yaml: (follow ex.yml for an example) 
`python detect.py -updatedSimpleYaml ex.yml`


## GPU-accelerated Detection
GPU-accelerated matched filter detection.

Requires tensorflow cuda (python).

GPU-acceleration implemented in twoDtense.py . This can be turned on and off with the "useGPU" flag in the parameter dictionary.

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
