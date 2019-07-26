# Introduction
Software for the classification of subcellular features within isolated myocytes and tissue tile scans. The predominant focus of this project is for the classification of transverse tubule structures.

# Index

- [About](#about)
- [Usage](#usage)
  - [Installation](#installation)
    - [Unix-Based Work Stations](#unix-based-work-stations)
    - [Windows Work Stations](#windows-work-stations)
    - [Virtual Machine Installation](#virtual-machine-installation)
    - [Python Package Dependencies](#python-package-dependencies)
  - [Regeneration of Paper Figures](#regeneration-of-published-figures)
  - [Command Line Usage for TT, LT, and TA Filters ](#command-line-usage-for-tt-lt-and-ta-filters)
  - [Command Line Usage for Arbitrary Filters](#command-line-usage-for-arbitrary-filters)
  - [YAML Parameters](#yaml-parameters)
    - [User Parameters](#user-parameters)
    - [Developer Parameters](#developer-parameters)
  - [Fiji](#fiji)
  - [Singularity](#singularity)
- [Development](#development)
  - [Validation Tests](#validation-tests)
- [Community](#community)
  - [Contribution](#contribution)
- [Credit/Acknowledgment](#creditacknowledgment)
- [License](#license)

# About
MatchedMyo is a matched-filter-based software package for subcellular T-system characterization in isolated cardiomyocytes and millimeter-scale myocardial sections.

# Usage
The software is traditionally ran via the command line, described [here](#command-line-usage-for-tt-lt-and-ta-filters).
However, a Fiji script has been developed to aid in the preprocessing of images and running of the software. Installation and usage of the Fiji script is described in [Fiji](#Fiji).

### Installation
You can find the OS-specific installation instructions below.

##### Unix-Based Work Stations
Installation for Mac and Linux is identical. A basic knowledge of how to navigate using terminal is required.
1. *Mercurial installation.* Ensure that mercurial is installed on your work station. If mercurial is not installed, please click the link [here](https://confluence.atlassian.com/get-started-with-bitbucket/install-and-set-up-mercurial-860009660.html) for detailed installation instructions.
2.  *Navigate to working directory/folder.* Open a terminal on your work station and navigate (using the `cd ` command in terminal) to the directory/folder where you would like for the MatchedMyo software to be located. 
3. *Clone the software from bitbucket to your work station.* Enter the command, `hg clone https://bitbucket.org/pkhlab/matchedmyo`. At this point, the algorithm and some sample images will begin to download. 
4. *Navigate to the new “matchedmyo” directory/folder.* Now, navigate to the new directory/folder created by the previous step. 
5. *Download Python libraries.* Choose to either automatically install python packages using PIP (preferred) or manually install the libraries. To automatically install the python libraries, type `./installation.bash` into the terminal and press “Enter”. The libraries will automatically begin installing via PIP. If manual installation is wished, refer to the list of python libraries and versions in Sect. S.1.6. 
6. *Generate filters.* Generate the filters needed for classification. To do this, execute the following command in terminal, `python util.py -genAllMyo`. This generates all filters necessary for running the software and stores them in the `myoimages` folder of the `matchedmyo` folder. 
7. *Verify installation.* The final step is to verify that the installation has completed without error. To do so, execute: `python matchedmyo.py fullValidation` The printing of “PASS” on the screen indicates that the MatchedMyo software has downloaded and installed correctly. If a `Runtime Error` is printed on the screen, the installation was not successful. This is likely due to library version inconsistencies. Run through the library installation instructions again and retry the validation. If this does not work, please email Dylan Colli at dfco222@g.uky.edu.

##### Windows Work Stations
1. *Install the Anaconda Distribution for Windows.* The recommended way to install and use the MatchedMyo software suite is through installation of the Python 2.7 Anaconda Distribution. For the download link and installation instructions, visit https://www.anaconda.com/download/#windows . 
2. *Open Anaconda Prompt.* In the Windows search bar, search for and launch the “Anaconda Prompt.” 
3. *Execute installation commands for Python libraries.* In the Anaconda Prompt, execute the following commands: 
```
$ pip install opencv-python==3.4.1.15 
$ pip install imutils==0.4.6 
$ pip install pygame==1.9.3 
$ pip install tifffile==2018.10.18 
```
4. *Install Mercurial for Windows.* In the Anaconda Prompt, execute the following command: `conda install -c conda-forge mercurial`. Confirm the packages to be installed by typing `y` and pressing "Enter." 
5. *Navigate to folder that will house algorithm.* Change your working folder in the Anaconda Prompt by using the `cd` command. It is recommended that you change your working folder to your desktop or something equally as accessible. An example of this command execution is as follows: `cd C:\Users\Dylan\Desktop`. However, individual paths to the desktop will vary based on username. 
6. *Clone the software from bitbucket to your work station.* To download the repository from bitbucket, execute the following command in the Anaconda Prompt: `hg clone https://bitbucket.org/pkhlab/matchedmyo`. 
7. *Change your working folder to the MatchedMyo folder.* To change your working folder, execute, `cd matchedmyo`. 
8. *Generate filters.* Generate the filters needed for classification. To do this, execute the following command: `python util.py -genAllMyo`. 
9. *Verify installation.* The final step is to verify that the installation has completed without error. To do so, execute: `python matchedmyo.py fullValidation` The printing of “PASS” on the screen indicates that the MatchedMyo software has downloaded and installed correctly. If a `Runtime Error` is printed on the screen, the installation was not successful. This is likely due to library version inconsistencies. Run through the library installation instructions again and retry the validation. If this does not work, please email Dylan Colli at dfco222@g.uky.edu.

##### Virtual Machine Installation
We provide a virtual machine for the usage of MatchedMyo. However, due to performance and data transfer overhead, it is heavily recommended that you install the software via the instructions above (and feel free to send questions to Dylan Colli at dfco222@g.uky.edu if you have them!). The virtual machine implementation can be found at https://drive.google.com/open?id=1KZxfzhJegcdxk0uaOViC1uiunWX370Sq. To execute the program within the virtual machine,the user must launch a terminal, navigate to “∼/scratch/matchedmyo/” and run the program as described herein.

##### Python Package Dependencies
The following is a list of the python package dependencies for the MatchedMyo software.

- OpenCV (cv2). Installed package via pip -> opencv-python (3.4.1.15)
- imutils (https://github.com/jrosebr1/imutils). Installed package via pip -> imutils (0.4.6)
- Numpy. Installed package via pip -> numpy (1.15.4)
- Matplotlib. Installed package via pip -> matplotlib (2.2.3)
- Tkinter. Default Matplotlib Pyplot GUI backend. python-tk (8.6)
- PyYAML. Installed package via pip -> PyYAML (3.12)
- Scipy. Installed package via pip -> scipy (1.1.0)
- Pandas. Installed package via pip -> pandas (0.23.4)
- Pygame (https://www.pygame.org/download.shtml). Installed package via pip -> pygame (1.9.3)
- Python Image Library (PIL). Installed package via pip -> Pillow (5.2.0)
- SciKit Learn. Installed package via pip -> scikit-learn (0.19.1)
- Tifffile. Installed package via pip -> tifffile (2018.10.18)

### Regeneration of Published Figures

To regenerate the figures found in the publication, run `python matchedmyo.py run --yamlFile YAML_files/<FILENAME>` where `<FILENAME>` is the name of the appropraite YAML file, e.g. `MI_M.yml`.

### Command Line Usage for TT, LT, and TA Filters 
MatchedMyo is ran via the command line. To do this, follow these steps: 
1. *Open a terminal.* MatchedMyo is operated through the command line interface. To access the command line interface on 1) Linux distributions, open the Dash and type “Terminal” into the search bar, 2) Mac, open Spotlight and search for “Terminal”, and 3) on Windows, launch the Anaconda Prompt. 
2. *Navigate to the directory/folder containing the MatchedMyo software.* Next, we must navigate to the directory or folder containing the MatchedMyo software. Change directories in terminal or the Anaconda Prompt using the `cd` command. 
3. *Create/edit a YAML (.yml) file.* Non-default parameters are specified through YAML files when running MatchedMyo. One must either create a fresh YAML file or edit the `template.yml` file included in the `YAML_files` folder of the MatchedMyo software. Only one parameter, the image name, is required. To specify this, include `imageName: <NAME_OF_IMAGE>` in the YAML file, where `<NAME_OF_IMAGE>` is the path to, and the name of, your image. To specify other parameters, include `<PARAMETER_NAME>: <PARAMETER>` in the YAML file, where `<PARAMETER_NAME>` is the name of the parameter and `<PARAMETER>` is the value of the parameter you would like to specify. For a full list of parameter options and acceptable values, see [YAML parameters](#yaml-parameters). 
4. *Run MatchedMyo.* This is accomplished by executing the following command in terminal, ` python matchedmyo.py run --yamlFile <YAML_FILE>`. Where `<YAML_FILE>` is the name of the YAML file generated in the previous step. 
5. *View outputs.* If `fileRoot` is specified in the YAML file, the classified images will be output to that location. Additionally, the analysis of TT, LT, and TA content will be stored in a comma-separated values (CSV) file. The default location for this CSV file is under the `results` folder in the MatchedMyo software folder, but a new location can be specified using the YAML file. See Sect. S.1.4  for full instructions on how to specify parameters via the YAML file.

### Command Line Usage for Arbitrary Filters
MatchedMyo is not limited by the filters with which you can use for morphological classification. It is very easy to adapt the software to use user-generated filters for their own morphological classification. To do so, generate the filter you would like to use using either an image manipulation software like Fiji or using a scripting langauge like Python. Make sure that the filter is saved as either a PNG or TIFF! Afterwards, generate the YAML file as in the case with the TT, LT, and TA filters, but add the `classificationType: arbitrary` line to the YAML file and specify the parameter dictionary as described in [developer parameters](#developer-parameters).

### YAML Parameters
The following is a list of all tunable parameters/options available through the YAML file functionality.

##### User Parameters
The following are parameters used to tune what classification is performed and what is output by the software.

- `imageName` - Path to, and name of, the image that is to be classified by the MatchedMyo software.
  - *Acceptable Inputs* - Any string.
  - *Example Input* - `imageName: /home/user1/Pictures/image1.png`

- `maskName` - Path to, and name of, the image that acts as a mask for analysis.
  - *Acceptable Inputs* - Any string.
  - *Example Input* - `imageName: /home/user1/Pictures/image1_mask.png`

- `outputParams` - This is how the output file name, file type, dots per inch (DPI, resolution) of the output images, and the classification results CSV file is specified. If the file name is `None`, like the default input, then the output files are not written. The default storage location for the classification results CSV file is located in the `results` folder of the MatchedMyo software. This CSV file contains information such as date and time of analysis, name of the image, TT, LT, and TA content, and output image location and name.
  - *Acceptable Inputs*
  ```
  outputParams:
    fileRoot: <a string>
    fileType: <png, tif, or pdf>
    dpi: <an integer>
    saveHitsArray: <True or False>
    csvFile: <a string>
  ```
  - *Default Input*
  ```
  outputParams:
    fileRoot: None
    fileType: png
    dpi: 300
    saveHitsArray: False
    csvFile: ./results/calssification_results.csv
  ```

- `preprocess` - This turns preprocessing on or off.
  - *Acceptable Inputs* - `preprocess: <True or False>`
  - *Default Input* - `preprocess: True`

- `filterTypes` - 
  - *Acceptable Inputs* 
  ```
  filterTypes:
    TT: <True or False>
    LT: <True or False>
    TA: <True or False>
  ```
  - *Default Input*
  ```
  filterTypes:
    TT: True
    LT: True
    TA: True
  ```

- `scopeResolutions` - This is the resolution of the confocal microscope used to collect images in pixels/voxels per micron. Note that this uses the convention of `x` being the first axis, corresponding to the rows of the image. Thus, `y` corresponds to the columns in the image
  - *Acceptable Inputs*
  ```
  scopeResolutions:
    x: <an integer or float>
    y: <an integer or float>
    z: <an integer or float>
  ```
  - *Default Input* - None
  - *Example Input*
  ```
  scopeResolutions:
    x: 5.03
    y: 5.03
  ```

- `iters` - This parameter controls the rotations at which the filters will be analyzed.
  - *Acceptable Inputs* - `iters: <a list of integers>`
  - *Default Input* - `iters: [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]`

- `returnAngles` - This option turns on/off the analysis and output of transverse tubule striation angle.
  - *Acceptable Inputs* - `returnAngles: <True or False>`
  - *Default Input* - `returnAngles: False`

- `returnPastedFilter` - This option turns on/off the superimposing of filter-sized unit cells on the hits of the original image. If this is turned off, only the original ’hits’ will be superimposed on the image. If this is turned on, for each hit, a filter-sized rectangle will be placed on the image to mark the filter on the image. Keeping this option on is much more intuitive.
  - *Acceptable Inputs* - `returnPastedFilter: <True or False>`
  - *Default Input* - `returnPastedFilter: True`
  
##### Developer Parameters
The following is a list of parameters that the ordinary user does not need to change. However, if a user wishes to change filters, detection strategies, or filter thresholds, this is how it is done.

- `filterTwoSarcomereSize` - This parameter designates the filter size in relation to the myocytes/tissue we wish to analyze. Unless experimenting with non-default filters, this should be left alone.
  - *Acceptable Inputs* - `filterTwoSarcomereSize: <an integer>`
  - *Default Input* - `filterTwoSarcomereSize: 25`

- `paramDicts` - This holds all of the parameters needed for each individual filtering routine (TT, LT, and TA as well as arbitrary filtering). `filterMode` refers to the type of SNR calculation being utilized for the classification. There are three options for this parameter, `simple` just calculates the convolution of the filter with the image,`regionalDeviation` utilizes the SNR calculation explained in Sect. S.2.2, “Penalty against filter with spectral overlap,” and `punishmentFilter` utilizes the SNR calculation explained in Sect. S.2.1, “Standard deviation criterion.” `filterName` is the name of the filter used for each morphological classification. `punishFilterName` is the name of the ’punishment’ filter used if `filterMode: punishmentFilter` is specified. `gamma` is the scalar used to weight the punishment filter response. `snrThresh` is the threshold used to differentiate classification ‘hits’ versus ‘non-hits’ using the signal to noise ratio. `stdDevThresh` is the threshold used to differentiate classification ‘hits’ versus ‘non-hits’ using the standard deviation criterion used in the `regionalDeviation` filter mode. `inverseSNRrefers` to the flag that indicates whether the classification ‘hits’ are above the `snrThresh` or below. `False` indicates that classification ‘hits’ are above the threshold, `True` indicates ‘hits’ are below the threshold.
  - *Acceptable Inputs*
  ```
  paramDicts: 
    TT: 
      filterMode: <simple, regionalDeviation, punishmentFilter> 
      filterName: <a string> 
      punishFilterName: <a string> 
      gamma: <an integer or float> 
      snrThresh: <an integer or float> 
    LT: 
      filterMode: <simple, regionalDeviation, punishmentFilter> 
      filterName: <a string> 
      snrThresh: <an integer or float> 
      stdDevThresh: <an integer or float> 
    TA: 
      filterMode: <simple, regionalDeviation, punishmentFilter> 
      filterName: <a string> 
      inverseSNR: <True or False> 
      snrThresh: <an integer or float> 
      stdDevThresh: <an integer or float>
  ```
  - *Default Input*
  ```
  paramDicts: 
    TT: 
      filterMode: punishmentFilter 
      filterName: ./myoimages/newSimpleWTFilter.png 
      punishFilterName: ./myoimages/newSimpleWTPunishmentFilter.png 
      gamma: 3.0 
      snrThresh: 0.35 
    LT: 
      filterMode: regionalDeviation 
      filterName: ./myoimages/LongitudinalFilter.png 
      snrThresh: 0.6 
      stdDevThresh: 0.2 
    TA: 
      filterMode: regionalDeviation 
      filterName: ./myoimages/LossFilter.png 
      inverseSNR: True 
      snrThresh: 0.04 
      stdDevThresh: 0.1
  ```
- `useGPU` - This flag turns on/off the utilization of GPU processing. This is currently implemented through the `pyopencl` and `gputools` python libraries. See [Singularity](#singularity) for a description of how to build a singularity container capable of GPU processing.
  - *Acceptable Inputs* - `useGPU: <True or False>`
  - *Default Input* - `useGPU: False`

### Fiji

This is currently under development. This will be filled in as Dylan develops it.

### Singularity
We have developed a pipeline for the construction of a Singularity container for the MatchedMyo repository that is capable of utilizing GPUs for convolution, vastly increasing the speed with which classification can be performed. Additionally, with the creation of a singularity container for the MatchedMyo software, we can port the container to most high performance computing (HPC) resources, again vastly improving the time of classification, opening the possibility of high-throughput analysis. Below, you can find instructions for building your own singularity container to run on your local machine or for running on HPC resources.

@shashank Please fill this in when you get a chance!

# Development
Thank you for considering contributing to the MatchedMyo project! Please contact Dylan Colli at dfco222@g.uky.edu.

### Validation Tests
It is important to validate any code before it is committed (and develop new validation tests as appropriate!). To run a full validation test for the project, execute:
```
$ python matchedmyo.py fullValidation
```
If any validation test is developed for the code base, it is important to add that validation test into the `fullValidation` routine. This is found in the `matchedmyo.py` script in a function titled, `fullValidation`.
To run a 2D test only:
```
$ python matchedmyo.py validate
```

To run a 3D test only:
```
$ python matchedmyo.py validate3D
```

To validate the usage of arbitrary filters:
```
$ python matchedmyo.py validate3D_arbitrary
```

# Community

Hello! Thanks for taking the time to read through the documentation to learn more about the MatchedMyo project. We welcome any sort of dialogue about the project and if you have any questions or concerns, feel free to email Dylan Colli at dfco222@g.uky.edu or see below for issue tracking and feature requests.

### Contribution

Your contributions are always welcome and appreciated. Following are the things you can do to contribute to this project.

1. **Report a bug**
If you think you have encountered a bug, and I should know about it, feel free to report it using the "Issues" tab in the bitbucket repository and I will take care of it.

2. **Request a feature**
You can also request for a feature using the issue tracker by selecting "proposal" when prompted by the "Kind" dialogue.

# Credit/Acknowledgment
The following is a list of contributors to the MatchedMyo project

Dylan Colli - dfco222@g.uky.edu
Ryan Blood
Shashank Bhatt
Peter Kekenes-Huskey - pmke226@g.uky.edu

# License
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. The licensce is included in the `LICENSE.txt` document.