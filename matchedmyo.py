#!/usr/bin/env python2

'''This script will contain all of the necessary wrapper routines to perform analysis on a wide range of 
cardiomyocyte/tissue images.

This is THE script that all sophisticated, general user-level routines will be routed through.
'''

import time
import sys
import util
import numpy as np
import optimizer
import bankDetect as bD
import matplotlib.pyplot as plt
import painter
import matchedFilter as mF
import argparse
import yaml


###################################################################################################
###################################################################################################
###################################################################################################
###
### Class Definitions
###
###################################################################################################
###################################################################################################
###################################################################################################

class Inputs:
  '''Class for the storage of inputs for running through the classification routines,
  giveMarkedMyocyte and give3DMarkedMyocyte. This class is accessed at all levels of 
  characterization so it's convenient to have a way to pass parameters around.
  '''
  def __init__(self,
               imageName = None,
               yamlFileName = None,
               mfOrig=None,
               scopeResolutions=None,
               useGPU=False,
               efficientRotationStorage=True,
              #  ttFilterName = './myoimages/newSimpleWTFilter.png',
              #  ttPunishFilterName = './myoimages/newSimpleWTPunishmentFilter.png',
              #  ltFilterName = './myoimages/LongitudinalFilter.png',
              #  taFilterName = './myoimages/LossFilter.png',
              #  ttFiltering = False,
              #  ltFiltering = False,
              #  taFiltering = False,
               paramDicts = None,
               yamlDict = None,
               ):
    '''
    Inputs:
      imageName -> Name of oringinal image that matched filtering will be performed on.
      yamlfileName -> str. Name of the yaml file for the characterization routine.
      mfOrig -> Original matched filter
      scopeResolutions -> Resolutions of the confocal microscope in pixels per micron. To be read in via YAML
      efficientRotationStorage -> Flag to tell the routine to use the newer, more efficient
                                    way to store information across rotations.
      *FilterName -> Names of the filters used in the matched filtering routines.
      ttFiltering, ltFiltering, taFiltering -> Flags to turn on or off the detection of TT, LT
                                                 and TA morphology.
      yamlDict -> dict. Dictionary read in by the setupYamlInputs() method. yamlFileName must be 
                    specified first.
    '''

    ### Store global class-level parameters
    self.imageName = imageName
    self.yamlFileName = yamlFileName
    self.mfOrig = mfOrig
    self.useGPU = useGPU
    self.efficientRotationStorage = efficientRotationStorage
    self.paramDicts = paramDicts

    ## Setup default dictionaries for classification
    self.setupDefaultDict()
    self.setupDefaultParamDicts()

    ## Update default dictionaries according to yaml file
    if yamlFileName:
      self.setupYamlInputs()
    else:
      self.yamlDict = None

  def setupDefaultDict(self):
    '''This method sets up a dictionary to hold default classification inputs. This will then be 
    updated by updateInputs() method.'''
    dic = dict()
    
    ### Globabl parameters
    dic['imageName'] = ''
    dic['scopeResolutions'] = []
    dic['efficientRotationStore'] = True
    dic['iters'] = [-25,-20,-15,-10,-5,0,5,10,15,20,25]
    
    ### Filtering flags to turn on or off
    dic['filterTypes'] = {
      'TT':False,
      'LT':False,
      'TA':False
    }

    ### Specify filter names
    dic['ttFilterName'] = './myoimages/newSimpleWTFilter.png'
    dic['ttPunishFilterName'] = './myoimages/newSimpleWTPunishmentFilter.png'
    dic['ltFilterName'] = './myoimages/LongitudinalFilter.png'
    dic['taFilterName'] = './myoimages/LossFilter.png'

    ### Store in the class
    self.dic = dic

  def updateDefaultDict(self):
    '''This method updates the default inputs dictionary formed from setupDefulatDict() method
    with the inputs specified in the yaml file.'''

    ### Iterate through all keys specified in yaml and figure out if they have a default value.
    ###   If they do, then we assign the non-default value specified in the yaml file in the 
    ###   dictionary we already formed.
    for key, value in self.yamlDict.iteritems():
      ## Check to see if the key is pointing to the parameter dictionary. If it is, skip this
      ##   since we have functions that update it already
      if key == 'paramDicts':
        continue
      
      ## Here we check if the key is present within the default dictionary. If it is, we can then
      ##   see if a non-default value is specified for it.
      try:
        ## checking to see if a default value is specified
        if self.dic[key] != None:
          ## if it is, we store the non-default value
          self.dic[key] = value
      except:
        ## if the key is not already specified in the default dictionary, then we continue on
        pass
    
  def load_yaml(self):
    '''Function to read and store the yaml dictionary'''
    self.yamlDict = util.load_yaml(self.yamlFileName)

  def setupDefaultParamDicts(self):
    '''This function forms the default parameter dictionaries for each filtering type, TT, LT, and 
    TA.'''
    ### Form dictionary that contains default parameters
    storageDict = dict()

    ## Check dimensionality of image to specify correct parameter dictionaries
    filterTypes = ['TT','LT','TA']

    ### Assign default parameters
    storageDict['TT'] = optimizer.ParamDict(typeDict=filterTypes[0])
    storageDict['LT'] = optimizer.ParamDict(typeDict=filterTypes[1])
    storageDict['TA'] = optimizer.ParamDict(typeDict=filterTypes[2])

    self.paramDicts = storageDict

  def updateParamDicts(self):
    '''This function updates the parameter dictionaries previously formed in the method, 
    setupDefaultParamDicts() with the specifications in the yaml file.'''
    ## Default parameter dictionary is set to 2D, so if image is 3D, we need to update these
    if self.dic['dimensions'] == 3:
      ## Loop through and update each parameter dictionary
      for filteringType in self.paramDicts.keys():
        self.paramDicts[filteringType] = optimizer.ParamDict(typeDict=filteringType+'_3D')

    ## Iterate through and assign non-default parameters to correct dictionaries
    for key, paramDict in self.yamlDict.iteritems():
      ## Check to see if the key is pointing to a parameter dictionary
      if not any(key == filt for filt in ['TT','LT','TA']):
        continue

      ## Go through and assign all specified non-default parameters in the yaml file to the 
      ##   storageDict
      for parameterName, parameter in paramDict.iteritems():
        self.paramDicts[key][parameterName] = parameter

  def updateInputs(self):
    '''This function updates the inputs class that's formed in matchedmyo.py script 
    
    Also updates parameteres based on parameters that are specified in the yaml dictionary 
    that is stored within this class.'''

    ### Read in the original image and determine number of dimensions from this
    self.imgOrig = util.ReadImg(self.yamlDict['imageName'], renorm=True)
    self.dic['dimensions'] = len(self.imgOrig.shape)

    ### Form the correct default parameter dictionaries from this dimensionality measurement
    self.updateDefaultDict()
    self.updateParamDicts()

  def setupYamlInputs(self):
    '''This function sets up inputs if a yaml file name is specified'''
    self.load_yaml()
    # self.setupDefaultDict()
    # self.setupDefaultParamDicts()
    self.updateInputs()


###################################################################################################
###################################################################################################
###################################################################################################
###
### Individual Filtering Routines
###
###################################################################################################
###################################################################################################
###################################################################################################

def TT_Filtering(inputs,
                 iters,
                 paramDict,
                 ttThresh=None,
                 ttGamma=None,
                 returnAngles=False):
  '''
  Takes inputs class that contains original image and performs WT filtering on the image
  '''
  print "TT Filtering"
  start = time.time()

  ### Specify necessary inputs
  ## Read in filter
  ttFilter = util.LoadFilter(inputs.dic['ttFilterName'])
  inputs.mfOrig = ttFilter

  paramDict['covarianceMatrix'] = np.ones_like(inputs.imgOrig)
  paramDict['mfPunishment'] = util.LoadFilter(inputs.dic['ttPunishFilterName'])
  print "phase out GPU"
  paramDict['useGPU'] = inputs.useGPU

  ## Check to see if parameters are manually specified
  if ttThresh != None:
    paramDict['snrThresh'] = ttThresh
  if ttGamma != None:
    paramDict['gamma'] = ttGamma

  ### Perform filtering
  WTresults = bD.DetectFilter(inputs,paramDict,iters,returnAngles=returnAngles)  

  end = time.time()
  print "Time for WT filtering to complete:",end-start,"seconds"

  return WTresults

def LT_Filtering(inputs,
                 iters,
                 paramDict,
                 returnAngles=False
                 ):
  '''
  Takes inputs class that contains original image and performs LT filtering on the image
  '''

  print "LT filtering"
  start = time.time()

  ### Specify necessary inputs
  inputs.mfOrig = util.LoadFilter(inputs.dic['ltFilterName'])
  print "Seriously, phase out GPU"
  paramDict['useGPU'] = inputs.useGPU

  ### Perform filtering
  LTresults = bD.DetectFilter(inputs,paramDict,iters,returnAngles=returnAngles)

  end = time.time()
  print "Time for LT filtering to complete:",end-start,"seconds"

  return LTresults

def TA_Filtering(inputs,
                 paramDict,
                 iters=None,
                 returnAngles=False,
                 ):
  '''
  Takes inputs class that contains original image and performs Loss filtering on the image
  '''
  print "TA filtering"
  start = time.time()

  ### Specify necessary inputs
  inputs.mfOrig = util.LoadFilter(inputs.dic['taFilterName'])
  print "PHASE OUT GPU"
  paramDict['useGPU'] = inputs.useGPU
  
  ## Check to see if iters (filter rotations are specified) if they aren't we'll just use 0 and 45
  ##   degrees since the loss filter is symmetric
  if iters != None:
    Lossiters = iters
  else:
    Lossiters = [0, 45] 
  
  ### Perform filtering
  Lossresults = bD.DetectFilter(inputs,paramDict,Lossiters,returnAngles=returnAngles)

  end = time.time()
  print "Time for TA filtering to complete:",end-start,"seconds"

  return Lossresults

def analyzeTT_Angles(testImageName,
                     inputs,
                     iters,
                     ImgTwoSarcSize,
                     WTstackedHits,
                     ttFilterName = "./myoimages/newSimpleWTFilter.png"
                     ):
  '''This function analyzes the tubule striation angle for the transverse tubule filter.
  The routine does this by smoothing the original image with a small smoothing filter, constructing
  a TT filter with a larger field of view (longer tubules in the filter), and calculating the SNR 
  using this filter. Then, the routine uses the previously detected hits from the TT_Filtering()
  function to mask out the hits from the larger FOV filter. Teh reason this is necessary is due to 
  the fact that the original TT filter is very specific in terms of hits, but is extremely variable 
  in terms of what rotation the hit occurs at.
  
  Inputs:
    testImageName -> str. Name of the image that you are analyzing.
    inputs -> class. Inputs class already constructed in the giveMarkedMyocyte() function.
    iters -> list. List of iterations (rotations) at which we are analyzing filter response
    ImgTwoSarcSize -> int. Size of the filter/image two sarcomere size.
    WTstackedHits -> numpy array. Array where hits are marked as their SNR and non-hits are marked
                       as zero.
    ttFilterName -> str. Name of the transverse tubule filter used in this analysis.
  '''

  ### Read in original colored image
  cImg = util.ReadImg(testImageName,cvtColor=False)

  ### perform smoothing on the original image
  dim = 5
  kernel = np.ones((dim,dim),dtype=np.float32)
  kernel /= np.sum(kernel)
  smoothed = mF.matchedFilter(inputs.imgOrig,kernel,demean=False)

  ### make longer WT filter so more robust to striation angle deviation
  ttFilter = util.LoadFilter(ttFilterName)
  longFilter = np.concatenate((ttFilter,ttFilter,ttFilter))
    
  rotInputs = Inputs(
    imageName = inputs.imageName,
    yamlFileName = inputs.yamlFileName,
    mfOrig = longFilter)
  rotInputs.imgOrig = smoothed

  params = optimizer.ParamDict(typeDict='WT')
  params['snrThresh'] = 0 # to pull out max hit
  params['filterMode'] = 'simple' # we want no punishment since that causes high variation
    
  ### perform simple filtering
  smoothedWTresults = bD.DetectFilter(rotInputs,params,iters,returnAngles=True)
  smoothedHits = smoothedWTresults.stackedAngles

  ### pull out actual hits from smoothed results
  smoothedHits[WTstackedHits == 0] = -1

  coloredAngles = painter.colorAngles(cImg,smoothedHits,iters)

  coloredAnglesMasked = util.ReadResizeApplyMask(coloredAngles,testImageName,
                                            ImgTwoSarcSize,
                                            filterTwoSarcSize=ImgTwoSarcSize)

  ### Check to see if we've used the new efficient way of storing information in the algorithm.
  ###   If we have, we already have the rotational information stored
  if rotInputs.efficientRotationStorage:
    angleCounts = smoothedHits.flatten()
  else:
    ## Otherwise, we have to go through and manually pick out rotations from their indexes in the 
    ##  iters list
    stackedAngles = smoothedHits
    dims = np.shape(stackedAngles)
    angleCounts = []
    for i in range(dims[0]):
      for j in range(dims[1]):
        rotArg = stackedAngles[i,j]
        if rotArg != -1:
          ### indicates this is a hit
          angleCounts.append(iters[rotArg])

  return angleCounts, coloredAnglesMasked

###################################################################################################
###################################################################################################
###################################################################################################
###
### Wrappers for Full Analysis of User-Supplied Images
###
###################################################################################################
###################################################################################################
###################################################################################################

def giveMarkedMyocyte(
      inputs,
      ImgTwoSarcSize=None,
      tag = "default_",
      writeImage = False,
      iters=[-25,-20,-15,-10,-5,0,5,10,15,20,25],
      returnAngles=False,
      returnPastedFilter=True,
      useGPU=False,
      fileExtension=".pdf",
      # efficientRotationStorage=True,
      ):
  '''
  This function is the main workhorse for the detection of features in 2D myocytes.
    See give3DMarkedMyocyte() for better documentation.
    TODO: Better document this
  '''
 
  start = time.time()

  ### Setup Parameters/Inputs
  ## Read in preprocessed image
  #img = util.ReadImg(testImage,renorm=True)

  ## defining inputs to be read by DetectFilter function
  #inputs = Inputs(imgOrig = util.ReadResizeApplyMask(img,testImage))

  ## Read in the yaml file if it is specified
  # yamlDict = util.load_yaml(yamlFileName)

  ## Store non-default parameters from yaml file into inputs
  # util.updateInputsFromYaml(inputs, yamlDict)

  ## Form parameter dictionaries for the classification
  #paramDicts = util.makeParamDicts(inputs=inputs)

  ### Perform Filtering Routines
  ## Transverse Tubule Filtering
  if inputs.dic['filterTypes']['TT']:
    WTresults = TT_Filtering(
      inputs = inputs,
      iters = iters,
      paramDict = inputs.paramDicts['TT'],
      returnAngles = returnAngles
    )
    WTstackedHits = WTresults.stackedHits
  else:
    WTstackedHits = np.zeros_like(inputs.imgOrig)

  ## Longitudinal Tubule Filtering
  if inputs.dic['filterTypes']['LT']:
    LTresults = LT_Filtering(
      inputs = inputs,
      iters = iters,
      paramDict = inputs.paramDicts['LT'],
      returnAngles = returnAngles
    )
    LTstackedHits = LTresults.stackedHits
  else:
    LTstackedHits = np.zeros_like(inputs.imgOrig)

  ## Tubule Absence Filtering
  if inputs.dic['filterTypes']['TA']:
    Lossresults = TA_Filtering(
      inputs=inputs,
      paramDict = inputs.paramDicts['TA'],
      returnAngles = returnAngles
    )
    LossstackedHits = Lossresults.stackedHits
  else:
    LossstackedHits = np.zeros_like(inputs.imgOrig)
 
  ### Perform Post-Processing
  ## Read in colored image for marking hits
  cI = util.ReadImg(inputs.yamlDict['imageName'],cvtColor=False).astype(np.float64)
  # killing the brightness a bit so looking at results isn't like staring at the sun
  cI *= 0.85
  cI = np.round(cI).astype(np.uint8)

  ## Marking superthreshold hits for loss filter
  LossstackedHits[LossstackedHits != 0] = 255
  LossstackedHits = np.asarray(LossstackedHits, dtype='uint8')

  ## applying a loss mask to attenuate false positives from WT and Longitudinal filter
  WTstackedHits[LossstackedHits == 255] = 0
  LTstackedHits[LossstackedHits == 255] = 0

  ## marking superthreshold hits for longitudinal filter
  LTstackedHits[LTstackedHits != 0] = 255
  LTstackedHits = np.asarray(LTstackedHits, dtype='uint8')

  ## masking WT response with LT mask so there is no overlap in the markings
  WTstackedHits[LTstackedHits == 255] = 0

  ## marking superthreshold hits for WT filter
  WTstackedHits[WTstackedHits != 0] = 255
  WTstackedHits = np.asarray(WTstackedHits, dtype='uint8')

  ## apply preprocessed masks
  wtMasked = util.ReadResizeApplyMask(
    WTstackedHits,
    inputs.yamlDict['imageName'],
    ImgTwoSarcSize,
    filterTwoSarcSize=ImgTwoSarcSize
  )
  ltMasked = util.ReadResizeApplyMask(
    LTstackedHits,
    inputs.yamlDict['imageName'],
    ImgTwoSarcSize,
    filterTwoSarcSize=ImgTwoSarcSize
  )
  lossMasked = util.ReadResizeApplyMask(
    LossstackedHits,
    inputs.yamlDict['imageName'],
    ImgTwoSarcSize,
    filterTwoSarcSize=ImgTwoSarcSize
  )

  if not returnPastedFilter:
    ### create holders for marking channels
    WTcopy = cI[:,:,0]
    LTcopy = cI[:,:,1]
    Losscopy = cI[:,:,2]

    ### color corrresponding channels
    WTcopy[wtMasked == 255] = 255
    LTcopy[ltMasked == 255] = 255
    Losscopy[lossMasked == 255] = 255

    ### mark mask outline on myocyte
    cI = util.markMaskOnMyocyte(
      cI,
      inputs.yamlDict['imageName']
    )
    if writeImage:
      ### mark mask outline on myocyte
      #cI_written = util.markMaskOnMyocyte(cI,testImage)
      cI_written = cI

      ### write output image
      plt.figure()
      plt.imshow(util.switchBRChannels(cI_written))
      plt.gcf().savefig(tag+"_output"+fileExtension,dpi=300)

  if returnPastedFilter:
    ### We create a dummy image to hold the marked hits
    dummy = np.zeros_like(cI)
    dummy = util.markPastedFilters(lossMasked, ltMasked, wtMasked, dummy)
    
    ### apply mask again so as to avoid content > 1.0
    dummy = util.ReadResizeApplyMask(
      dummy,
      inputs.yamlDict['imageName'],
      ImgTwoSarcSize,
      filterTwoSarcSize=ImgTwoSarcSize
    )
    cI[dummy==255] = 255

    ### Now based on the marked hits, we can obtain an estimate of tubule content
    estimatedContent = util.estimateTubuleContentFromColoredImage(cI)
  
    if writeImage:
      ### mark mask outline on myocyte
      #cI_written = util.markMaskOnMyocyte(cI.copy(),testImage)
      cI_written = cI

      ### write outputs	  
      plt.figure()
      plt.imshow(util.switchBRChannels(cI_written))
      plt.gcf().savefig(tag+"_output"+fileExtension,dpi=300)

  if returnAngles:
    angleCounts, coloredAnglesMasked = analyzeTT_Angles(
      testImageName=inputs.yamlDict['imageName'],
      inputs=inputs,
      iters=iters,
      ImgTwoSarcSize=ImgTwoSarcSize,
      WTstackedHits=WTstackedHits
    )

    if writeImage:
      #cv2.imwrite(tag+"_angles_output.png",coloredAnglesMasked)
      plt.figure()
      plt.imshow(util.switchBRChannels(coloredAnglesMasked))
      plt.gcf().savefig(tag+"_angles_output"+fileExtension)
    
    end = time.time()
    tElapsed = end - start
    print "Total Elapsed Time: {}s".format(tElapsed)
    return cI, coloredAnglesMasked, angleCounts

  end = time.time()
  tElapsed = end - start
  print "Total Elapsed Time: {}s".format(tElapsed)
  return cI 

def give3DMarkedMyocyte(
      # testImage,
      # scopeResolutions,
      # ttFilterName=None,
      # ltFilterName=None,
      # taFilterName=None,
      # ttPunishFilterName=None,
      # ltPunishFilterName=None,
      inputs,
      ImgTwoSarcSize=None,
      tag = None,
      xiters=[-10,0,10],
      yiters=[-10,0,10],
      ziters=[-10,0,10],
      returnAngles=False,
      returnPastedFilter=True,
      # efficientRotationStorage=True,
      ):
  '''
  This function is for the detection and marking of subcellular features in three dimensions. 

  Inputs:
    testImage -> str. Name of the image to be analyzed. NOTE: This image has previously been preprocessed by 
                   XXX routine.
    scopeResolutions -> list of values (ints or floats). List of resolutions of the confocal microscope for x, y, and z.
    tag -> str. Base name of the written files 
    xiters -> list of ints. Rotations with which the filters will be rotated about the x axis (yz plane)
    yiters -> list of ints. Rotations with which the filters will be rotated about the y axis (xz plane)
    ziters -> list of ints. Rotations with which the filters will be rotated about the z axis (xy plane)
    returnAngles -> Bool. Whether or not to return the angles with which the hit experienced the greatest SNR
                      NOTE: Do I need to delete this? Not like I'll be able to get a colormap for this
    returnPastedFilter -> Bool. Whether or not to paste filter sized unit cells where detections occur.
                            This translates to much more intuitive hit markings.
  
  Outputs:
    TBD
  '''
  start = time.time()

  ### Read in preprocessed test image and store in inputs class for use in all subroutines
  # inputs = Inputs(
  #   # imgOrig = util.ReadImg(testImage, renorm=True),
  #   # scopeResolutions = scopeResolutions
  #   yamlFileName=args.yamlFile
  # )

  ### Read in the yaml file if it is specified
  # yamlDict = util.load_yaml(yamlFileName)

  ### Form parameter dictionaries for classification
  #paramDicts = util.makeParamDicts(inputs=inputs)

  ### Form flattened iteration matrix containing all possible rotation combinations
  flattenedIters = []
  for i in xiters:
    for j in yiters:
      for k in ziters:
        flattenedIters.append( [i,j,k] )

  ### Transverse Tubule Filtering
  if inputs.dic['filterTypes']['TT']:
    TTresults = TT_Filtering(
      inputs,
      flattenedIters,
      paramDict = inputs.paramDicts['TT'],
      returnAngles = returnAngles
    )
    TTstackedHits = TTresults.stackedHits
  else:
    TTstackedHits = np.zeros_like(inputs.imgOrig)

  ### Longitudinal Tubule Filtering
  if inputs.dic['filterTypes']['LT']:
    LTresults = LT_Filtering(
      inputs,
      flattenedIters,
      paramDict = inputs.paramDicts['LT'],
      returnAngles = returnAngles
    )
    LTstackedHits = LTresults.stackedHits
  else:
    LTstackedHits = np.zeros_like(inputs.imgOrig)

  ### Tubule Absence Filtering
  if inputs.dic['filterTypes']['TA']:
    ## form tubule absence flattened rotation matrix. Choosing to look at tubule absence at one rotation right now.
    taIters = [[0,0,0]]
    TAresults = TA_Filtering(
      inputs,
      iters=taIters,
      paramDict = inputs.paramDicts['TA'],
      returnAngles = returnAngles
    )
    TAstackedHits = TAresults.stackedHits
  else:
    TAstackedHits = np.zeros_like(inputs.imgOrig)

  ### Mark Detections on the Image
  cImg = np.stack(
    (
      inputs.imgOrig,
      inputs.imgOrig,
      inputs.imgOrig
    ),
    axis=-1
  )
  ## Scale cImg and convert to 8 bit for color marking
  alpha = 0.75
  cImg = cImg.astype(np.float)
  cImg /= np.max(cImg)
  cImg *= 255 * alpha
  cImg = cImg.astype(np.uint8)
  if returnPastedFilter:
    ## Use routine to mark unit cell sized cuboids around detections
    cImg = util.markPastedFilters(TAstackedHits,
                             LTstackedHits,
                             TTstackedHits,
                             cImg,
                             ttName = inputs.dic['ttFilterName'],
                             ltName = inputs.dic['ltFilterName'],
                             taName = inputs.dic['taFilterName'])

    ### 'Measure' cell volume just by getting measure of containing array
    cellVolume = np.float(np.product(inputs.imgOrig.shape))

    ### Now based on the marked hits, we can obtain an estimate of tubule content
    estimatedContent = util.estimateTubuleContentFromColoredImage(
      cImg,
      totalCellSpace=cellVolume,
      taFilterName = inputs.dic['taFilterName'],
      ltFilterName = inputs.dic['ltFilterName'],
      ttFilterName = inputs.dic['ttFilterName']
    )

  else:
    ## Just mark exactly where detection is instead of pasting unit cells on detections
    cImg[:,:,:,2][TAstackedHits > 0] = 255
    cImg[:,:,:,1][LTstackedHits > 0] = 255
    cImg[:,:,:,0][TTstackedHits > 0] = 255

    ### Determine percentages of volume represented by each filter
    cellVolume = np.float(np.product(inputs.imgOrig.shape))
    taContent = np.float(np.sum(cImg[:,:,:,2] == 255)) / cellVolume
    ltContent = np.float(np.sum(cImg[:,:,:,1] == 255)) / cellVolume
    ttContent = np.float(np.sum(cImg[:,:,:,0] == 255)) / cellVolume

    print "TA Content per Cell Volume:", taContent
    print "LT Content per Cell Volume:", ltContent
    print "TT Content per Cell Volume:", ttContent

  if returnAngles:
    print "WARNING: Striation angle analysis is not yet available in 3D"
  
  ### Save detection image
  if tag:
    util.Save3DImg(cImg,tag+'.tif',switchChannels=True)

  end = time.time()
  print "Time for algorithm to run:",end-start,"seconds"
  
  return cImg

# def giveTissueAnalysis(grayTissue,
#                        iters = [-50,0], #TODO: FIX THIS TO ACTUAL RANGE
#                        ttFilterName="./myoimages/newSimpleWTFilter.png",
#                        ttPunishFilterName="./myoimages/newSimpleWTPunishmentFilter.png"):
#   '''This function is for the analysis and classification of subcellular morphology in confocal images
#   of tissue sections containing many myocytes. This presents a different set of challengs than the 
#   single myocyte analysis and for that reason, the longitudinal tubule response degrades significantly.
#   Thus, we only include the analysis of transverse tubule filter hits and rotation angles as well as 
#   tubule absence filter hits.

#   Inputs:
#     TODO
#   '''
#   ### BEGIN CODE TAKEN FROM BRANCH OF GPU DETECT
#   '''
#   This function will take the previously preprocessed tissue and run the WT 
#   filter across the entire image. From this, we will apply a large smoothing
#   function across the entire detection image. This should hopefully provide
#   a nice gradient that shows TT-density loss as proximity to the infarct
#   increases

#   NOTE: This function is meant to be called repeatedly by a bash script.
#           Some library utilized herein is leaky and eats memory for 
#           breakfast.

#   Inputs:
#     iteration - rotation at which to perform the detection

#   Outputs:
#     None

#   Written Files:
#     "tissueDetections_<iteration>.pkl"
#   '''
#   ### Load in tissue
#   grayTissue = LoadTissue()

#   # ### Load in tissue
#   # params = tis.params
#   # grayTissue = tis.Setup().astype(np.float32)
#   # grayTissue /= np.max(grayTissue)
  
#   print "Size of single tissue image in Gigabytes:",sys.getsizeof(grayTissue) / 1e9

#   # ttFilterName = "./myoimages/newSimpleWTFilter.png"
#   # ttPunishFilterName = "./myoimages/newSimpleWTPunishmentFilter.png"

#   inputs = empty()
#   inputs.imgOrig = grayTissue
#   # inputs.useGPU = False

#   # returnAngles = False
#   # returnAngles = True

#   startTime = time.time()
#   # ### This is mombo memory intensive
#   # thisIteration = [iteration]
#   tissueResults = TT_Filtering(inputs,iters,ttFilterName,ttPunishFilterName,None,None,False)
#   # #print "before masking"
#   # resultsImage = tissueResults.stackedHits > 0 
#   # #print "after masking"


#   # ### save tissue detection results
#   # #print "before dumping"
#   # name = "tissueDetections_"+str(iteration)
#   # ## consider replacing with numpy save function since it is much quicker
#   # #pkl.dump(resultsImage,open(name,'w'))
#   # np.save(name,resultsImage)
#   # #print "after dumping"
#   endTime = time.time()

#   print "Time for algorithm to run:",endTime-startTime

###################################################################################################
###################################################################################################
###################################################################################################
###
###  Validation Routines
###
###################################################################################################
###################################################################################################
###################################################################################################

def fullValidation(args):
  '''This routine wraps all of the written validation routines.
  This should be run EVERYTIME before changes are committed and pushed to the repository.
  '''

  validate(args)
  validate3D(args)

def validate(args,
             display=False
             ):
  '''This function serves as a validation routine for the 2D functionality of this repo.
  
  Inputs:
    display -> Bool. If True, display the marked image
  '''
  ### Specify the yaml file NOTE: This will be done via command line for main classification routines
  yamlFile = './YAML_files/validate.yml'

  ### Setup inputs for classification run
  inputs = Inputs(
    yamlFileName = yamlFile
  )

  ### Run algorithm to pull out content and rotation info
  markedImg, _, angleCounts = giveMarkedMyocyte(
    inputs=inputs,
    returnAngles=True
  )

  if display:
    plt.figure()
    plt.imshow(markedImg)
    plt.show()

  print "\nThe following content values are for validation purposes only.\n"

  ### Calculate TT, LT, and TA content  
  ttContent, ltContent, taContent = util.assessContent(markedImg)

  assert(abs(ttContent - 103050) < 1), "TT validation failed."
  assert(abs(ltContent -  68068) < 1), "LT validation failed."
  assert(abs(taContent - 156039) < 1), "TA validation failed."

  ### Calculate the number of hits at rotation equal to 5 degrees
  numHits = np.count_nonzero(np.asarray(angleCounts) == 5)
  print "Number of Hits at Rotation = 5 Degrees:", numHits
  assert(abs(numHits - 1621) < 1), "Rotation validation failed"

  print "\nPASSED!\n"

def validate3D(args):
  '''This function serves as a validation routine for the 3D functionality of this repo.

  Inputs:
    None
  '''
  ### Specify the yaml file. NOTE: This will be done via command line for main classification routines
  yamlFile = './YAML_files/validate3D.yml'

  ### Setup input parameters
  inputs = Inputs(
    yamlFileName = yamlFile
  )

  ### Define parameters for the simulation of the cell
  ## Probability of finding a longitudinal tubule unit cell
  ltProb = 0.3
  ## Probability of finding a tubule absence unit cell
  taProb = 0.3
  ## Amplitude of the Guassian White Noise
  noiseAmplitude = 0.
  ## Define scope resolutions for generating the filters and the cell. This is in x, y, and z resolutions
  # scopeResolutions = [10,10,5] #[vx / um]
  ## x, y, and z Dimensions of the simulated cell [microns]
  cellDimensions = [10, 10, 20]
  ## Define test file name
  testName = "./myoimages/3DValidationData.tif"
  ## Give names for your filters. NOTE: These are hardcoded in the filter generation routines in util.py
  # ttName = './myoimages/TT_3D.tif'
  # ttPunishName = './myoimages/TT_Punishment_3D.tif'
  # ltName = './myoimages/LT_3D.tif'
  # taName = './myoimages/TA_3D.tif'

  ### Simulate the small 3D cell
  util.generateSimulated3DCell(LT_probability=ltProb,
                               TA_probability=taProb,
                               noiseAmplitude=noiseAmplitude,
                               scopeResolutions=inputs.dic['scopeResolutions'],
                               cellDimensions=cellDimensions,
                               fileName=testName,
                               seed=1001,
                               )

  ### Analyze the 3D cell
  markedImage = give3DMarkedMyocyte(#testImage=testName,
                                    # scopeResolutions = scopeResolutions,
                                    #ttFilterName = ttName,
                                    #ttPunishFilterName = ttPunishName,
                                    #ltFilterName = ltName,
                                    #taFilterName = taName,
                                    inputs = inputs,
                                    xiters = [0],
                                    yiters = [0],
                                    ziters = [0],
                                    tag = '3DValidationData_analysis')

  print "\nThe following content values are for validation purposes only.\n"

  ### Assess the amount of TT, LT, and TA content there is in the image 
  ttContent, ltContent, taContent = util.assessContent(markedImage)

  ### Check to see that they are in close agreement with previous values
  ###   NOTE: We have to have a lot of wiggle room since we're generating a new cell for each validation
  assert(abs(ttContent - 301215) < 1), "TT validation failed."
  assert(abs(ltContent -  53293) < 1), "LT validation failed."
  assert(abs(taContent - 409003) < 1), "TA validation failed."
  print "\nPASSED!\n"

###################################################################################################
###################################################################################################
###################################################################################################
###
### Command Line Functionality
###
###################################################################################################
###################################################################################################
###################################################################################################

def run(args):
  '''This runs the main classification routines from command line
  '''

  ### Setup the inputs class
  inputs = Inputs(
    yamlFileName=args.yamlFile
  )

  ### Determine if classification is 2D or 3D and run the correct routine for it
  dim = len(np.shape(inputs.imgOrig))
  if dim == 2:
    giveMarkedMyocyte(inputs = inputs)
  elif dim == 3:
    give3DMarkedMyocyte(inputs = inputs)
  else:
    raise RuntimeError("The dimensions of the image specified in {} is not supported.".format(args.yamlFile))

def main(args):
  '''The routine through which all command line functionality is routed.
  '''
  ### Get a list of all function names in the script
  functions = globals()

  functions[args.functionToCall](args)


### Begin argument parser for command line functionality IF function is called via command line
if __name__ == "__main__":
  description = '''This is the main script for the analysis of 2D and 3D confocal images of 
  cardiomyocytes and cardiac tissues.
  '''
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('functionToCall', 
                      type=str,
                      help='The function to call within this script')
  parser.add_argument('--yamlFile',
                      type=str,
                      help='The name of the .yml file containing parameters for classification')
  args = parser.parse_args()
  main(args)
