#!/usr/bin/env python2

'''This script will contain all of the necessary wrapper routines to perform analysis on a wide range of 
cardiomyocyte/tissue images.

This is THE script that all sophisticated, general user-level routines will be routed through.
'''

import os
import time
import sys
import datetime
import csv
import util
import numpy as np
import optimizer
import bankDetect as bD
import matplotlib.pyplot as plt
import painter
import matchedFilter as mF
import argparse
import yaml
import preprocessing as pp


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
               colorImage = None,
               yamlFileName = None,
               mfOrig=None,
               scopeResolutions=None,
               efficientRotationStorage=True,
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
    dic['scopeResolutions'] = {
      'x': None,
      'y': None,
      'z': None
    }
    dic['efficientRotationStorage'] = True
    dic['iters'] = [-25,-20,-15,-10,-5,0,5,10,15,20,25]
    dic['returnAngles'] = False
    dic['returnPastedFilter'] = True
    dic['preprocess'] = True
    dic['filterTwoSarcomereSize'] = 25

    ### Output parameter dictionary
    dic['outputParams'] = {
      'fileRoot': None,
      'fileType': 'png',
      'dpi': 300,
      'saveHitsArray': False,
      'csvFile': './results/classification_results.csv'
    }
    
    ### Filtering flags to turn on or off
    dic['filterTypes'] = {
      'TT':False,
      'LT':False,
      'TA':False
    }

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

      ## Check to see if the key is pointing to the outputParams dictionary specified in the YAML file
      if key == "outputParams":
        ## iterate through the dictionary specified in the yaml file and store non-default values
        for outputKey, outputValue in value.iteritems():
          self.dic['outputParams'][outputKey] = outputValue
      
      ## Here we check if the key is present within the default dictionary. If it is, we can then
      ##   see if a non-default value is specified for it.
      try:
        ## checking to see if a default value is specified
        if self.dic[key] != None:
          ## if it is, we store the non-default value(s)
          self.dic[key] = value
      except:
        ## if the key is not already specified in the default dictionary, then we continue on
        pass
    
    ### Check to see if '.csv' is present in the output csv file
    if self.dic['outputParams']['csvFile'][-4:] != '.csv':
      self.dic['outputParams']['csvFile'] = self.dic['outputParams']['csvFile'] + '.csv'

    ### Convert the scope resolutions into a list
    if isinstance(self.dic['scopeResolutions'], dict):
      self.dic['scopeResolutions'] = [
        self.dic['scopeResolutions']['x'],
        self.dic['scopeResolutions']['y'],
        self.dic['scopeResolutions']['z']
      ]

    ### Flatten out iters if it is still a dictionary. This is necessary for 3D classification where
    ###   there are three axes of rotation
    if isinstance(self.dic['iters'], dict):
      flattenedIters = []
      for i in self.dic['iters']['x']:
        for j in self.dic['iters']['y']:
          for k in self.dic['iters']['z']:
            flattenedIters.append( [i,j,k] )
      self.dic['iters'] = flattenedIters
      
    
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
    try:
      ## If this works, there are parameter options specified
      yamlParamDictOptions = self.yamlDict['paramDicts']
      for filterType, paramDict in yamlParamDictOptions.iteritems():
        ## Go through and assign all specified non-default parameters in the yaml file to the 
        ##   storageDict
        for parameterName, parameter in paramDict.iteritems():
          self.paramDicts[filterType][parameterName] = parameter

    except:
      # if the above doesn't work, then there are no parameter options specified and we can exit 
      return
    
  def updateInputs(self):
    '''This function updates the inputs class that's formed in matchedmyo.py script 
    
    Also updates parameteres based on parameters that are specified in the yaml dictionary 
    that is stored within this class.'''

    ### Read in the original image and determine number of dimensions from this
    self.imgOrig = util.ReadImg(self.yamlDict['imageName'], renorm=True)
    self.dic['dimensions'] = len(self.imgOrig.shape)

    ### Make a 'color' image with 3 channels in the final index to represent the color channels
    ###   We also want to dampen the brightness a bit for display purposes, so we multiply by an
    ###   alpha value
    alpha = 0.85
    colorImageMax = 255
    self.colorImage = np.dstack((self.imgOrig, self.imgOrig, self.imgOrig)).astype(np.float32)
    self.colorImage *= alpha * colorImageMax
    self.colorImage = self.colorImage.astype(np.uint8)

    ### Form the correct default parameter dictionaries from this dimensionality measurement
    self.updateDefaultDict()
    self.updateParamDicts()

    ### Check to see if we need to preprocess the image at all
    if self.dic['preprocess']:
      self.imgOrig = pp.preprocess(self.dic['imageName'], self.dic['filterTwoSarcomereSize'])

  def load_yaml(self):
    '''Function to read and store the yaml dictionary'''
    self.yamlDict = util.load_yaml(self.yamlFileName)

  def check_yaml_for_errors(self):
    '''This function checks that the user-specified parameters read in through load_yaml() are valid'''

    ### Check that the scope resolutions are specified correctly
    for value in self.dic['scopeResolutions']:
      if not isinstance(value, (float, int, type(None))):
        raise RuntimeError("Scope resolutions are not specified correctly. Ensure that the "
                           +"resolutions are integers, floats, or are left blank.")

    ### Check efficientRotationStorage
    if not isinstance(self.dic['efficientRotationStorage'], bool):
      raise RuntimeError("The efficientRotationStorage parameter is not a boolean type "
                         +"(True or False). Ensure that this is correct in the YAML file.")

    ### Check the rotations
    if not isinstance(self.dic['iters'], list):
      raise RuntimeError('Double check that iters is specified correctly in the YAML file')
    for value in self.dic['iters']:
      ## Check if the entries are lists (3D) or floats/ints (2D)
      if not isinstance(value, (float, int, list)):
          raise RuntimeError('Double check that the values specified for the rotations (iters) are '
                             +'integers or floats.')

    if not isinstance(self.dic['returnAngles'], bool):
      raise RuntimeError('Double check that returnAngles is either True or False in the YAML file.')

    if not isinstance(self.dic['returnPastedFilter'], bool):
      raise RuntimeError('Double check that returnPastedFilter is either True or False in the YAML file.')

    if not isinstance(self.dic['filterTwoSarcomereSize'], int):
      raise RuntimeError('Double check that filterTwoSarcomereSize is an integer.')

    ### Check output parameters
    if not isinstance(self.dic['outputParams']['fileRoot'], (type(None), str)):
      raise RuntimeError('Ensure that the fileRoot parameter in outputParams is either a string '
                         +'or left blank.')
    if not self.dic['outputParams']['fileType'] in ['png','tif','pdf']:
      raise RuntimeError('Double check that fileType in outputParams is either "png," "tif," or "pdf."')
    if not isinstance(self.dic['outputParams']['dpi'], int):
      raise RuntimeError('Ensure that dpi in outputParams is an integer.')
    if not isinstance(self.dic['outputParams']['saveHitsArray'], bool):
      raise RuntimeError('Ensure that saveHitsArray in outputParams is either True or False')
    if not isinstance(self.dic['outputParams']['csvFile'], str):
      raise RuntimeError('Ensure that csvFile in outputParams is a string.')

    ### Check that filter types is either true or false for all entries
    for key, value in self.dic['filterTypes'].iteritems():
      if not isinstance(value, bool):
        raise RuntimeError('Check that {} in filterTypes is either True or False'.format(key))

    if self.dic['returnAngles']:
      if not self.dic['filterTypes']['TT']:
        raise RuntimeError('TT filtering must be turned on if returnAngles is specified as True')
    
  def setupYamlInputs(self):
    '''This function sets up inputs if a yaml file name is specified'''
    ### Check that the YAML file exists
    if not os.path.isfile(self.yamlFileName):
      raise RuntimeError("Double check that the yaml file that was specified is correct. Currently, "
                         +"the YAML file that was specified does not exist.")

    self.load_yaml()
    
    ### Double check that the image exists
    if not os.path.isfile(self.yamlDict['imageName']):
      raise RuntimeError('The specified image does not exist. Double-check that imageName is correct.')
    
    self.updateInputs()
    self.check_yaml_for_errors()

class ClassificationResults:
  '''This class holds all of the results that we will need to store.'''
  def __init__(self,
               markedImage=None,
               markedAngles=None,
               ttContent=None,
               ltContent=None,
               taContent=None):
    '''
    Inputs:
      markedImage -> Numpy array. The image with TT, LT, and TA hits superimposed on the original image.
      markedAngles -> Numpy array. The image with TT striation angle color-coded on the original image.
    '''
    self.markedImage = markedImage
    self.markedAngles = markedAngles
    self.ttContent = ttContent
    self.ltContent = ltContent
    self.taContent = taContent
  
  def writeToCSV(self, inputs):
    '''This function writes the results to a CSV file whose name is specified in the Inputs class 
    (Inputs.outputParams['csvFile'])'''

    ### Check if the output csv file already exists
    fileExists = os.path.isfile(inputs.dic['outputParams']['csvFile'])

    ### If the file does not already exist, we need to create headers for it
    if not fileExists:
      with open(inputs.dic['outputParams']['csvFile'], 'wb') as csvFile:
        ## Create instance of writer object
        dummyWriter = csv.writer(csvFile)
      
        ## If the csv file did not already exists, we need to create headers for the output file
        header = [
          'Date of Classification',
          'Time of Classification',
          'Image Name',
          'TT Content',
          'LT Content',
          'TA Content',
          'Output Image Location and Root'
        ]
        dummyWriter.writerow(header)

    with open(inputs.dic['outputParams']['csvFile'], 'ab') as csvFile:
      ## Create instance of writer object
      dummyWriter = csv.writer(csvFile)
      
      ## Get Date and Time
      now = datetime.datetime.now()

      ## Write the outputs of this classification to the csv file
      output = [
        now.strftime('%Y-%m-%d'),
        now.strftime('%H:%M:%S'),
        inputs.imageName,
        self.ttContent,
        self.ltContent,
        self.taContent,
        inputs.dic['outputParams']['fileRoot']        
      ]

      ## Write outputs to csv file
      dummyWriter.writerow(output)



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
                 paramDict,
                 ttThresh=None,
                 ttGamma=None,
                 ):
  '''
  Takes inputs class that contains original image and performs WT filtering on the image
  '''
  print "TT Filtering"
  start = time.time()

  ### Specify necessary inputs
  ## Read in filter
  ttFilter = util.LoadFilter(inputs.paramDicts['TT']['filterName'])
  inputs.mfOrig = ttFilter

  paramDict['covarianceMatrix'] = np.ones_like(inputs.imgOrig)
  paramDict['mfPunishment'] = util.LoadFilter(inputs.paramDicts['TT']['punishFilterName'])

  ## Check to see if parameters are manually specified
  if ttThresh != None:
    paramDict['snrThresh'] = ttThresh
  if ttGamma != None:
    paramDict['gamma'] = ttGamma

  ### Perform filtering
  WTresults = bD.DetectFilter(
    inputs,
    paramDict,
    inputs.dic['iters'],
    returnAngles=inputs.dic['returnAngles']
  )  

  end = time.time()
  print "Time for WT filtering to complete:",end-start,"seconds"

  return WTresults

def LT_Filtering(inputs,
                 paramDict,
                 ):
  '''
  Takes inputs class that contains original image and performs LT filtering on the image
  '''

  print "LT filtering"
  start = time.time()

  ### Specify necessary inputs
  inputs.mfOrig = util.LoadFilter(inputs.paramDicts['LT']['filterName'])

  ### Perform filtering
  LTresults = bD.DetectFilter(
    inputs,
    paramDict,
    inputs.dic['iters'],
    returnAngles=inputs.dic['returnAngles']
  )

  end = time.time()
  print "Time for LT filtering to complete:",end-start,"seconds"

  return LTresults

def TA_Filtering(inputs,
                 paramDict,
                 iters=None,
                 ):
  '''
  Takes inputs class that contains original image and performs Loss filtering on the image
  '''
  print "TA filtering"
  start = time.time()

  ### Specify necessary inputs
  inputs.mfOrig = util.LoadFilter(inputs.paramDicts['TA']['filterName'])
  
  ## Check to see if iters (filter rotations are specified) if they aren't we'll just use 0 and 45
  ##   degrees since the loss filter is symmetric
  if iters != None:
    Lossiters = iters
  else:
    Lossiters = [0, 45] 
  
  ### Perform filtering
  Lossresults = bD.DetectFilter(
    inputs,
    paramDict,
    Lossiters,
    returnAngles=inputs.dic['returnAngles']
  )

  end = time.time()
  print "Time for TA filtering to complete:",end-start,"seconds"

  return Lossresults

def analyzeTT_Angles(testImageName,
                     inputs,
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
  #cImg = util.ReadImg(testImageName,cvtColor=False)

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
  smoothedWTresults = bD.DetectFilter(rotInputs,params,inputs.dic['iters'],returnAngles=True)
  smoothedHits = smoothedWTresults.stackedAngles

  ### pull out actual hits from smoothed results
  smoothedHits[WTstackedHits == 0] = -1

  coloredAngles = painter.colorAngles(inputs.colorImage,smoothedHits,inputs.dic['iters'])

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
          angleCounts.append(inputs.dic['iters'][rotArg])

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
      ):
  '''
  This function is the main workhorse for the detection of features in 2D myocytes.
    See give3DMarkedMyocyte() for better documentation.
    TODO: Better document this
  '''
 
  start = time.time()
  ### Create storage object for results
  myResults = ClassificationResults()

  ### Perform Filtering Routines
  ## Transverse Tubule Filtering
  if inputs.dic['filterTypes']['TT']:
    WTresults = TT_Filtering(
      inputs = inputs,
      paramDict = inputs.paramDicts['TT'],
    )
    WTstackedHits = WTresults.stackedHits
  else:
    #WTstackedHits = np.zeros_like(inputs.imgOrig)
    WTstackedHits = None

  ## Longitudinal Tubule Filtering
  if inputs.dic['filterTypes']['LT']:
    LTresults = LT_Filtering(
      inputs = inputs,
      paramDict = inputs.paramDicts['LT'],
    )
    LTstackedHits = LTresults.stackedHits
  else:
    # LTstackedHits = np.zeros_like(inputs.imgOrig)
    LTstackedHits = None

  ## Tubule Absence Filtering
  if inputs.dic['filterTypes']['TA']:
    Lossresults = TA_Filtering(
      inputs=inputs,
      paramDict = inputs.paramDicts['TA'],
    )
    LossstackedHits = Lossresults.stackedHits
  else:
    # LossstackedHits = np.zeros_like(inputs.imgOrig)
    LossstackedHits = None
 
  ## Marking superthreshold hits for loss filter
  if inputs.dic['filterTypes']['TA']:
    LossstackedHits[LossstackedHits != 0] = 255
    LossstackedHits = np.asarray(LossstackedHits, dtype='uint8')

    ## applying a loss mask to attenuate false positives from WT and Longitudinal filter
    if inputs.dic['filterTypes']['TT']:
      WTstackedHits[LossstackedHits == 255] = 0
    if inputs.dic['filterTypes']['LT']:
      LTstackedHits[LossstackedHits == 255] = 0

  ## marking superthreshold hits for longitudinal filter
  if inputs.dic['filterTypes']['LT']:
    LTstackedHits[LTstackedHits != 0] = 255
    LTstackedHits = np.asarray(LTstackedHits, dtype='uint8')

    ## masking WT response with LT mask so there is no overlap in the markings
    if inputs.dic['filterTypes']['TT']:
      WTstackedHits[LTstackedHits == 255] = 0

  ## marking superthreshold hits for WT filter
  if inputs.dic['filterTypes']['TT']:
    WTstackedHits[WTstackedHits != 0] = 255
    WTstackedHits = np.asarray(WTstackedHits, dtype='uint8')

  ## apply preprocessed masks
  if inputs.dic['filterTypes']['TT']:
    wtMasked = util.ReadResizeApplyMask(
      WTstackedHits,
      inputs.yamlDict['imageName'],
      ImgTwoSarcSize,
      filterTwoSarcSize=ImgTwoSarcSize
    )
  else:
    wtMasked = None
  if inputs.dic['filterTypes']['LT']:
    ltMasked = util.ReadResizeApplyMask(
      LTstackedHits,
      inputs.yamlDict['imageName'],
      ImgTwoSarcSize,
      filterTwoSarcSize=ImgTwoSarcSize
    )
  else:
    ltMasked = None
  if inputs.dic['filterTypes']['TA']:
    lossMasked = util.ReadResizeApplyMask(
      LossstackedHits,
      inputs.yamlDict['imageName'],
      ImgTwoSarcSize,
      filterTwoSarcSize=ImgTwoSarcSize
    )
  else:
    lossMasked = None

  ## Save the hits as full-resolution arrays for future use 
  if inputs.dic['outputParams']['saveHitsArray']:
    outDict = inputs.dic['outputParams']
    if inputs.dic['filterTypes']['TA']:
      np.save(outDict['fileRoot']+'_TA_hits', lossMasked)
    
    if inputs.dic['filterTypes']['LT']:
      np.save(outDict['fileRoot']+'_LT_hits', ltMasked)

    if inputs.dic['filterTypes']['TT']:
      np.save(outDict['fileRoot']+'_TT_hits', wtMasked)

  if not inputs.dic['returnPastedFilter']:
    ### create holders for marking channels
    markedImage = inputs.colorImage.copy()
    WTcopy = markedImage[:,:,0]
    LTcopy = markedImage[:,:,1]
    Losscopy = markedImage[:,:,2]

    ### color corrresponding channels
    if inputs.dic['filterTypes']['TT']:
      WTcopy[wtMasked > 0] = 255
    if inputs.dic['filterTypes']['LT']:
      LTcopy[ltMasked > 0] = 255
    if inputs.dic['filterTypes']['TA']:
      Losscopy[lossMasked > 0] = 255

    ### mark mask outline on myocyte
    myResults.markedImage = util.markMaskOnMyocyte(
      markedImage,
      inputs.yamlDict['imageName']
    )

    if isinstance(inputs.dic['outputParams']['fileRoot'], str):
      ### mark mask outline on myocyte
      cI_written = myResults.markedImage

      ### write output image
      plt.figure()
      plt.imshow(util.switchBRChannels(cI_written))
      outDict = inputs.dic['outputParams']
      plt.gcf().savefig(outDict['fileRoot']+"_output."+outDict['fileType'],dpi=outDict['dpi'])

  if inputs.dic['returnPastedFilter']:
    ## Mark filter-sized unit cells on the image to represent hits
    myResults.markedImage = util.markPastedFilters(inputs, lossMasked, ltMasked, wtMasked)
    
    ### apply mask again so as to avoid content > 1.0
    myResults.markedImage = util.ReadResizeApplyMask(
      myResults.markedImage,
      inputs.yamlDict['imageName'],
      ImgTwoSarcSize,
      filterTwoSarcSize=ImgTwoSarcSize
    )
    # inputs.colorImage[dummy==255] = 255

    ### Now based on the marked hits, we can obtain an estimate of tubule content
    estimatedContent = util.estimateTubuleContentFromColoredImage(myResults.markedImage)
  
    if isinstance(inputs.dic['outputParams']['fileRoot'], str):
      ### mark mask outline on myocyte
      cI_written = myResults.markedImage

      ### write outputs	  
      plt.figure()
      plt.imshow(util.switchBRChannels(cI_written))
      outDict = inputs.dic['outputParams']
      plt.gcf().savefig(outDict['fileRoot']+"_output."+outDict['fileType'],dpi=outDict['dpi'])

  if inputs.dic['returnAngles']:
    angleCounts, myResults.markedAngles = analyzeTT_Angles(
      testImageName=inputs.yamlDict['imageName'],
      inputs=inputs,
      ImgTwoSarcSize=ImgTwoSarcSize,
      WTstackedHits=WTstackedHits
    )

    if isinstance(inputs.dic['outputParams']['fileRoot'], str):
      plt.figure()
      plt.imshow(util.switchBRChannels(myResults.markedAngles))
      outDict = inputs.dic['outputParams']
      plt.gcf().savefig(outDict['fileRoot']+"_angles_output."+outDict['fileType'],dpi=outDict['dpi'])
    
    end = time.time()
    tElapsed = end - start
    print "Total Elapsed Time: {}s".format(tElapsed)
    return myResults.markedImage, myResults.markedAngles, angleCounts

  ### Write results of the classification
  myResults.writeToCSV(inputs=inputs)

  end = time.time()
  tElapsed = end - start
  print "Total Elapsed Time: {}s".format(tElapsed)
  return myResults.markedImage 

def give3DMarkedMyocyte(
      inputs,
      ImgTwoSarcSize=None,
      returnAngles=False,
      returnPastedFilter=True,
      ):
  '''
  This function is for the detection and marking of subcellular features in three dimensions. 

  Inputs:
    testImage -> str. Name of the image to be analyzed. NOTE: This image has previously been preprocessed by 
                   XXX routine.
    scopeResolutions -> list of values (ints or floats). List of resolutions of the confocal microscope for x, y, and z.
    # tag -> str. Base name of the written files 
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

  ### Insantiate storage object for results
  myResults = ClassificationResults()

  ### Transverse Tubule Filtering
  if inputs.dic['filterTypes']['TT']:
    TTresults = TT_Filtering(
      inputs,
      paramDict = inputs.paramDicts['TT'],
      # returnAngles = returnAngles
    )
    TTstackedHits = TTresults.stackedHits
  else:
    TTstackedHits = np.zeros_like(inputs.imgOrig)

  ### Longitudinal Tubule Filtering
  if inputs.dic['filterTypes']['LT']:
    LTresults = LT_Filtering(
      inputs,
      paramDict = inputs.paramDicts['LT'],
      # returnAngles = returnAngles
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
      # returnAngles = returnAngles
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
  inputs.colorImage = cImg
  if returnPastedFilter:
    ## Use routine to mark unit cell sized cuboids around detections
    myResults.markedImage = util.markPastedFilters(inputs,
                             TAstackedHits,
                             LTstackedHits,
                             TTstackedHits,
                             ttName = inputs.paramDicts['TT']['filterName'],
                             ltName = inputs.paramDicts['LT']['filterName'],
                             taName = inputs.paramDicts['TA']['filterName']
                             )

    ### 'Measure' cell volume just by getting measure of containing array
    cellVolume = np.float(np.product(inputs.imgOrig.shape))

    ### Now based on the marked hits, we can obtain an estimate of tubule content
    estimatedContent = util.estimateTubuleContentFromColoredImage(
      myResults.markedImage,
      totalCellSpace=cellVolume,
      taFilterName=inputs.paramDicts['TA']['filterName'],
      ltFilterName=inputs.paramDicts['LT']['filterName'],
      ttFilterName=inputs.paramDicts['TT']['filterName']
    )

  else:
    ## Just mark exactly where detection is instead of pasting unit cells on detections
    myResults.markedImage = inputs.colorImage.copy()

    myResults.markedImage[:,:,:,2][TAstackedHits > 0] = 255
    myResults.markedImage[:,:,:,1][LTstackedHits > 0] = 255
    myResults.markedImage[:,:,:,0][TTstackedHits > 0] = 255

    ### Determine percentages of volume represented by each filter
    cellVolume = np.float(np.product(inputs.imgOrig.shape))
    myResults.taContent = np.float(np.sum(myResults.markedImage[:,:,:,2] == 255)) / cellVolume
    myResults.ltContent = np.float(np.sum(myResults.markedImage[:,:,:,1] == 255)) / cellVolume
    myResults.ttContent = np.float(np.sum(myResults.markedImage[:,:,:,0] == 255)) / cellVolume

    print "TA Content per Cell Volume:", myResults.taContent
    print "LT Content per Cell Volume:", myResults.ltContent
    print "TT Content per Cell Volume:", myResults.ttContent

  if returnAngles:
    print "WARNING: Striation angle analysis is not yet available in 3D"
  
  ### Save detection image
  if isinstance(inputs.dic['outputParams']['fileRoot'], str):
    util.Save3DImg(myResults.markedImage,inputs.dic['outputParams']['fileRoot']+'.'+inputs.dic['outputParams']['fileType'],switchChannels=True)

  ### Write results of the classification
  myResults.writeToCSV(inputs=inputs)

  end = time.time()
  print "Time for algorithm to run:",end-start,"seconds"
  
  return myResults.markedImage

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
    inputs=inputs
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

  ### Define parameters for the simulation of the cell
  ## Probability of finding a longitudinal tubule unit cell
  ltProb = 0.3
  ## Probability of finding a tubule absence unit cell
  taProb = 0.3
  ## Amplitude of the Guassian White Noise
  noiseAmplitude = 0.
  ## Define scope resolutions for generating the filters and the cell. This is in x, y, and z resolutions
  scopeResolutions = [10,10,5] #[vx / um]
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
                               scopeResolutions=scopeResolutions,
                               cellDimensions=cellDimensions,
                               fileName=testName,
                               seed=1001,
                               )

  ### Setup input parameters for classification
  inputs = Inputs(
    yamlFileName = yamlFile
  )

  ### Analyze the 3D cell
  markedImage = give3DMarkedMyocyte(
    inputs = inputs
  )

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
