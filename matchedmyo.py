#!/usr/bin/env python2

'''This script will contain all of the necessary wrapper routines to perform analysis on a wide range of 
cardiomyocyte/tissue images.

This is THE script that all sophisticated, general user-level routines will be routed through.
'''

import time
import util
import numpy as np
import optimizer
import bankDetect as bD
import matplotlib.pyplot as plt
import painter
import matchedFilter as mF
import argparse



### Make an empty class instance
class empty:
  pass

###################################################################################################
###################################################################################################
###################################################################################################
###
### Individual Filtering Routines
###
###################################################################################################
###################################################################################################
###################################################################################################

def WT_Filtering(inputs,
                 iters,
                 ttFilterName,
                 ttPunishFilterName,
                 ttThresh=None,
                 ttGamma=None,
                 returnAngles=False):
  '''
  Takes inputs class that contains original image and performs WT filtering on the image
  '''
  print "TT Filtering"
  start = time.time()

  ### Specify necessary inputs
  ttFilter = util.LoadFilter(ttFilterName)
  inputs.mfOrig = ttFilter
  if len(np.shape(inputs.imgOrig)) == 3:
    WTparams = optimizer.ParamDict(typeDict='WT3D')
  elif len(np.shape(inputs.imgOrig)) == 2:
    WTparams = optimizer.ParamDict(typeDict='WT')
  else:
    raise RuntimeError("The number of dimensions of the image stored in inputs.imgOrig is {}. \
                       This is not supported currently.".format(len(np.shape(inputs.imgOrig))))
  WTparams['covarianceMatrix'] = np.ones_like(inputs.imgOrig)
  WTparams['mfPunishment'] = util.LoadFilter(ttPunishFilterName)
  WTparams['useGPU'] = inputs.useGPU
  if ttThresh != None:
    WTparams['snrThresh'] = ttThresh
  if ttGamma != None:
    WTparams['gamma'] = ttGamma

  ### Perform filtering
  WTresults = bD.DetectFilter(inputs,WTparams,iters,returnAngles=returnAngles)  

  end = time.time()
  print "Time for WT filtering to complete:",end-start,"seconds"

  return WTresults

def LT_Filtering(inputs,
                 iters,
                 ltFilterName,
                 ltThresh=None,
                 ltStdThresh=None,
                 returnAngles=False
                 ):
  '''
  Takes inputs class that contains original image and performs LT filtering on the image
  '''

  print "LT filtering"
  start = time.time()

  ### Specify necessary inputs
  inputs.mfOrig = util.LoadFilter(ltFilterName)
  if len(np.shape(inputs.imgOrig)) == 3:
    LTparams = optimizer.ParamDict(typeDict='LT3D')
  elif len(np.shape(inputs.imgOrig)) == 2:
    LTparams = optimizer.ParamDict(typeDict='LT')
  else:
    raise RuntimeError("The number of dimensions of the image stored in inputs.imgOrig is {}. \
                       This is not supported currently.".format(len(np.shape(inputs.imgOrig))))
  if ltThresh != None:
    LTparams['snrThresh'] = ltThresh
  if ltStdThresh != None:
    LTparams['stdDevThresh'] = ltStdThresh
  LTparams['useGPU'] = inputs.useGPU

  ### Perform filtering
  LTresults = bD.DetectFilter(inputs,LTparams,iters,returnAngles=returnAngles)

  end = time.time()
  print "Time for LT filtering to complete:",end-start,"seconds"

  return LTresults

def Loss_Filtering(inputs,
                   lossFilterName,
                   iters=None,
                   lossThresh=None,
                   lossStdThresh=None,
                   returnAngles=False,
                   ):
  '''
  Takes inputs class that contains original image and performs Loss filtering on the image
  '''
  print "TA filtering"
  start = time.time()

  ### Specify necessary inputs
  inputs.mfOrig = util.LoadFilter(lossFilterName)
  if len(np.shape(inputs.imgOrig)) == 3:
    Lossparams = optimizer.ParamDict(typeDict='Loss3D')
  elif len(np.shape(inputs.imgOrig)) == 2:
    Lossparams = optimizer.ParamDict(typeDict='Loss')
  else:
    raise RuntimeError("The number of dimensions of the image stored in inputs.imgOrig is {}. \
                       This is not supported currently.".format(len(np.shape(inputs.imgOrig))))
  Lossparams['useGPU'] = inputs.useGPU
  if iters != None:
    Lossiters = iters
  else:
    Lossiters = [0, 45] # don't need many rotations for loss filtering
  if lossThresh != None:
    Lossparams['snrThresh'] = lossThresh
  if lossStdThresh != None:
    Lossparams['stdDevThresh'] = lossStdThresh

  ### Perform filtering
  Lossresults = bD.DetectFilter(inputs,Lossparams,Lossiters,returnAngles=returnAngles)

  end = time.time()
  print "Time for TA filtering to complete:",end-start,"seconds"

  return Lossresults

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
      ttFilterName="./myoimages/newSimpleWTFilter.png",
      ltFilterName="./myoimages/LongitudinalFilter.png",
      lossFilterName="./myoimages/LossFilter.png",
      wtPunishFilterName="./myoimages/newSimpleWTPunishmentFilter.png",
      ltPunishFilterName="./myoimages/newLTPunishmentFilter.png",
      testImage="./myoimages/MI_D_73_annotation.png",
      ImgTwoSarcSize=None,
      tag = "default_",
      writeImage = False,
      ttThresh=None,
      ltThresh=None,
      lossThresh=None,
      wtGamma=None,
      ltStdThresh=None,
      lossStdThresh=None,
      iters=[-25,-20,-15,-10,-5,0,5,10,15,20,25],
      returnAngles=False,
      returnPastedFilter=True,
      useGPU=False,
      fileExtension=".pdf"
      ):
  '''
  This function is the main workhorse for the detection of features in 2D myocytes.
    See give3DMarkedMyocyte() for better documentation.
    TODO: Better document this
  '''
 
  start = time.time()

  ### Read in preprocessed image
  img = util.ReadImg(testImage,renorm=True)

  ### defining inputs to be read by DetectFilter function
  inputs = empty()
  inputs.imgOrig = util.ReadResizeApplyMask(img,testImage,25,25) # just applies mask
  inputs.useGPU = useGPU

  ### WT filtering
  if ttFilterName != None:
    WTresults = WT_Filtering(
      inputs = inputs,
      iters = iters,
      ttFilterName = ttFilterName,
      ttPunishFilterName = wtPunishFilterName,
      ttThresh = ttThresh,
      ttGamma = wtGamma,
      returnAngles = returnAngles
    )
    WTstackedHits = WTresults.stackedHits
  else:
    WTstackedHits = np.zeros_like(inputs.imgOrig)

  ### LT filtering
  if ltFilterName != None:
    LTresults = LT_Filtering(
      inputs = inputs,
      iters = iters,
      ltFilterName = ltFilterName,
      ltThresh = ltThresh,
      ltStdThresh = ltStdThresh,
      returnAngles = returnAngles
    )
    LTstackedHits = LTresults.stackedHits
  else:
    LTstackedHits = np.zeros_like(inputs.imgOrig)

  ### Loss filtering
  if lossFilterName != None:
    Lossresults = Loss_Filtering(
      inputs=inputs,
      lossFilterName = lossFilterName,
      lossThresh = lossThresh,
      lossStdThresh = lossStdThresh,
      returnAngles = returnAngles
    )
    LossstackedHits = Lossresults.stackedHits
  else:
    LossstackedHits = np.zeros_like(inputs.imgOrig)
 
  ### Read in colored image for marking hits
  cI = util.ReadImg(testImage,cvtColor=False).astype(np.float64)
  # killing the brightness a bit so looking at results isn't like staring at the sun
  cI *= 0.85
  cI = np.round(cI).astype(np.uint8)

  ### Marking superthreshold hits for loss filter
  LossstackedHits[LossstackedHits != 0] = 255
  LossstackedHits = np.asarray(LossstackedHits, dtype='uint8')

  ### applying a loss mask to attenuate false positives from WT and Longitudinal filter
  WTstackedHits[LossstackedHits == 255] = 0
  LTstackedHits[LossstackedHits == 255] = 0

  ### marking superthreshold hits for longitudinal filter
  LTstackedHits[LTstackedHits != 0] = 255
  LTstackedHits = np.asarray(LTstackedHits, dtype='uint8')

  ### masking WT response with LT mask so there is no overlap in the markings
  WTstackedHits[LTstackedHits == 255] = 0

  ### marking superthreshold hits for WT filter
  WTstackedHits[WTstackedHits != 0] = 255
  WTstackedHits = np.asarray(WTstackedHits, dtype='uint8')

  ### apply preprocessed masks
  wtMasked = util.ReadResizeApplyMask(WTstackedHits,testImage,ImgTwoSarcSize,
                                 filterTwoSarcSize=ImgTwoSarcSize)
  ltMasked = util.ReadResizeApplyMask(LTstackedHits,testImage,ImgTwoSarcSize,
                                 filterTwoSarcSize=ImgTwoSarcSize)
  lossMasked = util.ReadResizeApplyMask(LossstackedHits,testImage,ImgTwoSarcSize,
                                   filterTwoSarcSize=ImgTwoSarcSize)

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
    cI = util.markMaskOnMyocyte(cI,testImage)
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
    dummy = util.ReadResizeApplyMask(dummy,testImage,ImgTwoSarcSize,filterTwoSarcSize=ImgTwoSarcSize)
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
    cImg = util.ReadImg(testImage,cvtColor=False)

    ### perform smoothing on the original image
    dim = 5
    kernel = np.ones((dim,dim),dtype=np.float32)
    kernel /= np.sum(kernel)
    smoothed = mF.matchedFilter(inputs.imgOrig,kernel,demean=False)

    ### make longer WT filter so more robust to striation angle deviation
    ttFilter = util.LoadFilter(ttFilterName)
    longFilter = np.concatenate((ttFilter,ttFilter,ttFilter))
    
    rotInputs = empty()
    rotInputs.imgOrig = smoothed
    rotInputs.mfOrig = longFilter

    params = optimizer.ParamDict(typeDict='WT')
    params['snrThresh'] = 0 # to pull out max hit
    params['filterMode'] = 'simple' # we want no punishment since that causes high variation
    
    ### perform simple filtering
    smoothedWTresults = bD.DetectFilter(rotInputs,params,iters,returnAngles=returnAngles)
    smoothedHits = smoothedWTresults.stackedAngles

    ### pull out actual hits from smoothed results
    smoothedHits[WTstackedHits == 0] = -1

    coloredAngles = painter.colorAngles(cImg,smoothedHits,iters)

    coloredAnglesMasked = util.ReadResizeApplyMask(coloredAngles,testImage,
                                              ImgTwoSarcSize,
                                              filterTwoSarcSize=ImgTwoSarcSize)
    stackedAngles = smoothedHits
    dims = np.shape(stackedAngles)
    angleCounts = []
    for i in range(dims[0]):
      for j in range(dims[1]):
        rotArg = stackedAngles[i,j]
        if rotArg != -1:
          ### indicates this is a hit
          angleCounts.append(iters[rotArg])

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
      testImage,
      scopeResolutions,
      ttFilterName=None,
      ltFilterName=None,
      taFilterName=None,
      ttPunishFilterName=None,
      ltPunishFilterName=None,
      ttThresh=None,
      ltThresh=None,
      lossThresh=None,
      wtGamma=None,
      ltStdThresh=None,
      lossStdThresh=None,
      ImgTwoSarcSize=None,
      tag = None,
      xiters=[-10,0,10],
      yiters=[-10,0,10],
      ziters=[-10,0,10],
      returnAngles=False,
      returnPastedFilter=True
      ):
  '''
  This function is for the detection and marking of subcellular features in three dimensions. 

  Inputs:
    testImage -> str. Name of the image to be analyzed. NOTE: This image has previously been preprocessed by 
                   XXX routine.
    scopeResolutions -> list of values (ints or floats). List of resolutions of the confocal microscope for x, y, and z.
    ttFilterName -> str. Name of the transverse tubule filter to be used
    ltFiltername -> str. Name of the longitudinal filter to be used
    lossFilterName -> str. Name of the tubule absence filter to be used
    ttPunishFilterName -> str. Name of the transverse tubule punishment filter to be used
    ltPunishFilterName -> str. Name of the longitudinal tubule punishment filter to be used NOTE: Delete?
    
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
  inputs = empty()
  inputs.imgOrig = util.ReadImg(testImage,renorm=True)
  inputs.useGPU = False
  inputs.scopeResolutions = scopeResolutions

  ### Form flattened iteration matrix containing all possible rotation combinations
  flattenedIters = []
  for i in xiters:
    for j in yiters:
      for k in ziters:
        flattenedIters.append( [i,j,k] )

  ### WT filtering
  if ttFilterName != None:
    TTresults = WT_Filtering(inputs,flattenedIters,ttFilterName,ttPunishFilterName,ttThresh,wtGamma,returnAngles)
    TTstackedHits = TTresults.stackedHits
  else:
    TTstackedHits = np.zeros_like(inputs.imgOrig)

  ### LT filtering
  if ltFilterName != None:
    LTresults = LT_Filtering(inputs,flattenedIters,ltFilterName,ltThresh,ltStdThresh,returnAngles)
    LTstackedHits = LTresults.stackedHits
  else:
    LTstackedHits = np.zeros_like(inputs.imgOrig)

  ### Loss filtering
  if taFilterName != None:
    ## form tubule absence flattened rotation matrix. Choosing to look at tubule absence at one rotation right now.
    taIters = [[0,0,0]]
    TAresults = Loss_Filtering(inputs,
                               taFilterName,
                               iters=taIters,
                               lossThresh=lossThresh,
                               lossStdThresh=lossStdThresh,
                               returnAngles=returnAngles)
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
                             wtName = ttFilterName,
                             ltName = ltFilterName,
                             lossName = taFilterName)

    ### 'Measure' cell volume just by getting measure of containing array
    cellVolume = np.float(np.product(inputs.imgOrig.shape))

    ### Now based on the marked hits, we can obtain an estimate of tubule content
    estimatedContent = util.estimateTubuleContentFromColoredImage(
      cImg,
      totalCellSpace=cellVolume,
      taFilterName = taFilterName,
      ltFilterName = ltFilterName,
      ttFilterName = ttFilterName
    )

  else:
    ## Just mark exactly where detection is instead of pasting until cells on detections
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

###################################################################################################
###################################################################################################
###################################################################################################
###
###  Validation Routines
###
###################################################################################################
###################################################################################################
###################################################################################################

def fullValidation():
  '''This routine wraps all of the written validation routines.
  This should be run EVERYTIME before changes are committed and pushed to the repository.
  '''

  validate()
  validate3D()

def validate(testImage="./myoimages/MI_M_45_processed.png",
             display=False
             ):
  '''This function serves as a validation routine for the 2D functionality of this repo.
  
  Inputs:
    testImage -> str. File path pointing to the image to be validated
    display -> Bool. If True, display the marked image
  '''
  
  # run algorithm
  markedImg = giveMarkedMyocyte(testImage=testImage)

  if display:
    plt.figure()
    plt.imshow(markedImg)
    plt.show()

  print "\nThe following content values are for validation purposes only.\n"

  # calculate wt, lt, and loss content  
  ttContent, ltContent, taContent = util.assessContent(markedImg)

  assert(abs(ttContent - 103050) < 1), "TT validation failed."
  assert(abs(ltContent -  68068) < 1), "LT validation failed."
  assert(abs(taContent - 156039) < 1), "TA validation failed."
  print "\nPASSED!\n"

def validate3D():
  '''This function serves as a validation routine for the 3D functionality of this repo.

  Inputs:
    None
  '''
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
  ttName = './myoimages/TT_3D.tif'
  ttPunishName = './myoimages/TT_Punishment_3D.tif'
  ltName = './myoimages/LT_3D.tif'
  taName = './myoimages/TA_3D.tif'

  ### Simulate the small 3D cell
  util.generateSimulated3DCell(LT_probability=ltProb,
                               TA_probability=taProb,
                               noiseAmplitude=noiseAmplitude,
                               scopeResolutions=scopeResolutions,
                               cellDimensions=cellDimensions,
                               fileName=testName,
                               seed=1001,
                               )

  ### Analyze the 3D cell
  markedImage = give3DMarkedMyocyte(testImage=testName,
                                    scopeResolutions = scopeResolutions,
                                    ttFilterName = ttName,
                                    ttPunishFilterName = ttPunishName,
                                    ltFilterName = ltName,
                                    taFilterName = taName,
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

def main(args):
  '''The routine through which all command line functionality is routed.
  '''
  ### Get a list of all function names in the script
  functions = globals()

  functions[args.functionToCall]()#(args)


### Begin argument parser for command line functionality  
description = '''This is the main script for the analysis of 2D and 3D confocal images of 
cardiomyocytes and cardiac tissues.
'''
parser = argparse.ArgumentParser(description=description)
parser.add_argument('functionToCall', 
                    type=str,
                    help='The function to call within this script')
args = parser.parse_args()
main(args)



#
# MAIN routine executed when launching this script from command line 
#
# tag = "default_" 
# if __name__ == "__main__":
#   msg = helpmsg()
#   remap = "none"

#   if len(sys.argv) < 2:
#       raise RuntimeError(msg)

#   # Loops over each argument in the command line 
#   for i,arg in enumerate(sys.argv):

#     ### Validation Routines
#     if(arg=="-validate"):
#       validate()
#       quit()

#     if(arg=='-validate3D'):
#       validate3D()
#       quit()

#     if(arg=='-fullValidation'):
#       ## This routine MUST be run before changes are committed to the repo
#       validate()
#       validate3D()
#       quit()