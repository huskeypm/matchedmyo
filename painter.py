from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.mlab as mlab
import cv2
from scipy.misc import toimage
from scipy.ndimage.filters import *
from scipy import ndimage
import matchedFilter as mF
import imutils
from imtools import *
from matplotlib import cm
import detection_protocols as dps
from scipy import signal
import util 
import util2

class empty:pass

##
## Performs matched filtering over desired angles
##
def correlateThresher(
        inputs,
        params,
        iters = [0,30,60,90],  
        printer = True, 
        filterMode=None,
        label="undef",
        ):

    # TODO - this should be done in preprocessing, not here
    #print "PKH: turn into separate proproccessing routine"
    img = inputs.imgOrig
    
    if params['doCLAHE']:
      if img.dtype != 'uint8':
        myImg = np.array((img * 255),dtype='uint8')
      else:
        myImg = img
      clahe99 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
      img = clahe99.apply(myImg)

    #filterRef = util.renorm(np.array(inputs.mfOrig,dtype=float),scale=1.)
    filterRef = inputs.mfOrig.copy()
    
    ##
    ## Iterate over all filter rotations desired 
    # Store all 'hits' at each angle 
    correlated = []

    inputs.demeanedImg = np.abs(np.subtract(inputs.imgOrig, np.mean(inputs.imgOrig)))
    for i in iters:
      result = empty()

      # Check dimensionality of iteration and rotate filters accordingly
      if type(i) == list:
        # This is 3D
        # pad/rotate filter
        rFN = util.rotate3DArray_Nonhomogeneous(filterRef,i,inputs.scopeResolutions)
        # Depad the array to reduce computational expense
        rFN = util.autoDepadArray(rFN)
        inputs.mf = rFN

        # check to see if we need to rotate other matched filters for the detection
        if params['filterMode'] == 'punishmentFilter':
          params['mfPunishmentRot'] = util.rotate3DArray_Nonhomogeneous(params['mfPunishment'].copy(),i,inputs.scopeResolutions)
          params['mfPunishmentRot'] = util.autoDepadArray(params['mfPunishmentRot'])
      else:
        # This is 2D
        # pad/rotate 
        params['angle'] = i
        rFN = util.PadRotate(filterRef,i)  
        # Depad the array to reduce computational expense
        rFN = util.autoDepadArray(rFN)
        inputs.mf = rFN  

        # check for other matched filters
        if params['filterMode'] == 'punishmentFilter':
          params['mfPunishmentRot'] = util.PadRotate(params['mfPunishment'].copy(),i)
      
      # matched filtering 
      result = dps.FilterSingle(inputs,params)      

      # store. Results contain both correlation plane and snr
      result.rFN = np.copy(rFN)
      correlated.append(result) 

    ##
    ## write
    ##
    #if label!=None and printer:
    if printer: 
       if label==None:
         label="undef"
       if filterMode==None:
         filterMode="undef"

       for i, angle in enumerate(iters):
        tag = filterMode 
        daTitle = "rot %4.1f "%angle # + "hit %4.1f "%hit 

        result = correlated[i]   
        plt.figure()
        plt.subplot(1,2,1)
        plt.title("Rotated filter") 
        plt.imshow(result.rFN,cmap='gray')
        plt.subplot(1,2,2)
        plt.title("Correlation plane") 
        plt.imshow(result.corr)                
        plt.colorbar()
        plt.tight_layout()
        fileName = label+"_"+tag+'_{}.png'.format(angle)
        plt.gcf().savefig(fileName,dpi=300)
        plt.close()

    return correlated  # a list of objects that contain SNR, etc 

def CalcSNR(signalResponse,sigma_n=1):
  print "PHASE ME OUT" 
  return signalResponse/sigma_n

##
## Collects all hits above some criterion for a given angle and 'stacks' them
## into a single image
##
def StackHits(correlated,  # an array of 'correlation planes'
              paramDict, # threshold,
              iters,
              display=False,
              rescaleCorr=False,
              doKMeans=False, #True,
              returnAngles=False):
     # TODO
    if rescaleCorr:
      raise RuntimeError("Why is this needed? IGNORING")
    if display:
      print "Call me debug" 
    # Function that iterates through correlations at varying rotations of a single filter,
    # constructs a mask consisting of 'NaNs' and returns a list of these masked correlations


    ##
    ## select hits based on those entries about the snrThresh 
    ##
    maskList = []
    simpleCorrMaskList = []
    for i, iteration in enumerate(iters):
        # routine for identifying 'unique' hits
        try:
          daMask = util2.makeMask(paramDict['snrThresh'],img = correlated[i].snr,
                                  doKMeans=doKMeans, inverseThresh=paramDict['inverseSNR'])
        except:
          print "DC: Using workaround for tissue param dictionary. Fix me."
          daMask = util2.makeMask(paramDict['snrThresh'], img=correlated[i].snr,
                                  doKMeans=doKMeans)
        # pull out where there is a hit on the simple correlation for use in rotation angle
        hitMask = daMask > 0.
        simpleCorrMask = correlated[i].corr
        simpleCorrMask[np.logical_not(hitMask)] = 0
                                  
        #  print debugging info                                    
        if display:
            plt.figure()
            plt.subplot(2,1,1)
            plt.imshow(correlated[i].snr)
            plt.colorbar()
            plt.subplot(2,1,2)
            plt.imshow(daMask,cmap="gray")
            #plt.axis("off")
            plt.gcf().savefig("stack_%d.png"%iteration)    

        maskList.append(daMask)
        simpleCorrMaskList.append(simpleCorrMask)

    # take maximum hit
    stacked = np.max(maskList,axis=0)
    
    # default behavior (just returned stacked images) 
    if not returnAngles:
      return stacked 
    
    # function that returns angle associated with optimal response
    if returnAngles:
      # have to subtract one from array so that we can use this to index rotations later on in alg
      doSimple = False
      if doSimple:
        stackedAngles = np.argmax(simpleCorrMaskList,axis=0)
        # mask out non hits
        stackedAngles[stacked == 0] = -1
      else:
        stackedAngles = np.argmax(maskList,axis=0)
        stackedAngles[stacked == 0] = -1
      return stacked, stackedAngles
  
###
### Function to color the angles previously returned in StackHits
###   
def colorAngles(rawOrig, stackedAngles,iters,leftChannel='red',rightChannel='blue'):
  channelDict = {'blue':0, 'green':1, 'red':2}

  dims = np.shape(stackedAngles)

  if len(np.shape(rawOrig)) > 2:
    coloredImg = rawOrig.copy()
    #plt.figure()
    #plt.imshow(coloredImg)
    #plt.show()
    #quit()
  else:
    # we need to make an RGB version of the image
    coloredImg = np.zeros((dims[0],dims[1],3),dtype='uint8')
    scale = 0.75
    coloredImg[:,:,0] = scale * rawOrig
    coloredImg[:,:,1] = scale * rawOrig
    coloredImg[:,:,2] = scale * rawOrig

  leftmostIdx = 0 # mark as left channel
  rightmostIdx = len(iters) # mark as right channel
  # any idx between these two will be colored a blend of the two channels

  spacing = 255 / rightmostIdx

  for i in range(dims[0]):
    for j in range(dims[1]):
      rotArg = stackedAngles[i,j]
      if rotArg != -1:
        coloredImg[i,j,channelDict[leftChannel]] = int(255 - rotArg*spacing)
        coloredImg[i,j,channelDict[rightChannel]] = int(rotArg*spacing)
  return coloredImg

def paintME(myImg, myFilter1,  threshold = 190, cropper=[24,129,24,129],iters = [0,30,60,90], fused =True):
  correlateThresher(myImg, myFilter1,  threshold, cropper,iters, fused, False)
  for i, val in enumerate(iters):
 
    if fused:
      palette = cm.gray
      palette.set_bad('m', 1.0)
      placer = ReadImg('fusedCorrelated_{}.png'.format(val))
    else:
      palette = cm.gray
      palette.set_bad('b', 1.0)
      placer = ReadImg('bulkCorrelated_{}.png'.format(val))
    plt.figure()

    #print "num maxes", np.shape(np.argwhere(placer>threshold))
    Zm = np.ma.masked_where(placer > threshold, placer)
    fig, ax1 = plt.subplots()
    plt.axis("off")
    im = ax1.pcolormesh(Zm, cmap=palette)
    plt.title('Correlated_Angle_{}'.format(val))
    plt.savefig('falseColor_{}.png'.format(val))
    plt.axis('equal')
    plt.close()
    
                

# Basically just finds a 'unit cell' sized area around each detection 
# for the purpose of interpolating the data 
def doLabel(result,cellDimensions = [10, None, None],thresh=0):
  '''
  Finds a unit cell sized area around each detection. This serves to display more intuitive representations 
    of detections. This is for 2 and 3 dimensional data.
  
  Inputs:
    result -> class. Result class from bankDetect.DetectFilter(). result.stackedHits is where the detetions are stored.
    cellDimensions -> list of ints. Measurement of the unit cell dimensions in the x, y, and z directions.
                   Specification of y and z is optional but should be specified for accurate results.
                   Specification of z only needed if image is 3D.
    thresh -> int or float. Threshold applied to result.stackedHits to determine where the hits are.

  Outputs:
    labeled -> boolean array. The image with detections dilated according to unit cell size.
  '''
  if len(result.stackedHits.shape) == 3:
    dx, dy, dz = cellDimensions
  else:
    dx, dy = cellDimensions

  ### Determine dimensions of unit cell if none are specified. If dy and dz are not specified, set equal to dx
  if dy == None:
    dy = dx
  if len(result.stackedHits.shape) == 3:
    if dz == None:
      dz = dx
  
  ### Determine where hits are present in stackedHits based on threshold
  img =result.stackedHits > thresh

  ### Construct kernel the size of the unit cell
  if len(result.stackedHits.shape) == 3:
    kernel = np.ones((dx,dy,dz),np.float32)/(float(dy*dx*dz))
  else:
    kernel = np.ones((dx,dy),np.float32)/(float(dy*dx))
  
  ### Perform convolution to determine where kernel overlaps with detection
  if len(result.stackedHits.shape) == 3:
    filtered = ndimage.convolve(img.astype(float), kernel) / np.sum(kernel)
  else:
    filtered = signal.convolve2d(img, kernel, mode='same') / np.sum(kernel)

  ### Perform boolean operation to pull out all dilated hits
  labeled = filtered > 0
  
  return labeled

def WT_SNR(Img, WTfilter, WTPunishmentFilter,C,gamma):
  # calculates SNR of WT filter
  
  # get two responses
  h = mF.matchedFilter(Img, WTfilter, demean=False)
  h_star = mF.matchedFilter(Img,WTPunishmentFilter,demean=False)
  
  # calculate SNR
  SNR = h / (C + gamma * h_star)

  return SNR
