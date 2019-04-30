from __future__ import print_function
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import cv2
from scipy.misc import toimage
from scipy import ndimage
from scipy import signal
import matchedFilter as mF
import imutils
import detection_protocols as dps
import util


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
        efficientRotationStorage = False
        ):
    '''This function iterates through the list of iters (rotation angles) and calls routines to 
    perform the convolution at each of these filter rotations.

    Inputs:
      inputs -> class. Generated by matchedmyo.giveMarkedMyocyte() or matchedmyo.give3DMarkedMyocyte()
      params -> dict. Parameter dictionary generated by optimizer.ParamDict() function
      iters -> list of ints. List of rotation angles with which we would like to perform convolution at
      printer -> Bool. If True, save the correlation plane for each rotation.
      filterMode -> str. Type of filtering being done. This is for display purposes.
      label -> str. String denoting the label we would like to give the plots
      efficientRotationStorage -> Bool. If true, we do not save the correlation plane for each rotation.
                                    This SIGNIFICANTLY reduces the RAM requirement for processing large
                                    images or large amounts of rotation at the cost of forgetting each
                                    rotation's correlation plane.
    
    Outputs:
      if efficientRotationStorage:
        correlated -> dict. Dictionary containing two entries:
                       'maxSNRArray' -> an array containing the maximum SNR calculated for that 
                          specific pixel/voxel.
                       'rotMaxSNRArray' -> an array containing the rotation (int or list) at which the 
                          certain pixel/voxel experienced the maximum SNR.
      else:
        correlated -> list. List of results class at each rotation that contains SNR and correlation plane
    '''
    img = inputs.imgOrig
    
    ### Save the unrotated filter as a reference
    filterRef = inputs.mfOrig.copy()
    
    ### Form storage arrays/lists needed for the analysis
    if efficientRotationStorage: 
      correlated = {}

      ## We need to create storage array for SNR at each pixel/voxel
      if params['inverseSNR']:
        ## If we use inverse SNR (meaning the SNR must be below a certain value to signify a hit), 
        ##   we need to create an array that has arbitrarily high starting numbers
        correlated['SNRArray'] = np.ones_like(img) * 5. * params['snrThresh']
      else:
        ## Otherwise, we can just instantiate the SNRArray with zeros
        correlated['SNRArray'] = np.zeros_like(img)

      ## We also need to create a storage array for the rotation which the maximum SNR occurred.
      if len(np.shape(img)) == 3:
        ## This means that the image is 3D and we must store the rotation arrays in 3 different arrays
        ##   We use 361 because no one in their right mind would want to rotate a filter >= 360 degrees
        correlated['rotSNRArray'] = {}
        correlated['rotSNRArray']['x'] = np.ones_like(img, dtype=np.int16) * 361
        correlated['rotSNRArray']['y'] = np.ones_like(img, dtype=np.int16) * 361
        correlated['rotSNRArray']['z'] = np.ones_like(img, dtype=np.int16) * 361
      else:
        ## Otherwise we only need one array to store the single rotation
        correlated['rotSNRArray'] = np.ones_like(img,dtype=np.int16) * 361
    else:
      ## Store all 'hits' at each angle 
      correlated = []

    ### Iterate over all filter rotations desired
    for rotNum,i in enumerate(iters):
      ## Print progress of filtering if designated so
      if inputs.dic['displayProgress']:
        print ("\tPerforming classification for rotation {}/{}.".format(rotNum + 1,len(iters)))

      ## Check dimensionality of iteration and rotate filters accordingly
      if type(i) == list:
        ## This is 3D
        ## pad/rotate filter
        rFN = util.rotate3DArray_Nonhomogeneous(filterRef,i,inputs.dic['scopeResolutions'])
        
        ## Depad the array to reduce computational expense
        rFN = util.trimFilter(rFN)

        ## Check and see if we can decompose the rotated filter into a combo of 1D filters (much quicker)
        decomposable, decomp_arrays = util.decompose_3D(rFN,verbose=True)

        #if decomposable:
          # store the decomposed filters in inputs
        #  inputs.mf = decomp_arrays
        #else: inputs.mf = rFN
        
        inputs.mf = decomp_arrays
        #inputs.mf = rFN

        ## check to see if we need to rotate other matched filters for the detection
        if params['filterMode'] == 'punishmentFilter':
          params['mfPunishmentRot'] = util.rotate3DArray_Nonhomogeneous(params['mfPunishment'].copy(),i,inputs.dic['scopeResolutions'])
          params['mfPunishmentRot'] = util.trimFilter(params['mfPunishmentRot'])
          _,params['mfPunishmentRot'] = util.decompose_3D(params['mfPunishmentRot'])
      else:
        ## This is 2D
        ## pad/rotate 
        params['angle'] = i
        rFN = util.PadRotate(filterRef,i)  
        ## Depad the array to reduce computational expense
        rFN = util.autoDepadArray(rFN)
        inputs.mf = rFN  

        ## check for other matched filters
        if params['filterMode'] == 'punishmentFilter':
          params['mfPunishmentRot'] = util.PadRotate(params['mfPunishment'].copy(),i)
      
      ## Perform matched filtering 
      result = dps.FilterSingle(inputs,params)      

      ## Check to see what our storage scheme is and store accordingly
      if efficientRotationStorage:
        ## Get element-wise comparison of this rotation's SNR to all previous SNRs
        if params['inverseSNR']:
          ## If we're using the inverseSNR scheme of hit detection, we need which SNRs are less
          ##   at this rotation than previous SNRs at previous rotations 
          SNRcomparison = np.less(result.snr, correlated['SNRArray'])
        else:
          ## Otherwise, we just pick out the SNRs which are greater at this rotation compared to
          ##   previous rotations
          SNRcomparison = np.greater(result.snr, correlated['SNRArray'])

        ## Pick out the greater SNRs and store in maxSNRArray
        correlated['SNRArray'][SNRcomparison] = result.snr[SNRcomparison]

        ## Pick out the rotations at which the maximum SNR is located at the current location and 
        ##   store in the array
        if len(np.shape(img)) == 3:
          correlated['rotSNRArray']['x'][SNRcomparison] = i[0]
          correlated['rotSNRArray']['y'][SNRcomparison] = i[1]
          correlated['rotSNRArray']['z'][SNRcomparison] = i[2]
        else:  
          correlated['rotSNRArray'][SNRcomparison] = i

      else:
        ## Store results contain both correlation plane and snr
        result.rFN = np.copy(rFN)
        correlated.append(result) 

    ## Write the correlation planes if this is desired
    if printer and not efficientRotationStorage: 
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
              returnAngles=False,
              efficientRotationStorage=False):
    '''This function goes through and 'stacks' the hits across all rotations if this hasn't been 
    done previously. This is necessary when efficientRotationStorage is False.

    Inputs:
      correlated -> list or dict. If efficientRotationStorage is False. Correlated is a list of
                      results classes (see correlateThresher function for objects in class).
                      If efficientRotationStorage is True, this is a dict. The keys of which
                      can also be seen in correlateThresher.
      paramDict -> dict. Dictionary containing parameters for the characterization routines. 
                     Generated via optimizer.ParamDict() function.
      iters -> list of ints (2D classification) or list of list of int (3D classification).
                List of rotations to consider for the classification.
      display -> Bool. If True, saves plots for debugging.
      rescaleCorr -> Deprecated.
      doKMeans -> Deprecated.
      returnAngles -> Boolean. If True, return the angles with which the greatest SNR appears.
      efficientRotationStorage -> Bool. If True, correlated already contains 'stacked' hits
                                    and that simplifies this routine immensely.
    
    Outputs:
      stacked -> array. Array with same dimensions as image where hits are marked with their 
                   maximum SNR across all rotations and non-hits are marked with NaNs
      if returnAngles:
        if image is 2D:
          stackedAngles -> array. Array with same shape as image where hits are marked as their
                             rotation of maximum SNR and non-hits are marked as -1.
        elif image is 3D:
          stackedAngles -> dict. Dictionary containing arrays containing rotation info for 
                            x, y, and z where keys are 'x', 'y', and 'z', respectively.
    '''
     # TODO
    if rescaleCorr:
      raise RuntimeError("Why is this needed? IGNORING")
    if display:
      print ("Call me debug" )

    ### Check to see if we previously used the new storage technique
    ###   if we did, then we can use a shortcut with this routine since we've done a lot of the leg
    ###   work already. Else, we'll have to do some more work to stack the detection or hits.
    if efficientRotationStorage:
      ## We've already wrote the previous routines in a way that we have the maximum SNR and stackedAngles 
      ##   in a format close to compatible with the other routines. We just need to mask out non-hits
      stackedHits = util.makeMask(paramDict['snrThresh'],
                                  img = correlated['SNRArray'],
                                  inverseThresh = paramDict['inverseSNR'])
      if returnAngles:
        ## Now we fix the stackedAngles format
        stackedAngles = correlated['rotSNRArray']
        if len(np.shape(stackedHits)) == 2: # indicates 2D image
          stackedAngles[stackedHits == 0] = 361 # 361 indicates no hit at that pixel/voxel
        else: # indicates 3D image
          # there's no good way to represent this using our current angle storage structure
          pass
        return stackedHits, stackedAngles
      else:
        return stackedHits

    ### Otherwise, we have to do some work to pick out the SNRs of our hits across all rotations
    ###   First, we select hits based on those entries about the snrThresh 
    maskList = []
    simpleCorrMaskList = []
    for i, iteration in enumerate(iters):
        ## routine for identifying 'unique' hits
        try:
          daMask = util.makeMask(paramDict['snrThresh'],img = correlated[i].snr,
                                  inverseThresh=paramDict['inverseSNR'])
        except:
          print ("DC: Using workaround for tissue param dictionary. Fix me.")
          daMask = util.makeMask(paramDict['snrThresh'], img=correlated[i].snr)
        ## pull out where there is a hit on the simple correlation for use in rotation angle
        hitMask = daMask > 0.
        simpleCorrMask = correlated[i].corr
        simpleCorrMask[np.logical_not(hitMask)] = 0
                                  
        ##  print debugging info                                    
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

  coloredImg = rawOrig.copy()

  leftmostIdx = 0 # mark as left channel
  rightmostIdx = len(iters) # mark as right channel
  # any idx between these two will be colored a blend of the two channels

  spacing = 255 / rightmostIdx

  for i in range(dims[0]):
    for j in range(dims[1]):
      rotArg = stackedAngles[i,j]
      if rotArg != -1 and rotArg != 361: # the two values that indicate no hit at this location
        rotIndex = np.argmax([rotArg == it for it in iters])
        coloredImg[i,j,channelDict[leftChannel]] = int(255 - rotIndex*spacing)
        coloredImg[i,j,channelDict[rightChannel]] = int(rotIndex*spacing)
  return coloredImg

###################################################################################################
###
### Filter Labeling Routines
###
###################################################################################################

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

def doLabel_dilation(filterResults, thisFilter, inputs, eps = 1e-14):
  '''Performs the superimposing of the filter onto each filter 'hit' of the result image by dilation
  instead of by convolution. This is a bit faster and can handle asymmetric filters. Thus, marking
  of filter hits at different rotations is much more favorable in this routine.'''
  
  # Get temporary binary representation of hits to use with dilation routine
  temp_binary_hits = np.greater(np.nan_to_num(filterResults.stackedHits),eps)

  # Get binary representation of the filter
  binary_filter = np.greater(thisFilter, np.min(thisFilter) + eps)

  # Get unique permutations of rotations for x, y, and z
  if isinstance(filterResults.correlated['rotSNRArray'], dict): # indicates the image has x, y, and z rotations
    unique_x = [x for x in np.unique(filterResults.correlated["rotSNRArray"]['x']) if x <= 360]
    unique_y = [x for x in np.unique(filterResults.correlated["rotSNRArray"]['y']) if x <= 360]
    unique_z = [x for x in np.unique(filterResults.correlated["rotSNRArray"]['z']) if x <= 360]

    print ("Unique xs: {}".format(unique_x))    
    print ("Unique ys: {}".format(unique_y))    
    print ("Unique zs: {}".format(unique_z))

    labeled = np.zeros_like(filterResults.stackedHits,dtype=bool)

    for x_rot in unique_x:
      these_unique_x_hits = np.equal(filterResults.correlated['rotSNRArray']['x'], x_rot)
      for y_rot in unique_y:
        these_unique_xy_hits = np.logical_and(
          these_unique_x_hits,
          np.equal(filterResults.correlated['rotSNRArray']['y'], y_rot)
        )
        for z_rot in unique_z:
          # rotate the filter 
          # rFN = util.rotate3DArray_Nonhomogeneous(
          #   thisFilter,
          #   [x_rot, y_rot, z_rot],
          #   inputs.dic['scopeResolutions']
          # )
          rFN = thisFilter

          binary_filter = np.greater(rFN, np.min(rFN + eps))

          # find the hits that correspond with this rotation angle exactly
          where_hits = np.logical_and(
            np.logical_and(
              these_unique_xy_hits,
              np.equal(filterResults.correlated['rotSNRArray']['z'], z_rot)
            ),
            temp_binary_hits
          )

          # dilate this with the binarized rotated filter
          new_dilation = ndimage.morphology.binary_dilation(where_hits, structure = binary_filter)

          # store these new dilated hits in the storage array
          labeled = np.logical_or(new_dilation, labeled)
  
  else: # this is 2D and much simpler
    # TODO
    labeled = ndimage.morphology.binary_dilation(filterResults.stackedHits, binary_filter)

  return labeled
  

###################################################################################################
###
### Miscellaneous Routines
###
###################################################################################################

def WT_SNR(Img, WTfilter, WTPunishmentFilter,C,gamma):
  # calculates SNR of WT filter
  
  # get two responses
  h = mF.matchedFilter(Img, WTfilter, demean=False)
  h_star = mF.matchedFilter(Img,WTPunishmentFilter,demean=False)
  
  # calculate SNR
  SNR = h / (C + gamma * h_star)

  return SNR
