import sys
import os
import matplotlib.pylab as plt 
import numpy as np 
import cv2
import scipy
import scipy.signal as sig
import scipy.fftpack as fftp
from scipy import ndimage
import imutils
import operator
import tifffile
import painter
import bankDetect as bD
import yaml
import optimizer

### Temporarily raising error with tensorflow.
###   We should be getting rid of this soon
try:
  import tensorflow as tf
except:
  print "Tensorflow was not found on this computer. Routines with GPU implementation will not work."
 
### Create an empty instance of a class
class empty:
  pass

root = "myoimages/"

###################################################################################################
###################################################################################################
###################################################################################################
###
### Functions for Convenience
###
###################################################################################################
###################################################################################################
###################################################################################################

def myplot(img,fileName=None,clim=None):
  plt.axis('equal')
  plt.pcolormesh(img, cmap='gray')
  plt.colorbar()
  if fileName!=None:
    plt.gcf().savefig(fileName,dpi=300)
  if clim!=None:
    plt.clim(clim)

def ReadImg(fileName,cvtColor=True,renorm=False,bound=False, dataType = np.float32):
  ### Check to see what the file type is
  fileType = fileName[-4:]

  if fileType == '.tif':
    ## Read in image
    img = tifffile.imread(fileName)

    ## Check dimensionality of image. If image is 3D, we want to roll the z axis to the last position since 
    ##   tifffile reads in the z-stacks in the first dimension.
    if len(np.shape(img)) == 3:
      img = np.moveaxis(img,source=0,destination=2)

  elif fileType == '.png':
    ## Read in image
    img = cv2.imread(fileName)

    ## Check that an image was actually read in
    if img is None:
        raise RuntimeError(fileName+" likely doesn't exist")

    ## Convert to grayscale
    if cvtColor:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## Check if the image is bounded
    if bound != False:
      img=img[bound[0]:bound[1],bound[0]:bound[1]]

  else:
    raise RuntimeError("File type is not understood. Please use a .png or .tif file.")

  ### Normalize the image to have maximum value of 1.
  if renorm:
    img = img / np.float(np.amax(img))

  ### Conver the data type of the image to the one that is specified
  img = img.astype(dataType)
  
  return img 

def LoadFilter(fileName):
  '''
  This function serves the purpose of reading in and ensuring that the sum of the filter is 1.
    This guarantees that no pixel in the filtered image will be above the local maximum of the filter.
    i.e. this is turning the read-in filter into a mean filter.
  '''
  
  ### Read image using previous routine
  filterImg = ReadImg(fileName,cvtColor=True,renorm=True).astype(np.float64)

  ### Divide out the sum of the filter s.t. new sum of the filter will be 1.
  filterImg /=  np.sum(filterImg)

  return filterImg

def Save3DImg(img, fileName, switchChannels=False):
  '''
  This function will roll the axis of img and save the image in the img variable to the fileName location.
    This is necessary since the convention that I have been using in the analysis scheme is to have:
      [Row, Column, Z-stack]
    BUT tifffile uses the following convention:
      [Z-stack, Row, Column]

  Inputs:
    img -> numpy array with dimensions [Row, Column, Z-stack]. Image to be saved.
    fileName -> str. String containing the location and file name with which image will be saved
    switchChannels -> Bool. If True, switch the channels in index 0 and 2 of the last axis.
  '''
  ### Switch the color channels because tifffile has the same color convention as matplotlib
  if switchChannels:
    img = switchBRChannels(img)

  ### Roll the third axis to the first position
  dummyImg = np.moveaxis(img,source=2,destination=0)

  ### Convert to 16 bit float format since ImageJ can't handle 64 or 32 bit
  if dummyImg.dtype == np.float64: #or dummyImg.dtype == np.float32:
    dummyImg = dummyImg.astype(np.float32)

  ### Write image
  tifffile.imsave(fileName, data=dummyImg)

  print "Wrote file to:",fileName

def measureFilterDimensions(grayFilter,returnFilterPaddingLocations=False,verbose=False,epsilon = 1e-8):
  '''
  Measures where the filter has any data and returns a minimum bounding
  rectangle around the filter. To be used in conjunction with the 
  pasteFilters flag in DataSet

  Inputs:
    epsilon -> float. Value that we determine that no information is present within the array. This is to deal with numerical artifacts.

  Outputs:
    newDimLengths -> List of ints. List of the lengths of the depadded dimensions
    paddingLocs -> List of ints. Location of the first slice with information in each axis. This is list as 
                     [[# of voxels before first slice, # of voxels after last slice]]. So to use this information
                     for indexing when this is returned, we need to index as:
                       grayFilter[# of voxels before first slice:-# of voxels after last slice]
  '''
  ### Measure shape of filter
  filterShape = np.shape(grayFilter)

  ### For each axis, determine the amount of padding in each direction and subtract that from the measured dimensions
  newDimLengths = []
  if returnFilterPaddingLocations:
    paddingLocs = []
  for i,dimLength in enumerate(filterShape):
    otherAxes = np.delete(np.arange(len(filterShape)),i)

    if verbose:
      print "Axes with which we are summing: {}".format(otherAxes)
       
    collapsedDim = np.sum(grayFilter,axis=tuple(otherAxes))

    if verbose:
      print "Sum along both of the previous axes {}".format(collapsedDim)
    
    ## Measure padding before the filter in this dimension
    previousPadding = np.argmax(collapsedDim > epsilon)
    ## Measure padding after the filter in this dimension
    afterPadding = np.argmax(collapsedDim[::-1] > epsilon)

    ## Find actual filter dimensions (minus padding)
    newDimLengths.append(dimLength - previousPadding - afterPadding)

    if returnFilterPaddingLocations:
      paddingLocs.append([previousPadding,afterPadding])

  if verbose:
    print "Filter Dimensions:",newDimLengths

  if returnFilterPaddingLocations:
    return newDimLengths, paddingLocs
  else:
    return newDimLengths

def viewer(tensor):
  '''
  Quick and dirty function for viewing a tensor used with tensorflow
  '''
  def dummy(tensor2):
    return tensor2

  with tf.Session() as sess:
    result = sess.run(dummy(tensor))
    print np.shape(result)
    print result
    return result

def markMaskOnMyocyte(img,imgName):
  '''
  Function that takes the image of the myocyte (either grayscale or RGB) and
    draws the contour of the mask onto the image in yellow

  INPUTS:
    - img -> image of the myocyte (grayscale or RGB)
    - imgName -> name of the image that was read in
  '''

  ### Read in mask
  try:
    maskName = imgName[:-4]+"_mask"+imgName[-4:]
    mask = np.asarray(ReadImg(maskName,renorm=True) * 255.,dtype=np.uint8)
  except:
    print "No mask found, circumventing marking"
    return img

  ### ROI is marked as greatest pixel intensity, so we thresh and only keep the ROI
  mask[mask<255] = 0

  ### use cv2 to find contours
  mask2, maskContours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  ### find channels of image
  imgDim = np.shape(img)

  ### convert the image to RGB if grayscale
  if len(imgDim) == 2:
    colorImg = np.zeros((imgDim[0],imgDim[1],3),dtype=img.dtype)
    for i in range(3):
      colorImg[:,:,i] = img
  else:
    colorImg = img

  ### use cv2 to draw contour on image as yellow line
  color = (0,255,255) # NOTE: this is using matplotlibs convention of color, not cv2
  lineThickness = 2
  colorImg = cv2.drawContours(colorImg,maskContours,-1, color, lineThickness)
  
  debug = False
  if debug:
    cv2.imshow('Myocyte with mask drawn',colorImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  return colorImg

def PasteFilter(img, filt):
  '''
  function to paste the filter in the upper left region of the img 
  to make sure they are scaled correctly
  '''
    
  myImg = img.copy()
  filtDim = np.shape(filt)
  if len(myImg.shape) == 4:
    for i in range(myImg.shape[-1]):
      myImg[:filtDim[0],:filtDim[1],:filtDim[2],i] = filt
  elif len(myImg.shape) == 3:
    myImg[:filtDim[0],:filtDim[1],:filtDim[2]] = filt
  else:
    myImg[:filtDim[0],:filtDim[1]] = filt
  return myImg

# Embegs signal into known image for testing 
def embedSignal(img,mf,loc=None,scale=0.5):
    #plt.figure()
    s= np.max(img)*scale
    mfs= np.array(mf*s,dtype=np.uint8)
    imgEmb = np.copy(img)
    dimr = np.shape(mf)
    if isinstance(loc,np.ndarray):
      1
    else: 
      loc = [0,0]
    imgEmb[loc[0]:(loc[0]+dimr[0]),loc[1]:(loc[1]+dimr[1])] += mfs 
    #imshow(imgEmb)
    return imgEmb

def switchBRChannels(img):
  '''This function is a quick way to switch between color conventions. Example, cv2 and matplotlib have blue and red channels switched.
  The 'right' color convention that we've chosen is for the:
    - TT content to be marked in the zeroth channel
    - LT content to be marked in the first channel
    - TA content to be marked in the second channel
  
  Inputs:
    img -> numpy array.

  Outputs:
    newImg -> numpy array. Copy of img but with the zeroth and second channels switched.
  '''
  newImg = img.copy()

  # ensuring to copy so that we don't accidentally alter the original image
  newImg[...,0] = img[...,2].copy()
  newImg[...,2] = img[...,0].copy()

  return newImg

###################################################################################################
###################################################################################################
###################################################################################################
###
### Functions for Image/Filter Generation
###
###################################################################################################
###################################################################################################
###################################################################################################

def makeCubeFilter(prismFilter):
  '''
  Function to make a filter that is sufficiently padded with zeros such that 
  any rotation performed on the filter will not cause the filter information
  to be clipped by any rotation algorithm.
  '''
  # get shape of old filter
  fy,fx,fz = np.shape(prismFilter)

  # get shape of new cubic filter
  biggestDimension = np.max((fy,fx,fz))
  newDim = int(np.ceil(np.sqrt(2) * biggestDimension))
  if newDim % 2 != 0:
    newDim += 1

  # construct holder for new filter
  cubeFilt = np.zeros((newDim,newDim,newDim),dtype=np.float64)
  center = newDim / 2

  # store old filter in the new filter
  cubeFilt[center - int(np.floor(fy/2.)):center + int(np.ceil(fy/2.)),
           center - int(np.floor(fx/2.)):center + int(np.ceil(fx/2.)),
           center - int(np.floor(fz/2.)):center + int(np.ceil(fz/2.))] = prismFilter

  return cubeFilt

def generateWTFilter(WTFilterRoot=root+"/filterImgs/WT/", filterTwoSarcSize=25):
  WTFilterImgs = []
  for fileName in os.listdir(WTFilterRoot):
      img = cv2.imread(WTFilterRoot+fileName)
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
      # let's try and measure two sarc size based on column px intensity separation
      colSum = np.sum(gray,axis=0)
      colSum = colSum.astype('float')
      # get slopes to automatically determine two sarcolemma size for the filter images
      rolledColSum = np.roll(colSum,-1)
      slopes = rolledColSum - colSum
      slopes = slopes[:-1]
      
      idxs = []
      for i in range(len(slopes)-1):
          if slopes[i] > 0 and slopes[i+1] <= 0:
            idxs.append(i)
      if len(idxs) > 2:
          raise RuntimeError, "You have more than two peaks striations in your filter, think about discarding this image"
    
      twoSarcDist = 2 * (idxs[-1] - idxs[0])
      scale = float(filterTwoSarcSize) / float(twoSarcDist)
      resizedFilterImg = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
      WTFilterImgs.append(resizedFilterImg)

  minHeightImgs = min(map(lambda x: np.shape(x)[0], WTFilterImgs))
  for i,img in enumerate(WTFilterImgs):
      WTFilterImgs[i] = img[:minHeightImgs,:]

  colSums = []
  sortedWTFilterImgs = sorted(WTFilterImgs, key=lambda x: np.shape(x)[1])
  minNumCols = np.shape(sortedWTFilterImgs[0])[1]
  for img in sortedWTFilterImgs:
      colSum = np.sum(img,axis=0)
    
  bestIdxs = []
  for i in range(len(sortedWTFilterImgs)-1):
      # line up each striation with the one directly 'above' it in list
      img = sortedWTFilterImgs[i].copy()
      img = img.astype('float')
      lengthImg = np.shape(img)[1]
      nextImg = sortedWTFilterImgs[i+1].copy()
      nextImg = nextImg.astype('float')
      nextLengthImg = np.shape(nextImg)[1]
      errOld = 10e10
      bestIdx = 0
      for idx in range(nextLengthImg - minNumCols):
          err = np.sum(np.power(np.sum(nextImg[:,idx:(minNumCols+idx)],axis=0) - np.sum(img,axis=0),2))
          if err < errOld:
              bestIdx = idx
              errOld = err
      bestIdxs.append(bestIdx)
      sortedWTFilterImgs[i+1] = nextImg[:,bestIdx:(minNumCols+bestIdx)]

  WTFilter = np.mean(np.asarray(sortedWTFilterImgs),axis=0)
  WTFilter /= np.max(WTFilter)
  return WTFilter

def generateWTPunishmentFilter(LongitudinalFilterName,
                               rowMin=2,rowMax=-1,colMin=6,colMax=13):
  # generates the punishment filter in WT SNR calculation based upon longitudinal filter
  LongFilter = ReadImg(LongitudinalFilterName)
  punishFilter = LongFilter.copy()
  if rowMax == None:
    punishFilter = punishFilter[rowMin:,:]
  else:
    punishFilter = punishFilter[rowMin:rowMax,:]
  if colMax == None:
    punishFilter = punishFilter[:,colMin:]
  else:
    punishFilter = punishFilter[:,colMin:colMax]
  return punishFilter

def saveSingleTTFilter():
  '''
  Generates a bar filter for detection of a single transverse tubule
  '''
  WTfilter = np.zeros((20,12),dtype=np.float64)
  WTfilter[:,3:-3] = 1.
  WTfilter *= 255.
  WTfilter = WTfilter.astype(np.uint8)
  cv2.imwrite("./myoimages/singleTTFilter.png",WTfilter)

def saveSingleTTPunishmentFilter():
  '''
  Generates a punishment corollary to the generateSingleTTFilter() function
  above.
  '''
  punishFilter = np.zeros((20,12),dtype=np.float64)
  punishFilter[:,:3] = 1.; punishFilter[:,-3:] = 1.
  punishFilter *= 255.
  punishFilter =  punishFilter.astype(np.uint8)
  cv2.imwrite("./myoimages/singleTTPunishmentFilter.png",punishFilter)

# # 'Fixing' filters (improving contrast) via basic thresholding
def fixFilter(Filter,pixelCeiling=0.7,pixelFloor=0.4,rowMin=0, rowMax=None, colMin=0, colMax=None):
  fixedFilter = Filter.copy()
  fixedFilter[fixedFilter > pixelCeiling] = pixelCeiling
  fixedFilter[fixedFilter < pixelFloor] = 0
  fixedFilter /= np.max(fixedFilter)
  if rowMax == None:
    fixedFilter = fixedFilter[rowMin:,:]
  else:
    fixedFilter = fixedFilter[rowMin:rowMax,:]
  if colMax == None:
    fixedFilter = fixedFilter[:,colMin:]
  else:
    fixedFilter = fixedFilter[:,colMin:colMax]
  return fixedFilter

def saveFixedWTFilter(WTFilterRoot=root+"filterImgs/WT/",filterTwoSarcSize=25,
                      pixelCeiling=0.6,pixelFloor=0.25,
                      rowMin=20, rowMax=None, colMin=1, colMax=None):
  # opting now to save WT filter and load into the workhorse script instead of generating filter every call
  WTFilter = generateWTFilter(WTFilterRoot=WTFilterRoot,filterTwoSarcSize=filterTwoSarcSize)
  fixedFilter = fixFilter(WTFilter,pixelCeiling=pixelCeiling,pixelFloor=pixelFloor,
                          rowMin=rowMin,rowMax=rowMax,colMin=colMin,colMax=colMax)
  # convert to png format
  savedFilt = fixedFilter * 255
  savedFilt = savedFilt.astype('uint8')

  # cropping image
  savedFilt = savedFilt[6:,:]

  # save filter
  cv2.imwrite(root+"WTFilter.png",savedFilt)

def saveSimpleWTFilter():
  '''
  function to write the wt filter used as of June 5, 2018
  '''
  filterLength = 10
  TTwidth = 6
  bufferWidth = 1
  punishWidth = 5
  filterWidth = TTwidth + bufferWidth + punishWidth + bufferWidth + TTwidth
  WTfilter = np.zeros((filterLength,filterWidth),dtype=np.uint8)
  punishFilter = np.zeros_like(WTfilter)

  WTfilter[:,:TTwidth] = 255
  WTfilter[:,-TTwidth:] = 255 

  punishFilter[:,(TTwidth+bufferWidth):-(TTwidth+bufferWidth)] = 255

  cv2.imwrite("./myoimages/newSimpleWTFilter.png", WTfilter)
  cv2.imwrite("./myoimages/newSimpleWTPunishmentFilter.png",punishFilter)

def saveGaussLongFilter():
  ### Take Some Measurements of LT Growths in Real Myocytes
  height = 3 # pixels
  width = 15 # pixels
  
  ### Make Gaussian for Filter
  std = 4
  squish = 1.2
  gauss = sig.gaussian(width,std)
  gauss /= squish
  gauss += (1 - 1./squish)

  ### Generate Filter
  LTfilter = np.zeros((height+1,width+2))
  imgDim = np.shape(LTfilter)
  cY,cX = int(round(imgDim[0]/2.)),int(round(imgDim[1]/2.))
  loc = 1
  while loc < height:
    LTfilter[loc,1:-1] = gauss
    loc += 1

  ### Save in CV2 Friendly Format
  LTfilter *= 255
  LTfilter = np.asarray(LTfilter,np.uint8)
  cv2.imwrite("./myoimages/LongitudinalFilter.png",LTfilter)

def saveFixedLossFilter():
  dim = 16
  img = np.zeros((dim+2,dim+2,),dtype='uint8')
  img[2:-2,2:-2] = 255
  cv2.imwrite(root+"LossFilter.png",img)

def saveFixedPunishmentFilter():
  punishFilter = np.zeros((28,7),dtype='uint8')
  punishFilter[1:-1,2:-2] = 255
  cv2.imwrite(root+"WTPunishmentFilter.png",punishFilter)

def saveAllMyo():
      print "Generating and saving all 2D filters."
      saveFixedWTFilter()
      saveSimpleWTFilter()
      saveGaussLongFilter()
      saveFixedLossFilter()
      saveFixedPunishmentFilter()
      saveSingleTTFilter()
      saveSingleTTPunishmentFilter()
      scopeResolutions = [10,10,5]
      print "Generating and saving all 3D filters."
      print "WARNING: This is generating 3D filters with an assumed scope resolution of xy = 10 vx/um and z = 5 vx/um."
      generate3DTTFilter(scopeResolutions)
      generate3DLTFilter(scopeResolutions)
      generate3DTAFilter(scopeResolutions)

def determineZStacksFromResolution(zResolution, # [voxels/um]
                                   DiameterTT=.2 # [um]
                                   ):
  '''
  This function determines the number of z stacks needed to cover transverse tubules given a 
    z-stack resolution of the confocal microscope used.

  Inputs:
    zResolution -> float. Resolution of the confocal microscope used to record the image in [voxels/um]
    DiameterTT -> float. Diameter of a typical transverse tubule in the species of the animal used.
                    Default value is .2 um
  '''

  numZStacksNeeded = DiameterTT * zResolution

  return numZStacksNeeded

def generate3DTTFilter(scopeResolutions, # [vx/um]
                       originalFilterName='./myoimages/newSimpleWTFilter.png',
                       originalPunishFilterName='./myoimages/newSimpleWTPunishmentFilter.png',
                       diameterTT = 0.2, # [um]
                       #filterZLength=3, # [um]
                       ):
  '''
  This function generates a 3D filter for the detection of prototypical transverse tubules
  
  Inputs:
    scopeResolutions - list of values. List of the resolutions of the confocal microscope for x, y, and z
    originalFilterName - str. Name of the original TT filter we would like to extrude
    originalPunishFilterName - str. Name of the original TT punishment filter we would like to extrude
    diameterTT - float or int. Diameter of the typical transverse tubule in the animal model used in microns
    #filterZLength - float or int. Length of the output TT filter in microns. NOT INCORPORATED YET
  '''

  ### Read in images
  filt = LoadFilter(originalFilterName)
  punishFilt = LoadFilter(originalPunishFilterName)

  ### Stack the 2D filters to extrude them and form 3D filter
  ###   Here we are extruding in what will be the x axis in the 3D filter, so we'll use that resolution
  filt3D = np.stack((filt,filt),axis=2)
  punishFilt3D = np.stack((punishFilt,punishFilt),axis=2)
  numXStacks = int(round(diameterTT * scopeResolutions[0]))
  if numXStacks > 1:
    for i in range(numXStacks - 1):
      filt3D = np.dstack((filt3D, filt))
      punishFilt3D = np.dstack((punishFilt3D,punishFilt))

  ### Rotate the 3D filters to orient them correctly
  filt3D = pad3DArray(filt3D)
  filt3D = rotate3DArray_Homogeneous(filt3D, angles=[0, 90, 0])
  punishFilt3D = pad3DArray(punishFilt3D)
  punishFilt3D = rotate3DArray_Homogeneous(punishFilt3D, angles=[0, 90, 0])
  
  ### Downsample the filters in the (new) z direction
  zZoomOut = float(scopeResolutions[2]) / float(scopeResolutions[0])
  zoomOut = [1., 1., zZoomOut]
  filt3D = ndimage.zoom(filt3D,zoom=zoomOut)
  punishFilt3D = ndimage.zoom(punishFilt3D, zoom=zoomOut)

  # temporarily save the intremediate result for debug purposes
  print np.max(filt3D)
  print np.min(filt3D)
  Save3DImg(filt3D.astype(np.uint16), './myoimages/DEBUG.tif')

  ### Threshold the filter to get rid of rotation numerical artifacts
  filt3DMean = np.mean(filt3D)
  #filt3DMax = np.max(filt3D)
  filt3D[filt3D > filt3DMean] = 1.
  filt3D[filt3D < filt3DMean] = 0
  punishFilt3DMean = np.mean(punishFilt3D)
  #punishFilt3DMax = np.max(punishFilt3D)
  punishFilt3D[punishFilt3D > punishFilt3DMean] = 1.
  punishFilt3D[punishFilt3D < punishFilt3DMean] = 0
  
  ### Save filters
  Save3DImg(filt3D.astype(np.uint16), './myoimages/TT_3D.tif')
  Save3DImg(punishFilt3D.astype(np.uint16), './myoimages/TT_Punishment_3D.tif')

def generate3DLTFilter(scopeResolutions, # [vx/um]
                       originalFilterName='./myoimages/LongitudinalFilter.png',
                       diameterTT = 0.2, # [um]
                       #filterZLength=3, # [um]
                       ):
  '''
  This function generates a 3D filter for the detection of prototypical longitudinal tubules
  
  Inputs:
    scopeResolutions - list of values. List of the resolutions of the confocal microscope for x, y, and z
    originalFilterName - str. Name of the original TT filter we would like to extrude
    originalPunishFilterName - str. Name of the original TT punishment filter we would like to extrude
    diameterTT - float or int. Diameter of the typical transverse tubule in the animal model used in microns
    #filterZLength - float or int. Length of the output TT filter in microns.
  '''
  ### Read in images
  filt = LoadFilter(originalFilterName)

  ### Stack the 2D filters to extrude them and form 3D filter
  ###   Here we are extruding in what will be the x axis in the 3D filter, so we'll use that resolution
  filt3D = np.stack((filt,filt),axis=2)
  numXStacks = int(round(diameterTT * scopeResolutions[0]))
  if numXStacks > 1:
    for i in range(numXStacks - 1):
      filt3D = np.dstack((filt3D, filt))

  ### Rotate the 3D filters to orient them correctly
  filt3D = pad3DArray(filt3D)
  filt3D = rotate3DArray_Homogeneous(filt3D, angles=[0, 90, 0])
  
  ### Downsample the filters in the (new) z direction
  zZoomOut = float(scopeResolutions[2]) / float(scopeResolutions[0])
  zoomOut = [1., 1., zZoomOut]
  filt3D = ndimage.zoom(filt3D,zoom=zoomOut)

  ### Threshold the filter to get rid of rotation numerical artifacts
  filt3DMean = np.mean(filt3D)
  #filt3DMax = np.max(filt3D)
  filt3D[filt3D > filt3DMean] = 1.
  filt3D[filt3D < filt3DMean] = 0
  
  ### Save filters
  Save3DImg(filt3D.astype(np.uint16), './myoimages/LT_3D.tif')

def generate3DTAFilter(scopeResolutions, # [vx/um]
                       originalFilterName='./myoimages/LossFilter.png',
                       lengthTARegion = 0.9, # [um]
                       ):
  '''
  This function generates a 3D filter for the detection of prototypical longitudinal tubules
  
  Inputs:
    scopeResolutions - list of values. List of the resolutions of the confocal microscope for x, y, and z
    originalFilterName - str. Name of the original TT filter we would like to extrude
    originalPunishFilterName - str. Name of the original TT punishment filter we would like to extrude
    lengthTARegion - float or int. Necessary length of missing TT structure needed to be considered a tubule absence region
  '''
  ### Read in images
  filt = LoadFilter(originalFilterName)

  ### Stack the 2D filters to extrude them and form 3D filter
  ###   Here we are extruding in what will be the x axis in the 3D filter, so we'll use that resolution
  filt3D = np.stack((filt,filt),axis=2)
  _,numXStacks = measureFilterDimensions(filt)
  #numXStacks = int(round(lengthTARegion * scopeResolutions[0]))
  for i in range(numXStacks - 1):
    filt3D = np.dstack((filt3D, filt))
  
  ### Rotate the 3D filters to orient them correctly
  filt3D = pad3DArray(filt3D)
  filt3D = rotate3DArray_Homogeneous(filt3D, angles=[0, 90, 0])
  
  ### Downsample the filters in the (new) z direction
  #zZoomOut = float(scopeResolutions[2]) / float(scopeResolutions[0])
  _,_,currentZLength = measureFilterDimensions(filt3D)
  zZoomOut = float(lengthTARegion * scopeResolutions[2]) / float(currentZLength)
  print "WARNING: TEMPORARILY ZOOMING OUT OF X AND Y AXIS TO MAKE TA FILTER SMALLER AND WORK WITH SIMULATED DATA"
  #zoomOut = [1., 1., zZoomOut]
  zoomOut = [.75, .75, zZoomOut]
  filt3D = ndimage.zoom(filt3D,zoom=zoomOut)
  
  ### Threshold the filter to get rid of rotation numerical artifacts
  filt3DMean = np.mean(filt3D)
  #filt3DMax = np.max(filt3D)
  filt3D[filt3D > filt3DMean] = 1.
  filt3D[filt3D < filt3DMean] = 0
  
  ### Save filters
  Save3DImg(filt3D.astype(np.uint16), './myoimages/TA_3D.tif')  

def generateSimulated3DCell(FilterTwoSarcomereSize = 25, # [vx]
                            FilterZStackSize = 5, # [vx] 
                            unitCellSizeX = 10, # [vx] 
                            TTradius = 3, # [vx] 
                            scopeResolutions = [5,5,2], # [vx / um] 
                            LT_probability = 0.1, 
                            TA_probability = 0.1, 
                            noiseType = 'additive',
                            noiseDistribution = 'Guassian', 
                            noiseAmplitude = 0.25, 
                            cellDimensions = [25, 100, 20], # [um] 
                            fileName = "simulatedData.tif",
                            verbose=False,
                            seed=None):
    '''
    This function provides the workflow to generate a three dimensional cell based on probabilities of finding 
      normal transverse tubules (TTs), longitudinal tubules (LTs), and regions of tubule absence (TA).
      
    Inputs:
        FilterTwoSarcomereSize - int. Size of two sarcomeres of the simulated cell in voxels.
        FilterZStackSize - int. Z-stack size of the filter to be tested. ARBITRARY for now
            = 5 # [vx] Complete guess. TODO: tune this based on very light optimization
        unitCellSizeX - int. Size of the unit cells in the x direction with which we construct the simulated cell.
                          This can be tuned for longer filters.
            = 10 # [vx] Arbitray. Currently setting it to the size of the TT filter
        TTradius - int. Radius of the transverse tubules in the xy direction. This is resized for the z direction
                     based on the provided resolutions.
            = 3 # [vx] in TODO: Change to where it is specified in microns
        scopeResolutions - list of values (can be ints, floats, etc). List of the x, y, and z resolutions in vx/um.
            = [5,5,2] # [vx / um] These are placeholder values for now. Aparna said z res roughly 1/2 that of xy res
        LT_probability - float between 0 and 1. Likelihood of finding a LT unit cell in the simulated cell.
            = 0.1 # [% likelihood unit cell is LT]. Float bounded [0,1]
        TA_probability - float between 0 and 1. Likelihood of finding a TA unit cell in the simulated cell.
            = 0.1 # [% likelihood unit cell is TA]. Float bounded [0,1]. These two should add to be <= 1
        # NOTE: TT probability is 1. - TA_probability - LT_probability
        noiseType - str. Type of noise to be incorporated into the cell. This isn't incorporated yet but is left as an option for future 
                      development.
            = 'additive'
        noiseDistribution - str. Distribution of the noise to be incorporated into the cell. This isn't incorporated yet but is left as an
                              option for future development.
            = 'Guassian' # these parameters are likely going to be the only ones considered
        noiseAmplitude - float. Amplitude of the noise incorporated into the simulated cell.
            = 0.25 # float. Amplitude of the noise
        cellDimensions - list of values. Size of the simulated cell in x, y, and z directions in microns.
            = [25, 100, 20] # [um] Size of the simulated cell in x, y, and z directions. Completely arbitrary right now. TODO find better estimates
        fileName - str. Name of the output file. Must be a .tif file.
            = "simulatedData.tif"
        seed -> int. Random seed used to initialize the pseudo-random number generator. Can be any integer between 0 and 2**32 - 1 inclusive
            
    Outputs:
        None
    '''
    
    ### 0. Check for errors in inputs
    if LT_probability + TA_probability > 1.0:
        raise RuntimeError("The combined probability of finding LT and TA in the cell has been specified to be above 100%. Decrease LT_probability and/or TA_probability")
    
    ### 1. Find the Number of Unit Cells Within the Simulated Cell
    ## find unit cell size
    unitCellVoxels = [None, None, None]
    unitCellVoxels[0] = unitCellSizeX
    unitCellVoxels[1] = int(np.floor(FilterTwoSarcomereSize / 2.))
    unitCellVoxels[2] = FilterZStackSize

    ## count number of voxels in each dimension based on specified cell size and scope resolution
    numVoxelsInDimension = []
    for i in range(3):
        value = int(np.floor(float(cellDimensions[i]) * float(scopeResolutions[i])))
        numVoxelsInDimension.append(value)

    ## count the number of unit cells in each dimension
    numUnitCellsInDimension = []
    for i in range(3):
        numUnitCellsInDimension.append(int(np.floor(float(numVoxelsInDimension[i]) / float(unitCellVoxels[i]))))

    ## amend the number of voxels in each dimension since we may have to clip some voxels based on unit cell size
    for i in range(3):
        numVoxelsInDimension[i] = int(np.floor(numUnitCellsInDimension[i] * unitCellVoxels[i]))

    for i in range(3):
        print "Final truncated cell size in {} dimension: {} [um]".format(i, float(numVoxelsInDimension[i])/float(scopeResolutions[i]))
        print "Number of unit cells in {} dimension: {}".format(i, numUnitCellsInDimension[i])
        print "Number of voxels in {} dimension: {}".format(i, numVoxelsInDimension[i])

    print "Total number of unit cells: {}".format(np.prod(numUnitCellsInDimension))
    
    ## construct the cell
    cell = np.zeros(
        (
            numVoxelsInDimension[0],
            numVoxelsInDimension[1],
            numVoxelsInDimension[2]
        ),
        dtype = np.float
    )
    
    ### 2. Define The Unit Cells to Assign in the Cell
    TTcell = np.zeros(
        (
            unitCellVoxels[0],
            unitCellVoxels[1],
            unitCellVoxels[2]
        ),
        dtype = np.float
    )

    unitCellMidPoint = [
        int(np.floor(float(unitCellVoxels[0]) / 2.)),
        int(np.floor(float(unitCellVoxels[1]) / 2.)),
        int(np.floor(float(unitCellVoxels[2]) / 2.))
    ]

    TTcell[
        unitCellMidPoint[0] - TTradius:unitCellMidPoint[0] + TTradius, 
        unitCellMidPoint[1] - TTradius:unitCellMidPoint[1] + TTradius,
        :
    ] = 1.

    ## The longitudinal cell just consists of two regular TT elements that are joined by a middle tubule branch.
    ##   So we can just join together two TT cells and join them to form a LT cell
    LTcell = np.concatenate((TTcell,TTcell),axis=1)
    TTradius_z = int(np.floor(TTradius * float(scopeResolutions[2]) / float(scopeResolutions[0])))

    LTcell[
        unitCellMidPoint[0] - TTradius : unitCellMidPoint[0] + TTradius,
        unitCellMidPoint[1] : unitCellMidPoint[1] + unitCellVoxels[1],
        unitCellMidPoint[2] - TTradius_z : unitCellMidPoint[2] + TTradius_z
    ] = 1.

    ### Save unit cells to display later
    print "Saving unit cells"
    Save3DImg(TTcell, './myoimages/TTcell.tif')
    Save3DImg(LTcell, './myoimages/LTcell.tif')

    ### Generate a random number for each unit cell
    if seed:
      randomInstance = np.random.RandomState(seed=seed)
    else:
      randomInstance = np.random.RandomState()
    unitCellChances = randomInstance.rand(
      numUnitCellsInDimension[0],
      numUnitCellsInDimension[1],
      numUnitCellsInDimension[2]
    )
    
    ### 3. Loop Through Cell and Assign Unit Cells to Chunks
    i = 0; j = 0; k = 0
    while i < numUnitCellsInDimension[0]:
        k = 0
        while k < numUnitCellsInDimension[2]:
            j = 0
            while j < numUnitCellsInDimension[1]:
                location = [
                    i * unitCellVoxels[0],
                    j * unitCellVoxels[1],
                    k * unitCellVoxels[2]
                ]

                # Get the random number chance for the specific unit cell
                randNum = unitCellChances[i,j,k]

                # Determine which unit cell we will place in the position
                if randNum < TA_probability:
                    # we don't actually have to do anything for TA since the unit cell is already zero. Increment j by 2.
                    j += 2
                elif randNum > TA_probability and randNum < TA_probability + LT_probability:
                    # check to see if there is room to stick the LT unit cell since it is twice as large as the others
                    try:
                        cell[
                            location[0]:location[0]+unitCellVoxels[0],
                            location[1]:location[1]+2*unitCellVoxels[1],
                            location[2]:location[2]+unitCellVoxels[2]
                        ] = LTcell
                        j += 2
                    except:
                        if verbose:
                            print "There is not enough room to place the LT unit cell. Setting this as TA for now"
                        j += 1
                else:
                    # Try to stick in two TT unit cells. If we can't do that, we've hit the edge of the cell
                    cell[
                        location[0]:location[0]+unitCellVoxels[0],
                        location[1]:location[1]+unitCellVoxels[1],
                        location[2]:location[2]+unitCellVoxels[2]
                    ] = TTcell
                    j += 1
                    try:
                        cell[
                            location[0]:location[0]+unitCellVoxels[0],
                            location[1]+unitCellVoxels[1]:location[1]+2*unitCellVoxels[1],
                            location[2]:location[2]+unitCellVoxels[2]
                        ] = TTcell
                        j += 1
                    except:
                        if verbose:
                            print "Attempted to insert two TT unit cells but could not do to cell boundary. Setting as single TT unit cell for now."
            k += 1
        i += 1
    
    ### 4. Add in the noise
    imgNoise = np.random.rand(
        numVoxelsInDimension[0],
        numVoxelsInDimension[1],
        numVoxelsInDimension[2]
    ) * noiseAmplitude 
    cell *= (np.max(cell) - noiseAmplitude) / np.max(cell)
    #print np.max(cell)
    #print np.min(cell)
    cell = cell + imgNoise
    
    ### 5. Convert to Format That Can Be Saved and Save
    ## rescale to 16 bit int
    cell *= 65534
    cell = cell.astype(np.uint16)
    cell = np.moveaxis(cell, 2, 0)
    tifffile.imsave(fileName,data=cell)
    print "Wrote:",fileName

###################################################################################################
###################################################################################################
###################################################################################################
###
### Functions for Image Manipulation
###
###################################################################################################
###################################################################################################
###################################################################################################

def rotateTFFilter2D(img,rotation):
  '''
  Function to 'tensorize' the rotation of the filter
  '''
  #rotated = tf.cast(img,dtype=tf.float64)
  #rotated = tf.cast(img,dtype=tf.float32)
  rotated = tf.to_float(img)
  rotated = tf.contrib.image.rotate(rotated,rotation,interpolation="BILINEAR")
  rotated = tf.cast(rotated,dtype=tf.complex64)
  return rotated

def renorm(img,scale=255):
    img = img-np.min(img)
    img/= np.max(img)
    img*=scale 
    return img

def GetRegion(region,sidx,margin):
      subregion = region[(sidx[0]-margin):(sidx[0]+margin+1),
                         (sidx[1]-margin):(sidx[1]+margin+1)]        
      area = np.float(np.prod(np.shape(subregion)))
      intVal = np.sum(subregion)  
      return subregion, intVal, area

def MaskRegion(region,sidx,margin,value=0):
      region[(sidx[0]-margin):(sidx[0]+margin+1),
                         (sidx[1]-margin):(sidx[1]+margin+1)]=value 

# either load in data from file (imgName!=None) or pass in data (img!=None)
def makeMask(threshold = 245, 
             img=None,
             imgName=None,
             inverseThresh=False
             ):
    # test if numpy array
    if isinstance(img, (list, tuple, np.ndarray)): 
      correlated = img
    # test if string
    elif isinstance(imgName, (str)):
      correlated = ReadImg(imgName)
    else:
      raise RuntimeError("Need to pass in arg") 

    imgDim = np.shape(correlated)

    corr = np.copy(correlated.flatten())
    masker = (np.zeros_like(corr))
    if inverseThresh == False:
      pts =np.argwhere(corr>threshold)
      masker[pts] = corr[pts]
    else:
      pts =np.argwhere(corr<threshold)
      masker[pts] = 1.
    newmasker= np.reshape(masker,imgDim)            

    return newmasker

def ReadResizeApplyMask(img,imgName,ImgTwoSarcSize=25,filterTwoSarcSize=25):
  # function to apply the image mask before outputting results
  maskName = imgName[:-4]; fileType = imgName[-4:]
  fileName = maskName+'_mask'+fileType
  mask = cv2.imread(fileName)                       
  try:
    maskGray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
  except:
    print "No mask named '"+fileName +"' was found. Circumventing masking."
    return img
  if ImgTwoSarcSize != None:
    scale = float(filterTwoSarcSize) / float(ImgTwoSarcSize)
    maskResized = cv2.resize(maskGray,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
  else:
    maskResized = maskGray
  normed = maskResized.astype('float') / float(np.max(maskResized))
  normed[normed < 1.0] = 0
  dimensions = np.shape(img)
  if len(dimensions) < 3:
    combined = img * normed 
  else:
    combined = img
    for i in range(dimensions[2]):
      combined[:,:,i] = combined[:,:,i] * normed
  return combined

def ApplyCLAHE(grayImgList, tileGridSize, clipLimit=2.0, plot=False):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahedImage = clahe.apply(grayImgList) # stupid hack
    return clahedImage

def PadWithZeros(img, padding = 15):
  '''
  routine to pad your image with a border of zeros. This reduces the 
  unwanted response from shifting the nyquist.
  '''

  imgType = type(img[0,0])
  imgDim = np.shape(img)
  newImg = np.zeros([imgDim[0]+2*padding, imgDim[1]+2*padding])
  newImg[padding:-padding,padding:-padding] = img
  newImg = newImg.astype(imgType)
  return newImg

def Depad(img, padding=15):
  '''
  routine to return the img passed into 'PadWithZeros' 
  '''
  imgType = type(img[0,0])
  imgDim = np.shape(img)
  newImg = img[padding:-padding,padding:-padding]
  newImg = newImg.astype(imgType)
  return newImg

def padWithZeros(array, padwidth, iaxis, kwargs):
    array[:padwidth[0]] = 0
    array[-padwidth[1]:]= 0
    return array

def PadRotate(myFilter1,val):
  dims = np.shape(myFilter1)
  diff = np.min(dims)
  paddedFilter = np.lib.pad(myFilter1,diff,padWithZeros)
  rotatedFilter = imutils.rotate(paddedFilter,-val)
  rF = np.copy(rotatedFilter)

  return rF

def rotate3DArray_Nonhomogeneous(A,angles,resolutions,clipValues=True, interpolationOrder=2, verbose=False):
  '''
  This function is for the rotation of matrices with non-homogeneous coordinates. For example, 
    rotation of images taken with scopes with anisotropic resolution
  
  Inputs:
    A -> np array. Matrix with which to rotate
    angles -> list of floats. List of angles with which A will be rotated.The convention used here is that 
                the zeroth coordinate corresponds to rotation angle about the x axis (zeroth axis)
                the first coordinate corresponds to rotation angle about the y axis (first axis)
                the second coordinate corresponds to rotation angle about the z axis (second axis)
                NOTE: Angles are in degrees
    resolution -> list of resolutions for 0th, 1st, and 2nd dimensions.
    clipValues -> Bool. If true, the routine clips the output of rotations and zooms to have 
                          maximums and minimums reflective of the input array, A.
    interpolationOrder -> int. Valid values range between 0-5. Order of the interpolation for rotation.
                                 NOTE: Cubic interpolation seems to introduce artifacts. Setting default to 2
    verbose -> Bool. If True, the routine will display filter mins and maxes at the zeroth slice.
  '''

  minVal = np.min(A)
  maxVal = np.max(A)

  if verbose:
    print "Original Filter Values\n\tFilter Min: {} \n\tFilter Max: {}".format(np.min(A[:,:,0]), np.max(A[:,:,0]))

  ### 1. Interpolate and Form New Matrix with Homogeneous Coordinates
  ## Find zoom levels based on resolutions. We need to zoom in for all dimensions with resolutiosn lower than the maximum
  maxResolution = np.max(resolutions).astype(float)

  zooms = []
  for res in resolutions:
    zooms.append(maxResolution / float(res))
  
  ## Form the new zoomed in matrix
  zoomed = ndimage.zoom(A,zoom=zooms)
  if clipValues:
    zoomed[zoomed < minVal] = minVal
    zoomed[zoomed > maxVal] = maxVal

  if verbose:
    print "Filter Values After Zooming In\n\tFilter Min: {} \n\tFilter Max: {}".format(np.min(zoomed[:,:,0]),np.max(zoomed[:,:,0]))

  ### 2. Rotate the Matrix Using Homogeneous Coordinate Rotation Routine
  ## Pad the zoomed in image first to ensure that rotation does not induce artifacts
  padded = pad3DArray(zoomed)

  if verbose:
    print "Filter Values After Padding\n\tFilter Min: {} \n\tFilter Max: {}".format(np.min(padded[:,:,0]),np.max(padded[:,:,0]))

  ## Rotate the matrix
  rotated = rotate3DArray_Homogeneous(padded, angles, clipOutput=clipValues, interpolationOrder=interpolationOrder)

  if verbose:
    print "Filter Values After Rotating\n\tFilter Min: {} \n\tFilter Max: {}".format(np.min(rotated[:,:,0]),np.max(rotated[:,:,0]))

  ### 3. Downsample by Zooming Out
  ## Find the amount we'll have to zoom out to return to previous levels
  zoomOutLevels = [1./ i for i in zooms]

  ## Zoom out
  zoomedOut = ndimage.zoom(rotated, zoom=zoomOutLevels)
  if clipValues:
    zoomedOut[zoomedOut < minVal] = minVal
    zoomedOut[zoomedOut > maxVal] = maxVal

  if verbose:
    print "Filter Values After Zooming Out\n\tFilter Min: {} \n\tFilter Max: {}".format(np.min(zoomedOut[:,:,0]),np.max(zoomedOut[:,:,0]))

  return zoomedOut

def pad3DArray(array, 
             padding = 4,
             paddingValue = 0):
  '''
  This routine takes a 3D array and pads the array in all dimensions
  
  Inputs:
    array -> numpy array. Array which padding will be applied to.
    padding -> int. 1/2 the number of rows,columns,and z stacks that will be applied to the array.
    paddingValue -> Value with which the array will be padded with. Should be the same data type 
                      as the original array.
  
  Outputs:
    padded -> numpy array. Padded array.
  '''

  ### Create a big array with which will store the padded array
  arrDims = np.shape(array)
  padded = np.ones(
    (
      arrDims[0]+2*padding,
      arrDims[1]+2*padding,
      arrDims[2]+2*padding
    ),
    dtype = array.dtype
  )

  ### Set the values of the array to the padding value
  padded *= paddingValue

  ### Place the original array into the padded array
  padded[padding:-padding,padding:-padding,padding:-padding] = array

  return padded

def rotate3DArray_Homogeneous(array, angles, clipOutput=True, interpolationOrder=3):
  '''
  Rotates an array in 3D space.

  Inputs:
    arr -> numpy array
    angles -> list of values for which to rotate arr by. Values proceed as rotating about x, y, and z axes corresponding to [0,1,2] dimensions of array.
                Convention is such that traveling along the rows is the x axis and traveling along the columns is the y axis. This is slightly
                counter-intuitive but it matches with the image processing libraries better this way.
                Rotation angles are for rotation in the clockwise direction.
    padding -> int. Number of rows/cols/z stacks to add for padding. The padding value is 0.
    clipOutput -> Bool. If true, then the output is clipped to the minimum and maximum values of the input array.
    interpolationOrder -> int. Valid values range between 0-5. Order of the interpolation for rotation.
                                 NOTE: Cubic interpolation seems to introduce artifacts. May think about setting default to 2
  Outputs:
    rot -> numpy array containing the rotated array
  '''
  ### Measure the input max and min if we wish to clip the output to the input max and min range
  if clipOutput:
    inputMin = np.min(array)
    inputMax = np.max(array)

  ### I don't like the ndimage convention of rotating counter-clockwise, so I wish to feed in clockwise rotation
  ###   angles and correct within this function
  for i, arg in enumerate(angles):
    angles[i] = -1. * arg

  ### Rotate about x axis (y,z plane)
  rot = ndimage.rotate(array, angles[0], axes=(1,2),order=interpolationOrder)

  ### Clip rotation output if desired
  if clipOutput:
    rot[rot < inputMin] = inputMin
    rot[rot > inputMax] = inputMax

  ### Rotate about y axis (x,z plane)
  rot = ndimage.rotate(rot, angles[1], axes=(0,2), order=interpolationOrder)

  ### Clip rotation output if desired
  if clipOutput:
    rot[rot < inputMin] = inputMin
    rot[rot > inputMax] = inputMax

  ### Rotate about z axis (x,y plane)
  rot = ndimage.rotate(rot, angles[2], axes=(0,1), order=interpolationOrder)

  ### Clip rotation output if desired
  if clipOutput:
    rot[rot < inputMin] = inputMin
    rot[rot > inputMax] = inputMax

  return rot

def autoDepadArray(img, verbose=False):
  '''
  This function is meant to automatically strip the padding from an array

  Inputs:
    img -> Numpy array
    verbose -> Bool. If True, information about img is printed
  '''

  ### Get the locations of the start and end of the padding in the image
  ###   NOTE: Need to think of a better way to determine epsilon. This is ad hoc right now and WILL break
  ###           eventually.
  filtDims, paddingLocs = measureFilterDimensions(img,returnFilterPaddingLocations=True,epsilon=5e-5)

  if verbose:
    print "Image Padding Locations: {}".format(paddingLocs)

  ### Check to see if there is padding in all 3 dimensions. If not, exit out of the routine
  for i in range(len(paddingLocs)):
    if paddingLocs[i][0] == 0 and paddingLocs[i][1] == 0:
      print "There is no padding in the {} dimension of the image. Exiting routine without depadding.".format(i)
      return img

  ### Make a new array to store the depadded array
  newImg = np.zeros(filtDims, dtype=img.dtype)

  ### Store depadded array in new array
  if len(img.shape) == 3:
    newImg[:,:,:] = img[
      paddingLocs[0][0]:-paddingLocs[0][1],
      paddingLocs[1][0]:-paddingLocs[1][1],
      paddingLocs[2][0]:-paddingLocs[2][1]
    ]
  elif len(img.shape) == 2:
    newImg[:,:] = img[
      paddingLocs[0][0]:-paddingLocs[0][1],
      paddingLocs[1][0]:-paddingLocs[1][1],
    ]
  else:
    raise RuntimeError("Image shape incompatible with routine.")
  
  return newImg

###################################################################################################
###################################################################################################
###################################################################################################
###
### Functions for Image Analysis
###
###################################################################################################
###################################################################################################
###################################################################################################

def assessContent(markedImg,imgName=None):
  '''This function analyzes the amount of TT, LT, and TA content that is present in a marked image
  returned by giveMarkedMyocyte or give3DMarkedMyocyte.

  Inputs:
    markedImg -> numpy array. The marked image that is returned from the marking routines mentioned above.
    imgName -> str. Name of the image. If this is specified, the routine will search for an image mask that 
                 has been constructed to obfuscate the extracellular content/organelles. The naming convention
                 for the masks is "<ORIGINAL_NAME>_masked.<FILETYPE>"

  Outputs:
    ttContent, ltContent, taContent
  '''
  ### Create copy of image
  imgCopy = markedImg.copy()

  ### Pull out content-specific channels
  tt = imgCopy[...,0]
  lt = imgCopy[...,1]
  ta = imgCopy[...,2]

  ### Get rid of everything that isn't a hit (hits are marked as 255)
  tt[tt != 255] = 0
  lt[lt != 255] = 0
  ta[ta != 255] = 0

  ### normalize
  ttNormed = np.divide(tt, np.max(tt))
  ltNormed = np.divide(lt, np.max(lt))
  taNormed = np.divide(ta, np.max(ta))

  ### calculate content
  ttContent = np.sum(ttNormed)
  ltContent = np.sum(ltNormed)
  taContent = np.sum(taNormed)

  if isinstance(imgName, (str)):
    if len(np.shape(imgCopy)) == 4:
      raise RuntimeError("WARNING: Masking is not implemented in 3D so the assessment of content does not include this. \
                          This may skew results.")
    ## if imgName is included, we normalize content to cell area
    dummy = np.multiply(np.ones_like(markedImg[:,:,0]), 255)
    mask = ReadResizeApplyMask(dummy,imgName,25,25)
    mask[mask <= 254] = 0
    mask[mask > 0] = 1
    cellArea = np.sum(mask,dtype=float)
    ttContent /= cellArea
    ltContent /= cellArea
    taContent /= cellArea
    print "TT Content:", ttContent
    print "LT Content:", ltContent
    print "Loss Content:", taContent
    print "Sum of Content:", ttContent+ltContent+taContent
    ## these should sum to 1 exactly but I'm leaving wiggle room
    assert (ttContent+ltContent+taContent) < 1.2, ("Something went " 
            +"wrong with the normalization of content to the cell area calculated "
            +"by the mask. Double check the masking routine.") 
  else:
    print "TT Content:", ttContent
    print "LT Content:", ltContent
    print "TA Content:", taContent  

  return ttContent, ltContent, taContent

def estimateTubuleContentFromColoredImage(cI,
                                          totalCellSpace = None,
                                          taFilterName = './myoimages/LossFilter.png',
                                          ltFilterName = './myoimages/LongitudinalFilter.png',
                                          ttFilterName = './myoimages/newSimpleWTFilter.png',
                                          verbose = True):
  '''This function gives the estimated morphological content (TT, LT, and TA) from the image that has been
  marked with the markPastedFilters() routine.

  Inputs:
    cI -> array. Generated from the markPastedFilters routine. This contains information about the hit
            detections for each type of filter. 
    totalCellSpace -> float. If specified, the measured content will be divided out by this number 
                        and will be indicative of area/volume fraction instead of total cellular content.
    taFilterName -> Name of the tubule absence filter used in the classification of the colored image
    ltFilterName -> Name of the longitudinal tubule filter used in the classification of the colored image
    ttFilterName -> Name of the tranverse tubule filter used in the classification of the colored image

  Outputs:
    content -> dict. Dictionary containing the estimated volume/area fraction of each filter/morphological 
                 feature.
  '''

  ### Pull out the channels that contain hits for each filter
  TAchannel = cI[...,2]
  LTchannel = cI[...,1]
  TTchannel = cI[...,0]

  ### Load in the filters
  TAfilter = LoadFilter(taFilterName)
  LTfilter = LoadFilter(ltFilterName)
  TTfilter = LoadFilter(ttFilterName)

  ### Measure the occupied volume/area fractions of each filter
  TAvolFrac = measureOccupiedVolumeFraction(TAfilter)
  LTvolFrac = measureOccupiedVolumeFraction(LTfilter)
  TTvolFrac = measureOccupiedVolumeFraction(TTfilter)

  ### Find the occupied area/volume of each filter's hits
  hitValue = 255
  totalTAhits = np.count_nonzero(TAchannel == hitValue)
  totalLThits = np.count_nonzero(LTchannel == hitValue)
  totalTThits = np.count_nonzero(TTchannel == hitValue)

  ### Multiply the total number of hits by the amount of occupied space in the filter
  correctedTAContent = totalTAhits * TAvolFrac
  correctedLTContent = totalLThits * LTvolFrac
  correctedTTContent = totalTThits * TTvolFrac

  content = {
    'TA':correctedTAContent,
    'LT':correctedLTContent,
    'TT':correctedTTContent
  }

  if totalCellSpace:
    for key,value in content.iteritems():
      content[key] /= float(totalCellSpace)

  if verbose:
    for key, value in content.iteritems():
      print "Corrected {} Morphological Feature Occupied Cell Space: {}".format(key, value)

  return content

def measureOccupiedVolumeFraction(inputArray):
  '''This function measures the occupied volume (or area) fraction of a given filter.

  Inputs:
    inputArray -> 2D or 3D array. The input 2D or 3D array that we wish to measure the volume/area fraction of.
  Outputs:
    volFrac -> Float. The volume fraction of occupied (non-zero) elements in the array compared to 
                 total elements in the array.
  '''

  ### Measure the dimensions of the filter using the previously written function
  dimensions = measureFilterDimensions(inputArray)

  ### Get the product of the array to determine total number of elements in the depadded array
  totalElements = np.prod(dimensions)

  ### Get the number of nonzero elements in the depadded array
  numNonzero = np.count_nonzero(inputArray)

  ### Determine the volume fraction
  volFrac = float(numNonzero) / float(totalElements)

  return volFrac

def markPastedFilters(
      inputs,
      taMasked=None,
      ltMasked=None, 
      wtMasked=None, 
      taName="./myoimages/LossFilter.png",
      ltName="./myoimages/LongitudinalFilter.png",
      ttName="./myoimages/newSimpleWTFilter.png"
      ):
  '''
  Given masked stacked hits for the 3 filters and a doctored colored image, 
  function will paste filter sized boxes around the characterized regions
  and return the colored image with filter sized regions colored.

  NOTE: Colored image was read in (not grayscale) and 1 was subtracted from
  the image. This was necessary for the thresholding to work with the painter
  function
  '''
  ### Exploiting architecture of painter function to mark hits for me
  ###   To do this, we need to mimic the class structure from bankDetect results
  if inputs.dic['filterTypes']['TA']:
    TAholder = bD.ClassificationResults(
      correlated = None,
      stackedHits = taMasked,
      stackedAngles = None
    )
  if inputs.dic['filterTypes']['LT']:
    LTholder = bD.ClassificationResults(
      correlated = None,
      stackedHits = ltMasked,
      stackedAngles = None
    )
  if inputs.dic['filterTypes']['TT']:
    TTholder = bD.ClassificationResults(
      correlated = None,
      stackedHits = wtMasked,
      stackedAngles = None
    )

  ### load in filters to get filter dimensions
  if inputs.dic['filterTypes']['TA']:
    taFilt = LoadFilter(taName)
  if inputs.dic['filterTypes']['LT']:
    ltFilt = LoadFilter(ltName)
  if inputs.dic['filterTypes']['TT']:
    ttFilt = LoadFilter(ttName)

  ### get filter dimensions
  if inputs.dic['filterTypes']['TA']:
    TADimensions = measureFilterDimensions(taFilt)
  if inputs.dic['filterTypes']['LT']:
    LTDimensions = measureFilterDimensions(ltFilt)
  if inputs.dic['filterTypes']['TT']:
    TTDimensions = measureFilterDimensions(ttFilt)

  ### we want to mark WT last since that should be the most stringent
  # Opting to mark TA, then Long, then WT
  if inputs.dic['filterTypes']['TA']:
    labeledTA = painter.doLabel(TAholder,cellDimensions=TADimensions,thresh=0)
    if inputs.dic['dimensions'] == 3:
      print "Warning: Shifting TA hits down one index in the z domain to make consistent hit detection."
      dummy = np.zeros_like(labeledTA[:,:,0])
      labeledTA = np.dstack((dummy,labeledTA))[:,:,:-1]
  # else:
  #   labeledTA = np.zeros_like(taMasked,dtype=int)
  if inputs.dic['filterTypes']['LT']:
    labeledLT = painter.doLabel(LTholder,cellDimensions=LTDimensions,thresh=0)
    if inputs.dic['dimensions'] == 3:
      print "Warning: Shifting LT hits down one index in the z domain to make consistent hit detection."
      dummy = np.zeros_like(labeledLT[:,:,0])
      labeledLT = np.dstack((dummy,labeledLT))[:,:,:-1]
  # else:
  #   labeledLT = np.zeros_like(ltMasked,dtype=int)
  if inputs.dic['filterTypes']['TT']:
    labeledTT = painter.doLabel(TTholder,cellDimensions=TTDimensions,thresh=0)
  # else:
  #   labeledTT = np.zeros_like(wtMasked,dtype=int)

  ### perform masking
  if inputs.dic['filterTypes']['TA']:
    TAmask = labeledTA.copy()
  # else:
  #   TAmask = TAholder.stackedHits.astype(bool)
  if inputs.dic['filterTypes']['LT']:
    LTmask = labeledLT.copy()
  # else:
  #   LTmask = LTholder.stackedHits.astype(bool)
  if inputs.dic['filterTypes']['TT']:
    TTmask = labeledTT.copy()
  # else:
  #   TTmask = TTholder.stackedHits.astype(bool)

  if inputs.dic['filterTypes']['TA'] and inputs.dic['filterTypes']['TT']:
    TTmask[labeledTA] = False
  if inputs.dic['filterTypes']['TT'] and inputs.dic['filterTypes']['LT']:
    TTmask[labeledLT] = False
  if inputs.dic['filterTypes']['LT'] and inputs.dic['filterTypes']['TA']:
    LTmask[labeledTA] = False
  if inputs.dic['filterTypes']['LT'] and inputs.dic['filterTypes']['TT']:
    LTmask[TTmask] = False # prevents double marking of WT and LT

  ### Dampen brightness and mark hits
  markedImage = inputs.colorImage.copy()
  alpha = 1.0
  hitValue = int(round(alpha * 255))
  if inputs.dic['filterTypes']['TA']:
    markedImage[...,2][TAmask] = hitValue
  if inputs.dic['filterTypes']['LT']:
    markedImage[...,1][LTmask] = hitValue
  if inputs.dic['filterTypes']['TT']:
    markedImage[...,0][TTmask] = hitValue

  return markedImage

# Prepare matrix of vectorized of FFT'd images
def CalcX(
  imgs,
  debug=False
  ):
  nImg,d1,d2 = np.shape(imgs)
  dd = d1*d2  
  #print nImg, d2
  X = np.zeros([nImg,dd],np.dtype(np.complex128))
    
  for i,img in enumerate(imgs):
    xi = np.array(img,np.dtype(np.complex128))     
    # FFT (don't think I need to shift here)  
    Xi = fftp.fft2( xi )    
    if debug:
      Xi = xi    
    #myplot(np.real(Xi))
    # flatten
    Xif = np.ndarray.flatten(Xi)
    X[i,:]=Xif
  return X 

def GetAnnulus(region,sidx,innerMargin,outerMargin=None):
  if outerMargin==None: 
      # other function wasn't really an annulus 
      raise RuntimeError("Antiquated. See GetRegion")

  if innerMargin%2==0 or outerMargin%2==0:
      print "WARNING: should use odd values for margin!" 

  # grab entire region
  outerRegion,dummy,dummy = GetRegion(region,sidx,outerMargin)

  # block out interior to create annulus 
  annulus = np.copy(outerRegion) 
  s = np.shape(annulus)
  aM = outerMargin - innerMargin
  xMin,xMax = 0+aM, s[0]-aM
  yMin,yMax = 0+aM, s[1]-aM
  interior = np.copy(annulus[xMin:xMax,yMin:yMax])
  annulus[xMin:xMax,yMin:yMax]=0. 

  return annulus,interior

def CalcPSD(Hs): # fourier xformed data
    psd = np.real(np.conj(Hs)*Hs)
    eps = 1e-5
    psd[ psd < eps ] = eps
    return fftp.fftshift(np.log( psd ))

def dissimilar(
    theFilter,
    theDecoy,
    Cv = 1.,
    Clutter = 1.,
    beta = 0.,
    gamma = 0.0001):

    s = 1.
    s2 = 3.

    # filter FFT
    kernel = np.ones((s2,s2),np.float32)/np.float(s2*s2)
    h = np.array(theFilter,dtype=np.float)
    h = cv2.resize(h, None, fx = s, fy = s, interpolation = cv2.INTER_CUBIC)
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(h,cmap="gray")
    hs = fftp.fftshift(h)
    Hs = fftp.fftn(hs)
    plt.subplot(2,2,2)
    plt.imshow(CalcPSD(Hs))

    # noise FFT 
    Cn = Cv*np.ones_like(h,dtype=np.float)

    # clutter FFT
    Cc = np.ones_like(h,dtype=np.float)

    # difference image 
    p = cv2.resize(theDecoy, None, fx = s, fy = s, interpolation = cv2.INTER_CUBIC)
    p = cv2.filter2D(p,-1,kernel)
    k = p - h
    k[ k < 0] = 0
    plt.subplot(2,2,3)
    plt.imshow(k,cmap='gray')
    ks = fftp.fftshift(k)
    Ks = fftp.fftn(ks)
    print np.min(k), np.min(Ks)
    #Ks = cv2.filter2D(np.real(Ks),-1,kernel)
    #Ks = cv2.filter2D(np.real(Ks),-1,kernel)
    #Ks = cv2.filter2D(np.real(Ks),-1,kernel)    
    plt.subplot(2,2,4)
    plt.imshow(CalcPSD(Ks))

    ### modified filter
    Fs = Hs / (Cn + beta*Cc + gamma*np.abs(Ks))
    
    fs = fftp.ifftn(Fs)
    f  = fftp.ifftshift(fs)
    f-= np.min(f); f/=np.max(f); f*=255
    f = np.array(np.real(f),dtype=np.uint8)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(f,cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(CalcPSD(Fs))
    
    
    f = cv2.resize(f, None, fx = 1/s, fy = 1/s, interpolation = cv2.INTER_CUBIC)
    return f, Hs,Ks



###################################################################################################
###################################################################################################
###################################################################################################
###
### Setup/YAML Routines
###
###################################################################################################
###################################################################################################
###################################################################################################

def load_yaml(fileName):
  '''Function to load in the yaml file that is specified in 'fileName' and returns a dictionary'''

  if fileName:
    with open(fileName) as yam:
      data = yaml.load(yam)
  else:
    data = None
  
  return data

###################################################################################################
###################################################################################################
###################################################################################################
###
### Command Line Functionality
###
###################################################################################################
###################################################################################################
###################################################################################################

# Message printed when program run without arguments 
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: 
 
Usage:
"""
  msg+="  %s -validation" % (scriptName)
  msg+="""
  
 
Notes:

"""
  return msg

#
# MAIN routine executed when launching this script from command line 
#
if __name__ == "__main__":
  msg = helpmsg()
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):
    if(arg=="-genWT"):
      saveFixedWTFilter()
      quit()
    elif(arg=="-genSimpleWTFilter"):
      saveSimpleWTFilter()
      quit()
    elif(arg=="-genLoss"):
      saveFixedLossFilter()
      quit()
    elif(arg=="-genPunishment"):
      saveFixedPunishmentFilter()
      quit()
    elif(arg=="-genAll3DFilters"):
      scopeResolutions = [10,10,5]
      generate3DTTFilter(scopeResolutions)
      generate3DLTFilter(scopeResolutions)
      generate3DTAFilter(scopeResolutions)
      quit()
    elif(arg=="-genAllMyo"): 
      saveAllMyo()
      quit()
    elif(arg=="-genWeirdLong"):
      raise RuntimeError("WARNING: DEPRECATED. Use -genGaussLongFilter")
      #saveWeirdLongFilter()
      #quit()
    elif(arg=="-genSimpleLong"):
      raise RuntimeError("WARNING: DEPRECATED. Use -genGaussLongFilter")
      #saveSimpleLongFilter()
      #quit()
    elif(arg=="-genGaussLong"):
      saveGaussLongFilter()
      quit()
    elif(i>0):
      raise RuntimeError("Arguments not understood")
