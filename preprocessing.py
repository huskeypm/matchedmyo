###
### Group of functions that will walk the user fully through the preprocessing
### routines.
###
import sys
import os
import cv2
import util
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pygame, sys
from PIL import Image
pygame.init() # initializes pygame modules
from sklearn.decomposition import PCA
import imutils
import matchedFilter as mF

###############################################################################
###
### Normalization Routines
###
##############################################################################

def normalizeToStriations(img, subsectionIdxs,filterSize):
  '''
  function that will go through the subsection and find average smoothed peak 
  and valley intensity of each striation and will normalize the image 
  based on those values.
  '''

  print "Normalizing myocyte to striations"
  
  ### Load in filter that will be used to smooth the subsection
  WTfilterName = "./myoimages/singleTTFilter.png"
  WTfilter = util.ReadImg(WTfilterName,renorm=True)
  # divide by the sum so that we are averaging across the filter
  WTfilter /= np.sum(WTfilter)

  ### Perform smoothing on subsection
  smoothed = np.asarray(mF.matchedFilter(img,WTfilter,demean=False),dtype=np.uint8)

  ### Grab subsection of the smoothed image
  smoothedSubsection = smoothed.copy()[subsectionIdxs[0]:subsectionIdxs[1],
                                       subsectionIdxs[2]:subsectionIdxs[3]]
  #plt.figure()
  #plt.imshow(smoothedSubsection)
  #plt.colorbar()
  #plt.show()
  
  ### Perform Gaussian thresholding to pull out striations
  # blockSize is pixel neighborhood that each pixel is compared to
  blockSize = int(round(float(filterSize) / 3.57)) # value is empirical
  # blockSize must be odd so we have to check this
  if blockSize % 2 == 0:
    blockSize += 1
  # constant is a constant that is subtracted from each distribution for each pixel
  constant = 0
  # threshValue is the value at which super threshold pixels are marked, else px = 0
  threshValue = 1
  gaussSubsection = cv2.adaptiveThreshold(smoothedSubsection, threshValue,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, blockSize,
                                          constant)
  #plt.figure()
  #plt.imshow(gaussSubsection)
  #plt.colorbar()
  #plt.show()

  ### Calculate the peak and valley values from the segmented image
  peaks = smoothedSubsection[np.nonzero(gaussSubsection)]
  peakValue = np.mean(peaks)
  peakSTD = np.std(peaks)
  valleys = smoothedSubsection[np.where(gaussSubsection == 0)]
  valleyValue = np.mean(valleys)
  valleySTD = np.std(valleys)

  print "Average Striation Value:", peakValue
  print "Standard Deviation of Striation:", peakSTD
  print "Average Striation Gap Value:", valleyValue
  print "Stand Deviation of Striation Gap", valleySTD

  ### Calculate ceiling and floor thresholds empirically
  ceiling = peakValue + 3 * peakSTD
  floor = valleyValue - valleySTD
  if ceiling > 255:
    ceiling = 255.
  if floor < 0:
    floor = 0
  
  ceiling = int(round(ceiling))
  floor = int(round(floor))
  print "Ceiling Pixel Value:", ceiling
  print "Floor Pixel Value:", floor

  ### Threshold
  #img = img.astype(np.float64)
  #img /= np.max(img)  
  img[img>=ceiling] = ceiling
  img[img<=floor] = floor
  img -= floor
  img = img.astype(np.float64)
  img /= np.max(img)
  img *= 255
  img = img.astype(np.uint8)

  return img

###############################################################################
###
### FFT Filtering Routines
###
###############################################################################




###############################################################################
###
###  Reorientation Routines
###
###############################################################################

def autoReorient(img):
  '''
  Function to automatically reorient the given image based on principle component
    analysis. This isn't incorporated into the full, 'robust' preprocessing routine
    but is instead used for preprocessing the webserver uploaded images since 
    we want as little user involvement as possible.

  INPUTS:
    - img: The image that is to be reoriented. Uploaded as a float with 0 <= px <= 1.
  '''
  raise RuntimeError( "Broken for some reason. Come back to debug")

  dummy = img.copy()
  dumDims = np.shape(dummy)
  minDim = np.min(dumDims)
  maxDim = np.max(dumDims)
  argMaxDim = np.argmax(dumDims)
  diff = maxDim - minDim
  padding = int(round(diff / 2.))
  if argMaxDim == 1:
    padded = np.zeros((minDim+2*padding,maxDim))
    padded[padding:-padding,:] = dummy
  else:
    padded = np.zeros((maxDim,minDim+2*padding))
    padded[:,padding:-padding] = dummy
  plt.figure()
  plt.imshow(padded)
  plt.show()
  quit()
  

  pca = PCA(n_components=2)
  pca.fit(padded)
  majorAxDirection = pca.explained_variance_
  yAx = np.array([0,1])
  degreeOffCenter = (180./np.pi) * np.arccos(np.dot(yAx,majorAxDirection)\
                    / (np.linalg.norm(majorAxDirection)))

  print "Image is", degreeOffCenter," degrees off center"

  ### convert img to cv2 acceptable format
  acceptableImg = np.asarray(img * 255.,dtype=np.uint8)
  rotated = imutils.rotate_bound(acceptableImg, -degreeOffCenter)

  return rotated

def setup(array):
    #px = pygame.image.load(path)
    px = pygame.surfarray.make_surface(array)
    screen = pygame.display.set_mode( px.get_rect()[2:] )
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    return screen, px

def displayImageLine(screen, px, topleft, prior):
    # ensure that the rect always has positive width, height
    x, y = topleft
    xNew = pygame.mouse.get_pos()[0]
    yNew = pygame.mouse.get_pos()[1]
    width = xNew - x
    height = yNew - y
    if width < 0:
        x += width
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)

    # eliminate redundant drawing cycles (when mouse isn't moving)
    current = x, y, width, height
    if not (width and height):
        return current
    if current == prior:
        return current
    
    # draw line on the image
    red = (255, 0, 0)
    startPoint = topleft
    endPoint = (xNew,yNew)
    screen.blit(px,px.get_rect())
    pygame.draw.line(screen,red,startPoint,endPoint)
    pygame.display.flip()

    # return current box extents
    return (x, y, width, height)

def mainLoopLine(screen, px):
    topleft = bottomright = prior = None
    n=0
    while n!=1:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                if not topleft:
                    topleft = event.pos
                else:
                    bottomright = event.pos
                    n=1
        if topleft:
            prior = displayImageLine(screen, px, topleft, prior)
    return ( topleft + bottomright )

def giveSubsectionLine(array):
    # pygame has weird indexing
    newArray = np.swapaxes(array,0,1)
    screen, px = setup(newArray)
    pygame.display.set_caption("Draw a Line Orthogonal to Transverse Tubules")
    left, upper, right, lower = mainLoopLine(screen, px)
    pygame.display.quit()
    
    directionVector = (right-left,upper-lower)
    return directionVector

def reorient(img):

  '''
  Function to reorient the myocyte based on a user selected line that is
  orthogonal to the TTs 
  '''
  print "Reorienting Myocyte"

  ### get direction vector from line drawn by user
  dVect = giveSubsectionLine(img)

  ### we want rotation < 90 deg so we ensure correct axis
  if dVect[0] >= 0:
    xAx = [1,0]
  else:
    xAx = [-1,0]

  ### compute degrees off center from the direction vector
  dOffCenter = (180./np.pi) * np.arccos(np.dot(xAx,dVect)/np.linalg.norm(dVect))

  ### ensure directionality is correct 
  if dVect[1] <= 0:
    dOffCenter *= -1
  print "Image is",dOffCenter,"degrees off center"

  ### rotate image
  rotated = imutils.rotate_bound(img,dOffCenter)

  return rotated, dOffCenter

###############################################################################
###
### Resizing Routines
###
###############################################################################
def displayImage(screen, px, topleft, prior):
    # ensure that the rect always has positive width, height
    x, y = topleft
    width =  pygame.mouse.get_pos()[0] - topleft[0]
    height = pygame.mouse.get_pos()[1] - topleft[1]
    if width < 0:
        x += width
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)

    # eliminate redundant drawing cycles (when mouse isn't moving)
    current = x, y, width, height
    if not (width and height):
        return current
    if current == prior:
        return current

    # draw transparent box and blit it onto canvas
    screen.blit(px, px.get_rect())
    im = pygame.Surface((width, height))
    im.fill((128, 128, 128))
    pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
    im.set_alpha(128)
    screen.blit(im, (x, y))
    pygame.display.flip()

    # return current box extents
    return (x, y, width, height)

def mainLoop(screen, px):
    topleft = bottomright = prior = None
    n=0
    while n!=1:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                if not topleft:
                    topleft = event.pos
                else:
                    bottomright = event.pos
                    n=1
        if topleft:
            prior = displayImage(screen, px, topleft, prior)
    return ( topleft + bottomright )

def giveSubsection(array):
    # pygame has weird indexing
    newArray = np.swapaxes(array,0,1)
    screen, px = setup(newArray)
    pygame.display.set_caption("Draw a Rectangle Around Several Conserved Transverse Tubule Striations")
    left, upper, right, lower = mainLoop(screen, px)

    # ensure output rect always has positive width, height
    if right < left:
        left, right = right, left
    if lower < upper:
        lower, upper = upper, lower
    subsection = array.copy()[upper:lower,left:right]
    indexes = np.asarray([upper, lower, left, right])
    subsection = np.asarray(subsection, dtype=np.float64)
    pygame.display.quit()
    return subsection, indexes

def resizeToFilterSize(img,filterTwoSarcomereSize):
  '''
  Function to semi-automate the resizing of the image based on the filter
  '''

  print "Resizing myocyte based on user selected subsection"

  ### 1. Select subsection of image that exhibits highly conserved network of TTs
  subsection,indexes = giveSubsection(img)#,dtype=np.float32)
  # best to normalize the subsection for display purposes
  subsection /= np.max(subsection)

  ### 2. Resize based on the subsection
  resized, scale, newIndexes = resizeGivenSubsection(img,subsection,filterTwoSarcomereSize,indexes)

  print "Image Two Sarcomere Size:",scale
  
  return resized,scale,subsection,newIndexes

def resizeGivenSubsection(img,subsection,filterTwoSarcomereSize,indexes):
  '''
  Function to resize img given a subsection of the image
  '''
  ### Using this subsection, calculate the periodogram
  fBig, psd_Big = signal.periodogram(subsection)
  # finding sum, will be easier to identify striation length with singular dimensionality
  bigSum = np.sum(psd_Big,axis=0)

  ### Mask out the noise in the subsection periodogram
  # NOTE: These are imposed assumptions on the resizing routine
  maxStriationSize = 50.
  minStriationSize = 5.
  minPeriodogramValue = 1. / maxStriationSize
  maxPeriodogramValue = 1. / minStriationSize
  bigSum[fBig < minPeriodogramValue] = 0.
  bigSum[fBig > maxPeriodogramValue] = 0.

  display = False
  if display:
    plt.figure()
    plt.plot(fBig,bigSum)
    plt.title("Collapsed Periodogram of Subsection")
    plt.show()

  ### Find peak value of periodogram and calculate striation size
  striationSize = 1. / fBig[np.argmax(bigSum)]
  imgTwoSarcomereSize = int(round(2 * striationSize))
  print "Two Sarcomere size:", imgTwoSarcomereSize,"Pixels per Two Sarcomeres"

  if imgTwoSarcomereSize > 70 or imgTwoSarcomereSize < 10:
    print "WARNING: Image likely failed to be properly resized. Manual resizing",\
           "may be necessary!!!!!"

  ### Using peak value, resize the image
  scale = float(filterTwoSarcomereSize) / float(imgTwoSarcomereSize)
  resized = cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)

  ### Find new indexes in image
  newIndexes = indexes * scale
  newIndexes = np.round(newIndexes).astype(np.int32)

  return resized, scale, newIndexes



###############################################################################
###
### CLAHE Routines
###
###############################################################################

def applyCLAHE(img,filterTwoSarcomereSize):
  print "Applying CLAHE to Myocyte"

  kernel = (filterTwoSarcomereSize, filterTwoSarcomereSize)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=kernel)

  clahedImage = clahe.apply(img)

  return clahedImage
###############################################################################
###
### Main Routines
###
###############################################################################

def preprocess(fileName,filterTwoSarcomereSize):
  img = util.ReadImg(fileName)

  img,degreesOffCenter = reorient(img)
  img,resizeScale,subsection,idxs = resizeToFilterSize(img,filterTwoSarcomereSize)
  img = applyCLAHE(img,filterTwoSarcomereSize)
  img = normalizeToStriations(img,idxs,filterTwoSarcomereSize)

  # fix mask based on img orientation and resize scale
  try:
    processMask(fileName,degreesOffCenter,resizeScale)
  except:
    1

  # write file
  name,fileType = fileName[:-4],fileName[-4:]
  newName = name+"_processed"+fileType
  cv2.imwrite(newName,img)

  return img

def processMask(fileName,degreesOffCenter,resizeScale):
  '''
  function to reorient and resize the mask that was generated for the original
  image.
  '''
  maskName = fileName[:-4]+"_mask"+fileName[-4:]
  mask = util.ReadImg(maskName)
  reoriented = imutils.rotate_bound(mask,degreesOffCenter)
  resized = cv2.resize(reoriented,None,fx=resizeScale,fy=resizeScale,interpolation=cv2.INTER_CUBIC)
  cv2.imwrite(fileName[:-4]+"_processed_mask"+fileName[-4:],resized)

def preprocessTissue():
  '''
  Function to preprocess the original tissue level image
  '''

  #root = "/net/share/pmke226/DataLocker/cardiac/Sachse/171127_tissue/"
  root = "./myoimages/"
  fileName = "tissue.tif"


  ### read in image
  tissueImg = cv2.imread(root+fileName)
  tissueImg = cv2.cvtColor(tissueImg,cv2.COLOR_BGR2GRAY)

  ### rescale to filter size
  imgTwoSarcSize = 22
  filterTwoSarcSize = 25
  scale = float(filterTwoSarcSize) / float(imgTwoSarcSize)
  resizedTissueImg = cv2.resize(tissueImg,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)

  ### smooth the large image
  ## This caused a weird ringing effect so I'm opting to do this after the CLAHE
  #smoothedTissueImg = cv2.blur(resizedTissueImg,(3,3))

  ### applying a much more global CLAHE routine to kill dye imbalance
  tissueDims = np.shape(resizedTissueImg)
  claheTileSize = int(1./8. * tissueDims[0])
  CLAHEDtissueImg = applyCLAHE(resizedTissueImg,claheTileSize)

  ### smooth the CLAHED image
  kernelSize = (3,3)
  smoothedTissueImg = cv2.blur(CLAHEDtissueImg,kernelSize)

  ### apply an intensity ceiling and floor to apply contrast stretching
  floorValue = 6
  ceilingValue = 10
  clippedTissueImg = smoothedTissueImg
  clippedTissueImg[clippedTissueImg < floorValue] = floorValue
  clippedTissueImg[clippedTissueImg > ceilingValue] = ceilingValue
  clippedTissueImg -= floorValue
  clippedTissueImg = clippedTissueImg.astype(np.float32)
  clippedTissueImg *= 255. / np.max(clippedTissueImg)
  clippedTissueImg = clippedTissueImg.astype(np.uint8)

  ### save image
  cv2.imwrite("./myoimages/preprocessedTissue.png",clippedTissueImg)

def preprocessAll():
  '''
  function meant to preprocess all of the images needed for data reproduction
  '''
  root = './myoimages/'
  imgNames = ["HF_1.png", 
              "MI_D_73.png",
              "MI_D_76.png",
              "MI_D_78.png",
              "MI_M_45.png",
              "MI_P_16.png",
              "Sham_11.png",
              "Sham_M_65.png"]
  for name in imgNames:
    filterTwoSarcomereSize = 25
    # perform preprocessing on image
    preprocess(root+name,filterTwoSarcomereSize)
  
  ### now we preprocess the tissue image
  preprocessTissue()

def preprocessDirectory(directoryName,filterTwoSarcomereSize=25):
  '''
  function for preprocessing an entire directory of images
  '''
  for name in os.listdir(directoryName):
    preprocess(directoryName+name,filterTwoSarcomereSize)


###############################################################################
###
### Execution of File
###
###############################################################################

#
# Message printed when program run without arguments 
#
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
tag = "default_"
if __name__ == "__main__":
  msg = helpmsg()
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):

    ### Main routine to run
    if (arg=="-preprocess"):
      fileName = str(sys.argv[i+1])
      try:
        filterTwoSarcomereSize = sys.argv[i+2]
      except:
        filterTwoSarcomereSize = 25
      preprocess(fileName,filterTwoSarcomereSize)
      quit()
    if (arg=="-preprocessTissue"):
      preprocessTissue()
      quit()
    if (arg=="-preprocessAll"):
      preprocessAll()
      quit()
    if (arg=="-preprocessDirectory"):
      directory = str(sys.argv[i+1])
      preprocessDirectory(directory)
      quit()
      
  raise RuntimeError("Arguments not understood")

