import sys
import os
import matplotlib.pylab as plt 
import numpy as np 
import cv2
import scipy
import scipy.signal as sig
import scipy.fftpack as fftp
import imutils
import operator
import tensorflow as tf
 

root = "myoimages/"

def myplot(img,fileName=None,clim=None):
  plt.axis('equal')
  plt.pcolormesh(img, cmap='gray')
  plt.colorbar()
  if fileName!=None:
    plt.gcf().savefig(fileName,dpi=300)
  if clim!=None:
    plt.clim(clim)

def ReadImg(fileName,cvtColor=True,renorm=False,bound=False):
    img = cv2.imread(fileName)
    if img is None:
        raise RuntimeError(fileName+" likely doesn't exist")
    if cvtColor:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if bound!=False:
      img=img[bound[0]:bound[1],bound[0]:bound[1]]
    if renorm:# rescaling
      img = img / np.float(np.amax(img))
    return img  

def LoadFilter(fileName):
  filterImg = ReadImg(fileName,cvtColor=True,renorm=True).astype(np.float64)
  filterImg /=  np.sum(filterImg)
  return filterImg

def measureFilterDimensions(grayFilter):
  '''
  Measures where the filter has any data and returns a minimum bounding
  rectangle around the filter. To be used in conjunction with the 
  pasteFilters flag in DataSet
  '''
  filtery,filterx = np.shape(grayFilter)

  collapsedRows = np.sum(grayFilter,0)
  leftPadding = np.argmax(collapsedRows>0)
  rightPadding = np.argmax(collapsedRows[::-1]>0)
  numRows = filtery - leftPadding - rightPadding

  collapsedCols = np.sum(grayFilter,1)
  topPadding = np.argmax(collapsedCols>0)
  bottomPadding = np.argmax(collapsedCols[::-1]>0)
  numCols = filterx - topPadding - bottomPadding

  print "filter y,x:",numRows,numCols

  return numRows, numCols

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

def viewer(tensor):
    def dummy(tensor2):
        return tensor2

    with tf.Session() as sess:
        result = sess.run(dummy(tensor))
        print np.shape(result)
        print result
        return result

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

def renorm(img,scale=255):
    img = img-np.min(img)
    img/= np.max(img)
    img*=scale 
    return img

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

def GetRegion(region,sidx,margin):
      subregion = region[(sidx[0]-margin):(sidx[0]+margin+1),
                         (sidx[1]-margin):(sidx[1]+margin+1)]        
      area = np.float(np.prod(np.shape(subregion)))
      intVal = np.sum(subregion)  
      return subregion, intVal, area

def MaskRegion(region,sidx,margin,value=0):
      region[(sidx[0]-margin):(sidx[0]+margin+1),
                         (sidx[1]-margin):(sidx[1]+margin+1)]=value  


def ApplyCLAHE(grayImgList, tileGridSize, clipLimit=2.0, plot=False):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahedImage = clahe.apply(grayImgList) # stupid hack
    return clahedImage

### Generating filters
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

def PasteFilter(img, filt):
  '''
  function to paste the filter in the upper left region of the img 
  to make sure they are scaled correctly
  '''
    
  myImg = img.copy()
  filtDim = np.shape(filt)
  myImg[:filtDim[0],:filtDim[1]] = filt
  return myImg

def saveAllMyo():
      saveFixedWTFilter()
      saveSimpleWTFilter()
      saveGaussLongFilter()
      saveFixedLossFilter()
      saveFixedPunishmentFilter()
      saveSingleTTFilter()
      saveSingleTTPunishmentFilter()

# Embegs signal into known iomage for testing 
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
if __name__ == "__main__":
  msg = helpmsg()
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):
    if(arg=="-genWT"):
      saveFixedWTFilter()
    elif(arg=="-genSimpleWTFilter"):
      saveSimpleWT()
    elif(arg=="-genLoss"):
      saveFixedLossFilter()
    elif(arg=="-genPunishment"):
      saveFixedPunishmentFilter()
    elif(arg=="-genAllMyo"): 
      saveAllMyo()
    elif(arg=="-genWeirdLong"):
      print "WARNING: DEPRECATED. Use -genGaussLongFilter"
      saveWeirdLongFilter()
      quit()
    elif(arg=="-genSimpleLong"):
      print "WARNING: DEPRECATED. Use -genGaussLongFilter"
      saveSimpleLongFilter()
      quit()
    elif(arg=="-genGaussLong"):
      saveGaussLongFilter()
      quit()
    elif(i>0):
      raise RuntimeError("Arguments not understood")
