import sys
import os 
import cv2
import matplotlib
#matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import time
import scipy.fftpack as fftp
import util
import imutils
import optimizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import cPickle as Pickle

height = 50
#iters = [-30,-15,0,15,30]
iters = [-10,0]

class empty:
  pass


#import imutils
def LoadImage(
  imgName = "/home/AD/pmke226/DataLocker/cardiac/Sachse/171127_tissue/tissue.tif",
  mid = 10000,
  maxDim = 100,
  angle = -35.    
):
    # read image 
    img = np.array(util.ReadImg(imgName),dtype=np.float64)
    
    # extract subset 
    imgTrunc = img[
      (mid-maxDim):(mid+maxDim),
      (mid-maxDim):(mid+maxDim),(mid-maxDim):(mid+maxDim)]

    # rotate to align TTs while we are testing code 
    #imgR = imutils.rotate(imgTrunc, angle)
    #cv2.imshow(imgR,cmap="gray")

    return imgR




# In[132]:

def MakeTestImage(dim = 400,height = height):
    l = np.zeros(2); l[0:0]=1.
    z = l
    # there are smarter ways of doing this 
    dim = int(dim)
    l = np.zeros(dim)
    for i in range(4):
      l[i::10]=1
    #print l

    striped = np.outer(np.ones(dim),l)
    #cv2.imshow(striped)
    img2 = striped

    if height != 0:
      height3 = np.ones((height))
      cross = np.outer(img2, height3)
      imgR = np.reshape(cross,(dim,dim,height))
      imgR[:,(dim/2):,:(height/2)] = 0
    else:
      imgR = img2
      imgR[:,(dim/2):] = 0

    #print "Made test Image"
    return imgR

# ### Make filter
# - Measured about 35 degree rotation (clockwise) to align TT with y axis
# - z-lines are about 14 px apart
# - z-line about 3-4 pix thick
# - 14-20 px tall is probably reasonable assumption

# In[128]:

def MakeFilter(
    fw = 4,
    fdim = 14,
    height=10
  ): 
    if height != 0:
      dFilter = np.zeros([fdim,fdim,height])
      dFilter[:,0:fw,0:fw]=1.
      # test 
      #dFilter[:] = 1 # basically just blur image
      yFilter = np.roll(dFilter,-np.int(fw/2.),axis=1)
      #cv2.imshow(dFilter,cmap="gray")
    else:
      dFilter = np.zeros([fdim,fdim])
      dFilter[0:fw,0:fw] = 1.
      yFilter = np.roll(dFilter,-np.int(fw/2.),axis=1)
    
    return dFilter


# In[128]:




# In[136]:

def Pad(imgR,
        dFilter):

    fdim = np.shape(dFilter)
    imgDim = np.shape(imgR)
    if len(imgDim) == 3:
      assert (fdim[2] <= imgDim[2]), "Your filter is larger in the z direction than your image"
    filterPadded = np.zeros_like( imgR)
    if len(imgDim) == 3:
      filterPadded[0:fdim[0],0:fdim[1],0:fdim[2]] = dFilter
      hfwz = np.int(fdim[2]/2.)
    else:
      filterPadded[0:fdim[0],0:fdim[1]] = dFilter
    hfw = np.int(fdim[0]/2.)
    hfwx = np.int(fdim[1]/2.)

    # we do this rolling here in the same way we shift for FFTs
    yroll = np.roll(filterPadded,-hfw,axis=0)
    xyroll = np.roll(yroll,-hfwx,axis=1)
    if len(imgDim) == 3:
      xyzroll = np.roll(xyroll,-hfwz, axis=2)
    else:
      xyzroll = xyroll

    # I don't think you need to shift this at all
    #xyzroll = filterPadded

    return xyzroll

def tfMF(img,mf,dimensions=3):
  '''
  Generic workhorse routine that sets up the image and matched filter for 
    matched filtering using tensorflow. This routine is called by whichever
    detection scheme has been chosen. NOT by the doTFloop explicitly.

  INPUTS:
    img - 'tensorized' image
    mf - rotated and 'tensorized' matched filter same dimension as img
  '''

  if dimensions == 3:
    xF = tf.fft3d(img)
    mFF = tf.fft3d(mf)
    # this is something to do with the shifting done previously
    xFc = tf.conj(xF)
    out = tf.multiply(xF,mFF)
    xR = tf.ifft3d(out)
  else:
    xF = tf.fft2d(img)
    mFF = tf.fft2d(mf)
    # this is something to do with the shifting done previously
    #xFc = tf.conj(xF)
    out = tf.multiply(xF,mFF)
    xR = tf.ifft2d(out)

  return xR

def convertToTFFilter(imgOrig,mfOrig):
  '''
  Routine to take CPU computation ready filter and convert to a filter ready for
    calculation with tensorflow
  '''
  tfFilt = tf.constant(mfOrig)
  ## sticks filter in upper left corner of padded image with same shape as original image
  paddings = tf.constant([[0,imgOrig.shape[0]-mfOrig.shape[0]],
                          [0,imgOrig.shape[1]-mfOrig.shape[1]]])
  padFilt = tf.pad(tfFilt,paddings,"CONSTANT")
  ## roll the filter in comparable way to FFT shifting
  #rollFilt = tf.manip.roll(padFilt,shift=[-mfOrig.shape[0]/2,-mfOrig.shape[1]/2],axis=[0,1])
  ## roll the filter to the middle of the image so that we can apply rotations later on
  #shiftY = imgOrig.shape[0]/2 - mfOrig.shape[0]/2
  #shiftX = imgOrig.shape[1]/2 - mfOrig.shape[1]/2
  shiftY = -mfOrig.shape[0]/2
  shiftX = -mfOrig.shape[1]/2
  rollFilt = tf.manip.roll(padFilt,shift=[shiftY,shiftX],axis=[0,1])
  pF = tf.cast(rollFilt, dtype=tf.complex64)
  tfFilt = tf.Variable(pF, dtype=tf.complex64)

  return tfFilt


##
## Tensor flow part 
##
def doTFloop(inputs,
           #img,# test image
           #mFs, # shifted filter
           paramDict,
           xiters=[0],
           yiters=[0],
           ziters=[0]
           ):
  '''
  Function that performs filtering across all rotations of a given filter
    using tensorflow routines and GPU speedup
  '''

  # TODO: Once TF 1.9 is out and stable, change complex data types to 128 instead of 64
  #       I believe there is some small error arising from this


  # We may potentially want to incorporate all 3 filters into this low level
  # loop to really speed things up
  
  with tf.Session() as sess:
    start = time.time()

    ### Create and initialize variables
    tfImg = tf.Variable(inputs.imgOrig, dtype=tf.complex64)
    tfFilt = convertToTFFilter(inputs.imgOrig,inputs.mfOrig)

    if paramDict['inverseSNR']:
      # if we are using an inverse threshold, we need a storage container that contains pixels > thresh
      stackedHitsDummy = np.ones_like(inputs.imgOrig) * 2. * paramDict['snrThresh']
    else:
      # otherwise, we need a container that contains pixels < thresh
      stackedHitsDummy = np.zeros_like(inputs.imgOrig)
    stackedHits = tf.Variable(stackedHitsDummy, dtype=tf.float64)

    # create an array for best angle storage = -1 so we know where no hits were found
    bestAngles = np.zeros_like(inputs.imgOrig) - 1.
    bestAngles = tf.Variable(bestAngles,dtype=tf.float64)
    snr = tf.Variable(np.zeros_like(tfImg),dtype=tf.complex64)

    # make big angle container
    numXits = np.shape(xiters)[0]
    numYits = np.shape(yiters)[0]
    numZits = np.shape(ziters)[0]
    cnt = tf.Variable(tf.constant(numXits * numYits * numZits-1,dtype=tf.int64))

    # It's late and I'm getting lazy
    bigIters = []
    if len(np.shape(inputs.imgOrig)) == 3:
      for i in xiters:
        for j in yiters:
          for k in ziters:
            bigIters.append([i,j,k])
    else:
      bigIters = ziters

    # convert iterations from degrees into radians
    bigIters = np.asarray(bigIters,dtype=np.float32)
    bigIters = bigIters * np.pi / 180.

    # have to convert to a tensor so that the rotations can be indexed during tf while loop
    bigIters = tf.Variable(tf.convert_to_tensor(bigIters,dtype=tf.float32))

    # initialize paramDict variables
    paramDict['inverseSNRTF'] = tf.Variable(paramDict['inverseSNR'], dtype=tf.bool)

    # set up filtering variables
    if paramDict['filterMode'] == 'punishmentFilter':
      paramDict['mfPunishment'] = convertToTFFilter(inputs.imgOrig,paramDict['mfPunishment'])
      paramDict['covarianceMatrix'] = tf.Variable(paramDict['covarianceMatrix'],dtype=tf.complex64)
      paramDict['gamma'] = tf.Variable(paramDict['gamma'],dtype=tf.complex64)
      sess.run(tf.variables_initializer([paramDict['mfPunishment'],paramDict['covarianceMatrix'],paramDict['gamma']]))

    sess.run(tf.variables_initializer([tfImg,tfFilt,cnt,bigIters,stackedHits,bestAngles,snr,paramDict['inverseSNRTF']]))

    inputs.tfImg = tfImg
    inputs.tfFilt = tfFilt
    inputs.bigIters = bigIters


    # While loop that counts down to zero and computes reverse and forward fft's
    def condition(cnt,stackedHits,bestAngles):
      return cnt >= 0

    def body3D(cnt,stackedHits,bestAngles):
      # pick out rotation to use
      rotations = bigIters[cnt]

      # rotating matched filter to specific angle
      rotatedMF = util.rotateFilterCube3D(inputs.tfFilt,
                                          rotations[0],
                                          rotations[1],
                                          rotations[2])

      # get detection/snr results
      snr = doDetection(inputs,paramDict,dimensions=3)
      stackedHitsNew,bestAnglesNew = doStackingHits(inputs,paramDict,stackedHits,bestAngles,snr,cnt)
      #stackedHitsNew = stackedHits

      cntnew=cnt-1
      return cntnew,stackedHitsNew,bestAnglesNew

    def body2D(cnt,stackedHits,bestAngles):
      rotation = bigIters[cnt]
 
      ### center filter for rotation scheme
      shiftY = inputs.imgOrig.shape[0]/2
      shiftX = inputs.imgOrig.shape[1]/2
      inputs.rotatedMF = tf.manip.roll(inputs.tfFilt,shift=[shiftY,shiftX],axis=[0,1])
      ### rotate filter
      inputs.rotatedMF = util.rotateTFFilter2D(inputs.rotatedMF,rotation)
      ### shift filter again
      inputs.rotatedMF = tf.manip.roll(inputs.rotatedMF,shift=[shiftY,shiftX],axis=[0,1])
      ### shift the rotated mf
      #inputs.rotatedMF = tf.manip.roll(inputs.rotatedMF,
      #        shift=[inputs.imgOrig.shape[0],inputs.imgOrig.shape[1]],axis=[0,1])

      if paramDict['filterMode'] == "punishmentFilter":
        ### center filter for rotation scheme
        inputs.rotatedPunishment = tf.manip.roll(paramDict['mfPunishment'],shift=[shiftY,shiftX],axis=[0,1])
        inputs.rotatedPunishment = util.rotateTFFilter2D(inputs.rotatedPunishment,rotation)
        inputs.rotatedPunishment = tf.manip.roll(inputs.rotatedPunishment,shift=[shiftY,shiftX],axis=[0,1])

        ### shift the rotated punishment mf
        #inputs.rotatedPunishment = tf.manip.roll(inputs.rotatedPunishment,
        #        shift=[inputs.imgOrig.shape[0],inputs.imgOrig.shape[1]],axis=[0,1])

      snr = doDetection(inputs,paramDict,dimensions=2)
      stackedHitsNew,bestAnglesNew = doStackingHits(inputs,paramDict,stackedHits,bestAngles,snr,cnt)
      cntnew=cnt-1
      return cntnew,stackedHitsNew,bestAnglesNew

    # can we optimize parallel_iterations based on memory allocation?
    '''
    NOTE: when images get huge (~>512x512x50) and we use parallel iterations=10, they eat a lot of memory
          since the intermediate results from the while loop are stored for back propogation.
          This memory consumption can be offloaded if we use swap_memory=True in the tf.while_loop.
          However, this slows down the calculation immensely, so we want to always cleverly pick
          the sweet spot between NOT offloading the memory eating tensors and NOT bricking the computer.
    '''
    # TODO: See if there is a way to grab maximum available memory for the GPU to automatically
    #       and cleverly determine the sweetspot to where we don't have to offload tensors to cpu
    #       but can also efficiently determine parallel_iterations number
    if len(np.shape(inputs.imgOrig)) == 3:
      cnt,stackedHits,bestAngles = tf.while_loop(condition, body3D,
                                      [cnt,stackedHits,bestAngles], parallel_iterations=10)
    else:
      cnt,stackedHits,bestAngles = tf.while_loop(condition, body2D,
                                      [cnt,stackedHits,bestAngles], parallel_iterations=10)

    compStart = time.time()
    cntF,stackedHitsF,bestAnglesF =  sess.run([cnt,stackedHits,bestAngles])
    compFin = time.time()
    print "Time for tensor flow to execute run:{}s".format(compFin-compStart)

    results = empty()
    
    results.stackedHits = stackedHitsF
    # TODO: pull out best angles from bigIters
    results.stackedAngles = bestAnglesF

    #if paramDict['inverseSNR']:
    #  bestAnglesF[stackedHitsF > paramDict['snrThresh']] = -1
    #else:
    #  bestAnglesF[stackedHitsF < paramDict['snrThresh']] = -1


    #start = time.time()
    tElapsed = time.time()-start

    ### something weird is going on and image is flipped in each direction


    print 'Total time for tensorflow to run:{}s'.format(tElapsed)


    return results, tElapsed


def MF(
    dImage,
    dFilter,
    useGPU=False,
    dim=2,
    xiters=[0],
    yiters=[0],
    ziters=[0]
    ):
    # NOTE: May need to do this padding within tensorflow loop itself to 
    #       ameliorate latency due to loading large matrices into GPU.
    #       Potentially a use for tf.dynamic_stitch?
    lx = len(xiters); ly = len(yiters); lz = len(ziters)
    numRots = lx * ly * lz
    filt = Pad(dImage,dFilter)

    inputs = empty()
    inputs.imgOrig = dImage
    inputs.mfOrig = filt
    if useGPU:
      # nothing special about picking WT, just need a param Dictionary to run the alg
      paramDict = optimizer.ParamDict(typeDict='WT')
      paramDict['filterMode'] = 'simple'
      corr,tElapsed = doTFloop(inputs,paramDict,xiters=xiters,yiters=yiters,ziters=ziters)
      corr = np.real(corr)
    else:        
      if dim == 3:
        corr,tElapsed = MFSubroutine3D(dImage,filt,xiters,yiters,ziters)
      else:
        corr,tElapsed = MFSubroutine2D(dImage,filt,ziters)

      print 'fftp:{}s'.format(tElapsed)
       
    return corr,tElapsed    

def MFSubroutine3D(img,filt,xiters,yiters,ziters):
  '''
  subroutine to perform 3D CPU matched filtering
  '''

  start = time.time()
  filtTF = tf.convert_to_tensor(filt)
  for i,x in enumerate(xiters):
    ii = tf.constant(i,dtype=tf.float64)
    for j,y in enumerate(yiters):
      jj = tf.constant(j,dtype=tf.float64)
      for k,z in enumerate(ziters):
        print "Filtering Progress:", str((i*ly*lz)+(j*lz)+k)+'/'+str(numRots)
        # rotate filter using tensorflow routine anyway
        # NOTE: I may change this
        kk = tf.constant(k,dtype=tf.float64)
        filtRot = util.rotateFilterCube3D(filtTF,ii,jj,kk)
        filtRot = tf.Session().run(filtRot)
        I = img
        T = filtRot
        fI = fftp.fftn(I)
        fT = fftp.fftn(T)
        c = np.conj(fI)*fT
        corr = fftp.ifftn(c)
        corr = np.real(corr)
  tElapsed = time.time()-start
  print 'fftp:{}s'.format(tElapsed)

  return corr,tElapsed


def MFSubroutine2D(img,filt,iters):
  '''
  subroutine to perform 2D CPU matched filtering
  '''

  start = time.time()
  for i,x in enumerate(iters):
    print "Filtering Progress", str(i)+'/'+str(len(iters))
    filtRot = imutils.rotate(filt,x)
    I = img
    T = filtRot
    fI = fftp.fft2(I)
    fT = fftp.fft2(T)
    c = np.conj(fI) * fT
    corr = fftp.ifft2(c)
    corr = np.real(corr)
  tElapsed = time.time()-start
  print 'fftp:{}s'.format(tElapsed)

  return corr,tElapsed


###################################################################################################
###
### Detection Schemes: taken from detection_protocols.py and 'tensorized'
###                    See detection_protocols.py for further documentation of functions
###################################################################################################

def doDetection(inputs,paramDict,dimensions=3):
  '''
  Function to route all tf detection schemes through.

  inputs - a class structure containing all necessary inputs
  paramDict - a dictionary containing your parameters

  These should follow the basic structure for the non TF calculations, but just
    'tensorized'
  '''

  mode = paramDict['filterMode']
  if mode=="lobemode":
    print "Tough luck, this isn't supported yet. Bug the developers to implement this. Quitting program"
    quit()
  elif mode=="punishmentFilter":
    results = punishmentFilterTensor(inputs,paramDict,dimensions=dimensions)
  elif mode=="simple":
    results = simpleDetectTensor(inputs,paramDict,dimensions=dimensions)
  elif mode=="regionalDeviation":
    results =regionalDeviationTensor(inputs,paramDict,dimensions=dimensions)
  elif mode=="filterRatio":
    print "Ignoring this for now. Bug the developers to implement if you want this detection scheme."
    quit()
  else:
    print "That mode isn't understood. Returning image instead."
    results = inputs.img

  return results

def punishmentFilterTensor(inputs,paramDict,dimensions=3):
  # call generalized tensorflow matched filter routine
  corr = tfMF(inputs.tfImg,inputs.rotatedMF,dimensions=dimensions)
  corrPunishment = tfMF(inputs.tfImg,inputs.rotatedPunishment,dimensions=dimensions)
  # calculate signal to noise ratio
  snr = tf.divide(corr,tf.add(paramDict['covarianceMatrix'],tf.multiply(paramDict['gamma'], corrPunishment)))
  return snr

def simpleDetectTensor(inputs,paramDict,dimensions=3):
  snr = tfMF(inputs.tfImg,inputs.rotatedMF,dimensions=dimensions)
  return snr


def regionalDeviationTensor(inputs,paramDict,dimensions=3):
  ### Perform simple detection
  img = inputs.tfImg
  mf = inputs.rotatedMF
  corr = tfMF(img, mf,dimensions=dimensions)

  ### construct new kernel for standard deviation calculation
  # find where filter > 0
  kernelLocs = tf.greater(tf.cast(mf,tf.float64),0.)
  kernel = tf.where(kernelLocs,tf.ones_like(mf),tf.zeros_like(mf))
  kernel = tf.cast(kernel,tf.complex64)

  ### construct array that contains number of elements in each window
  n = tfMF(tf.ones_like(img),kernel,dimensions=dimensions)

  ### square image for standard deviation calculation
  imgSquared = tf.square(img)

  ### Calculate standard deviation
  s = tfMF(img,kernel,dimensions=dimensions)
  q = tfMF(imgSquared,kernel,dimensions=dimensions)
  stdDev = tf.sqrt( tf.divide( tf.subtract(q, tf.divide(tf.square(s),n)), tf.subtract(n,1)))

  ### since this is dually conditional, I'll have to mask out the super threshold std Dev hits
  stdDevHits = tf.less(tf.cast(stdDev,tf.float64),paramDict['stdDevThresh'])
  maskedOutHits = tf.where(stdDevHits,corr,tf.zeros_like(corr))

  return maskedOutHits


def doStackingHits(inputs,paramDict,stackedHits,bestAngles,snr,cnt):
  '''
  Function to threshold the calculated snr and apply to the stackedHits container
  '''
  snr = tf.cast(tf.real(snr),dtype=tf.float64)

  # check inverse snr toggle, if true, find snr < stackedHits, if false, find snr > stackedHits
  snrHits = tf.cond(paramDict['inverseSNRTF'], 
                    lambda: tf.less(snr,stackedHits),
                    lambda: tf.greater(snr,stackedHits))
  stackedHits = tf.where(snrHits,snr,stackedHits)
  cntHolder = tf.multiply(tf.ones_like(stackedHits), tf.cast(cnt,tf.float64))
  bestAngles = tf.where(snrHits,cntHolder,bestAngles)
  return stackedHits,bestAngles

###################################################################################################
###
###  End detection schemes
###
###################################################################################################

def writer(testImage,name="out.tif"):
    # rescale
    img = testImage
    imgR = img - np.min(img)
    imgN = imgR/np.max(imgR)*(2**8  - 1 ) # for 8bit channel
    
    
    # convert to unsigned image 
    out = np.array(imgN,dtype=np.uint8)
    cv2.imwrite(name,out)
    #plt.pcolormesh(out)
    #plt.gcf().savefig("x.png")

    
    



#!/usr/bin/env python
##################################
#
# Revisions
#       10.08.10 inception
#
##################################

#
# ROUTINE  
#
def test0(useGPU=True):
  testImage = MakeTestImage()
  dFilter = MakeFilter()
  corr = MF(testImage,dFilter,useGPU=useGPU,dim=3,xiters=iters,yiters=iters,ziters=iters) 
  #writer(corr)
  return testImage,corr

def runner(dims):
  times =[]
  f = open("GPU_Benchmark.txt","w")
  f.write('Dims:{}'.format(dims))
  f.write('\nCPU:')
  for i,d in enumerate(dims):
    print "dim", d
    testImage = MakeTestImage(d)
    dFilter = MakeFilter()
    corr,time = MF(testImage,dFilter,useGPU=False,dim=3,xiters=iters,yiters=iters,ziters=iters)
    times.append(time)
    #if dim == dims[-1]:
    #f.write('\ndim:{},time:{};'.format(dim,time))
    #f.write('dim:{},time:{};'.format(dim,time))
  f.write('{}'.format(times))

  timesGPU =[]
  f.write('\nGPU:') 
  for j,d in enumerate(dims):
    print "dim", d
    testImage = MakeTestImage(d)
    dFilter = MakeFilter()
    corr,time = MF(testImage,dFilter,useGPU=True,dim=3,xiters=iters,yiters=iters,ziters=iters)
    timesGPU.append(time)
    #f.write('\ndim:{},time:{};'.format(dim,time))
  f.write('{}'.format(timesGPU))

  #Results = { "CPU":"%s"%(str(times)),"GPU":"%s"%str(timesGPU)}
  #pickle.dump(Results, open("%Benchmark.p"%(str(R),str(length/nm),str(cKCl)),"wb"))
    
  return times, timesGPU

def runner2D(dims,iters=iters):
  '''
  Same basic function as runner but this is for 2D benchmarking
  '''

  times =[]
  f = open("GPU_Benchmark.txt","w")
  f.write('Dims:{}'.format(dims))
  f.write('\nCPU:')
  for i,d in enumerate(dims):
    print "dim", d
    testImage = MakeTestImage(d,height=0)
    dFilter = MakeFilter(height=0)
    corr,time = MF(testImage,dFilter,useGPU=False,dim=2,ziters=iters)
    times.append(time)
  f.write('{}'.format(times))

  timesGPU =[]
  f.write('\nGPU:')
  for j,d in enumerate(dims):
    print "dim", d
    testImage = MakeTestImage(d,height=0)
    dFilter = MakeFilter(height=0)
    corr,time = MF(testImage,dFilter,useGPU=True,dim=2,ziters=iters)
    timesGPU.append(time)
  f.write('{}'.format(timesGPU))

  return times, timesGPU


def test1(maxDim=100):
  testImage = LoadImage(maxDim=maxDim)
  dFilter = MakeFilter()
  corr = MF(testImage,dFilter,useGPU=True,dim=3,xiters=iters,yiters=iters,ziters=iters) 
  writer(corr)
  return testImage,corr, dFilter




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

  #fileIn= sys.argv[1]
  #if(len(sys.argv)==3):
  #  1
  #  #print "arg"

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):
    # calls 'test0' with the next argument following the argument '-validation'
    if(arg=="-validation"):
      test0()      
      test0(useGPU = False)
      quit()


    if(arg=="-test1"):
      test1()
      quit()
    if(arg=="-test2"):
      test1(maxDim=5000)
      quit()
    if(arg=="-running"):
      #dims = [5,6,7,8,9,10]
      dims = [6,7]
      dims = map(lambda x: 2**x,dims)
      times,timesGPU = runner(dims)
      print "CPU", times
      print"\n" + "GPU",timesGPU
      quit()

    if(arg=="-2DBenchmark"):
      dims = np.arange(5,13)
      dims = map(lambda x: 2**x,dims)
      iters = [-25,-20,-15,-10,-5,0,5,10,15,20,25]
      CPUtimes,GPUtimes = runner2D(dims,iters=iters)
      print "CPU Times:",CPUtimes
      print "\n" + "GPU Times:", GPUtimes


      quit()
  





  raise RuntimeError("Arguments not understood")




