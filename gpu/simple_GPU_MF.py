import sys
import cv2
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import time
import imutils
import scipy.fftpack as fftp
import util


def LoadImage(
  imgName = "/home/AD/pmke226/DataLocker/cardiac/Sachse/171127_tissue/tissue.tif",
  mid = 10000,
  maxDim = 100,
  angle = -35.    
):
    # read image 
    img = cv2.imread(imgName)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img, dtype=np.float)
    
    # extract subset 
    imgTrunc = img[
      (mid-maxDim):(mid+maxDim),
      (mid-maxDim):(mid+maxDim)]

    # rotate to align TTs while we are testing code 
    imgR = imutils.rotate(imgTrunc, angle)
    #cv2.imshow(imgR,cmap="gray")

    return imgR




# In[132]:

def MakeTestImage(dim = 100):
    l = np.zeros(2); l[0:0]=1.
    z = l
    # there are smarter ways of doing this 

    l = np.zeros(dim)
    for i in range(4):
      l[i::10]=1
    #print l

    striped = np.outer(np.ones(dim),l)
    #cv2.imshow(striped)
    imgR = striped
    return imgR

# ### Make filter
# - Measured about 35 degree rotation (clockwise) to align TT with y axis
# - z-lines are about 14 px apart
# - z-line about 3-4 pix thick
# - 14-20 px tall is probably reasonable assumption

# In[128]:

def MakeFilter(
    fw = 4,
    fdim = 14
  ): 
    dFilter = np.zeros([fdim,fdim])
    dFilter[:,0:fw]=1.
    # test 
    #dFilter[:] = 1 # basically just blur image
    dFilter = np.roll(dFilter,-np.int(fw/2.),axis=1)
    #cv2.imshow(dFilter,cmap="gray")
    
    return dFilter


# In[128]:




# In[136]:

def Pad(
    imgR,dFilter):
    fdim = np.shape(dFilter)[0]


    # big filter
    filterPadded = np.zeros_like( imgR)
    filterPadded[0:fdim,0:fdim] = dFilter
    hfw = np.int(fdim/2.)
    #print hfw

    xroll = np.roll(filterPadded,-hfw,axis=0)
    xyroll = np.roll(xroll,-hfw,axis=1)
    filt = xyroll
    #cv2.imshow(filt,cmap="gray")

    return filt

def padWithZeros(array, padwidth, iaxis, kwargs):
    array[:padwidth[0]] = 0
    array[-padwidth[1]:]= 0
    return array


##
## Tensor flow part 
##
def doTFloop(img,# test image
           mFs, # shifted filter
           nrots=4   # like filter rotations) 
           ):

  with tf.Session() as sess:
    # Create and initialize variables
    cnt = tf.Variable(tf.constant(nrots))
    specVar = tf.Variable(img, dtype=tf.complex64)
    filtVar = tf.Variable(mFs, dtype=tf.complex64)
    sess.run(tf.variables_initializer([specVar,filtVar,cnt]))

    # While loop that counts down to zero and computes reverse and forward fft's
    def condition(x,mf,cnt):
      return cnt > 0

    def body(x,mf,cnt):
      ## Essentially the matched filtering parts 
      xF  =tf.fft2d(x)
      xFc = tf.conj(xF)
      mFF =tf.fft2d(mf)
      out = tf.multiply(xFc,mFF) # elementwise multiplication for convolutiojn 
      xR  =tf.ifft2d(out)
      # DOESNT LIKE THIS xRr = tf.real(xR)
      ## ------

      cntnew=cnt-1
      return xR, mf,cntnew

    start = time.time()

    final, mfo,cnt= tf.while_loop(condition, body,
                              [specVar,filtVar,cnt], parallel_iterations=1)
    final, mfo,cnt =  sess.run([final,mfo,cnt])
    corr = np.real(final) 

    #start = time.time()
    tElapsed = time.time()-start
    print 'tensorflow:{}s'.format(tElapsed)


    return final, tElapsed


def MF(
    dImage,
    dFilter,
    useGPU=False
    ):

    
    if useGPU:
        # NOTE: I pass in an 'nrots' argument, but it doesn't actually do anything (e.g. 'some assembly required')
       corr,tElapsed = doTFloop(dImage,filt,nrots=1)
       corr = np.real(corr)
       #corr,tElapsed = doTFloop(dImage,dFilter,nrots=1)
       #corr = np.real(corr)
    else:        
       start = time.time()
       I = dImage
       T = filt
       fI = fftp.fftn(I)
       fT = fftp.fftn(T)
       c = np.conj(fI)*fT
       corr = fftp.ifftn(c)
       corr = np.real(corr)
       tElapsed = time.time()-start
       print 'CPU - fftp:{}s'.format(tElapsed)
       
    #cv2.imshow(dImage,cmap="gray")
    #plt.figure()
    # I'm doing something funky here, s.t. my output is rotated by 180. wtf
    corr = imutils.rotate(corr,180.)
    return corr    

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
def test0():
  testImage = MakeTestImage()
  dFilter = MakeFilter()
  corr = MF(testImage,dFilter,useGPU=True) 
  writer(corr)
  return testImage,corr

def test1(maxDim=100):
  testImage = LoadImage(maxDim=maxDim)
  dFilter = MakeFilter()
  corr = MF(testImage,dFilter,useGPU=True) 
  writer(corr)
  return testImage,corr

def gpuMF(testImage,dFilter,useGPU=True):
  corr = MF(testImage,dFilter,useGPU=True)
  #writer(corr)
  return corr
  


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
      quit()
    if(arg=="-test1"):
      test1()
      quit()
    if(arg=="-test2"):
      test1(maxDim=5000)
      quit()
  





  raise RuntimeError("Arguments not understood")




