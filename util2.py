"""
Temporary home for routines while I get some code working
PKH
"""


import matplotlib.pylab as plt
import numpy as np
import cv2
import imtools as it 
import random as rand
def rotater(img, ang):
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst    
    


def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def find_centers(X, K):
    # Initialize to K random centers
    oldmu = rand.sample(X, K)
    mu = rand.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)


class AFusedBox():   # format is (y,x)    is (36,18) for bulk    is (26,44) for fused
  def __init__(self,y,x): #x1, y1 is top left corner
    self.x1 = x - 22
    self.x2 = x + 22
    self.y1 = y - 13
    self.y2 = y + 13
  #def inside(self,x1,y1)

    
class ABulkBox():   # format is (y,x)    is (36,18) for bulk    is (26,44) for fused
  def __init__(self,y,x): #x1, y1 is top left corner
    self.x1 = x - 9
    self.x2 = x + 9
    self.y1 = y - 18
    self.y2 = y + 18
    
    
def buildBox(dims, filterType = 'fused'):
  if filterType == 'fused':
        mybox = AFusedBox(dims[0],dims[1])
  elif filterType == 'bulk':
        mybox = ABulkBox(dims[0],dims[1])
  return mybox


    
# either load in data from file (imgName!=None) or pass in data (img!=None)
def makeMask(threshold = 245, 
             img=None,imgName=None,
             doKMeans = True,
             K = 4,  # Ryan what is this parameter? 
             inverseThresh=False
             ):
    # test if numpy array
    if isinstance(img, (list, tuple, np.ndarray)): 
      correlated = img
    # test if string
    elif isinstance(imgName, (str)):
      correlated = it.ReadImg(imgName)
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

    if doKMeans ==False:
      return newmasker     
    else:
      raise RuntimeError("probably don't need this anymore") 

    print "WARNING: please dig into why find_centers fails from time to time (or look into more robust clustering routine)"
    
    threshed = np.argwhere(correlated>threshold)
    centers = find_centers(X = threshed, K=K)



    boxlist= []
    for c, center in enumerate(centers[0]):
        xs= []
        
        others = np.copy(centers[0])
        IS = np.argwhere(center)
        #print "IS", IS
        #print "center old", others
        otherCenters = np.delete(others, IS)
        #print "center new", otherCenters
        oC =otherCenters.reshape(3,2)
        for k, kenter in enumerate(oC):
          mybool = center[1] not in xs
          #distance = abs(kenter[1]-center[1])<3 and abs(kenter[0]-center[0])>2
          if mybool:# and distance:
            boxlist.append([int(np.floor(center[0])),int(np.floor(center[1]))])
            break
            xs.append(center[0])
    #print "final boxlist", boxlist
    myMask = (np.zeros_like(correlated)) 
    #print "myMask shape", np.shape(myMask)

    for b, box in enumerate(boxlist):
        thisBox = buildBox(box)
        #print 'box', thisBox.y1,thisBox.y2,thisBox.x1,thisBox.x2
        myMask[thisBox.y1:thisBox.y2,thisBox.x1:thisBox.x2] = 1
    return myMask  
