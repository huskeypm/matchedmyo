import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.mlab as mlab
import cv2
def myplot(img,fileName=None):
    plt.axis('equal')
    plt.pcolormesh(img, cmap='gray')
    if fileName!=None:
        plt.gcf().savefig(fileName,dpi=300)
    plt.axis('off')
    plt.tight_layout()

def ReadImg(fileName):
    img = cv2.imread(fileName)
    if img is None:
        raise RuntimeError(fileName+" likely doesn't exist")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def mytriplot(img,h1,h2):
    plt.subplot(231)
    plt.axis('equal')
    plt.pcolormesh(img,cmap='gray')
    plt.title("Raw")
    plt.axis('off')
    
    plt.subplot(232)
    plt.axis('equal')
    plt.pcolormesh(h1,cmap='gray')
    plt.title("Filter Xtal")
    plt.axis('off')
    
    plt.subplot(233)
    plt.axis('equal')
    plt.pcolormesh(h2,cmap='gray')
    plt.title("Filter Interface")
    plt.axis('off')

    plt.tight_layout()




def padWithZeros(array, padwidth, iaxis, kwargs):
    array[:padwidth[0]] = 0
    array[-padwidth[1]:]= 0
    return array

def rotater(img, ang):
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst



