import numpy as np
from corefuncs import Rectifier

class ImageStats(Rectifier):
    """
    This function generates statistical image products for a given set of 
    images and corresponding extrinsics/intrinsics. The statistical image
    products are the timex, brightest, variance, darkest. 
    
    Rectified images and image products can be produced in world
    or local coordinates. This can function can be used for a collection with
    variable (UAS)  and fixed intrinsics in addition to single/multi camera
    capability.

    Args:
        oblique images and extrinsics solutions
        rectification grid information
        save_flag

    Attributes:
        brightest
        darkest
        timex
        variance
        metadata

    """
    def __init__(self, rect, save_flag=0):

        Rectifier.__init__(self, rect.xlims, rect.ylims, rect.dx, rect.z, rect.coords, rect.origin, rect.mType)
        self.save_flag = save_flag

        self.ims = []
        self.num_images = 0

    def addImage(self,Im):

        self.ims.append(Im)
        self.num_images +=1

    def saveProducts(self):
        dummy = 0

    def dispProducts(self):
        dummy = 0

    def calcStats(self):

        for j in self.num_images:
            image = self.ims[j]
             
            # Initiate Image Product variables  
            if j==0:
                self.s = image.shape
                iDark = np.ones((self.s[0], self.s[1]))*255 # Can't initialize as zero, will always be dark
                iTimex = np.zeros((self.s[0], self.s[1], self.num_images))
                iBright = np.zeros((self.s[0], self.s[1]))
            
            # Perform Statistical Calcutions

            # Timex: Sum Values, will divide by total number at last frame
            iTimex[:,:,j] = image 
            
            # Darkest: Compare New to Old value, save only the mimumum intensity
            iDark = np.nanmin(np.concatenate((iDark,image),axis=2),axis=2) 
            
            # Brightest: Compare New to Old value, save only the maximum intensity
            iBright = np.nanmax(np.concatenate((iBright,image),axis=2),axis=2) 
            
        self.iDark = iDark
        self.iBright = iBright
        self.iTimex = np.uint8(np.nanmean(iTimex, axis=2)) 