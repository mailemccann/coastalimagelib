import numpy as np
import cv2 as cv
import sys

class ImageStats():
    """
    This class takes in rectified images as arguments in addImage() 
    then calculates and saves statistical image products such as 
    time- exposure, variance, brightest pixels, and darkest pixels.

    All images must be in black and white, and must all have the
    same dimensions.  

    Attributes:
        iBright: brightest pixels at each location across the 
                collection
        iDark: darkest pixels at each location across the collection
        iTimex: time- exposure (mean) of all pixels at each location
                across the collection
        iVariance: standard deviation of all pixels at each location
                across the collection

    """
    def __init__(self, images):
        self.calc_flag = 0
        self.im_mat = images

    def addImage(self,Im):
        self.im_mat = np.dstack((self.im_mat,Im))

    def saveProducts(self):
        '''
        Function to save all image products

        Can only be called after calc stats has run
        '''
        if self.calc_flag:
            cv.imwrite('Darkest.jpg', self.Dark)
            cv.imwrite('Brightest.jpg', self.Bright)
            cv.imwrite('Timex.jpg', self.Timex)
            cv.imwrite('Variance.jpg', self.Variance)
            cv.waitKey(0) 

        else: sys.exit('calcStats() must be called before products can be saved.')

    def dispProducts(self):
        '''
        Function to display all image products

        Can only be called after calc stats has run
        '''

        if self.calc_flag:
            cv.imshow('Darkest', self.Dark)
            cv.imshow('Brightest', self.Bright)
            cv.imshow('Timex', self.Timex)
            cv.imshow('Variance', self.Variance)
            cv.waitKey(0) 

        else: sys.exit('calcStats() must be called before products can be displayed.')

    def calcStats(self, save_flag=0):
        '''
        This function generates statistical image products for a given set of 
        images and corresponding extrinsics/intrinsics. The statistical image
        products are the timex, brightest, variance, darkest. 

        Args:
            save_flag: flag to indicate if products should be saved
                    automatically to the user's drive

        '''

        self.calc_flag = 1
        self.Dark = np.uint8(np.nanmin(self.im_mat,axis=2))
        self.Bright = np.uint8(np.nanmax(self.im_mat,axis=2))
        self.Timex = np.uint8(np.nanmean(self.im_mat, axis=2)) 
        self.Variance = np.uint8(np.nanstd(self.im_mat, axis=2))
        if save_flag: self.saveProducts()