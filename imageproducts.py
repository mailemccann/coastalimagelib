import numpy as np
import cv2 as cv
import sys

def imageStats(im_mat, save_flag=0):
    '''
    This function generates statistical image products for a given set of 
    images and corresponding extrinsics/intrinsics. The statistical image
    products are the timex, brightest, variance, darkest. 

    All images must have the same dimensions.  

    Attributes:
        Bright: brightest pixels at each location across the 
                collection
        Dark: darkest pixels at each location across the collection
        Timex: time- exposure (mean) of all pixels at each location
                across the collection
        Variance: standard deviation of all pixels at each location
                across the collection

    Args:
        save_flag: flag to indicate if products should be saved
                automatically to the user's drive

    '''

    Dark = np.uint8(np.nanmin(im_mat,axis=2))
    Bright = np.uint8(np.nanmax(im_mat,axis=2))
    Timex = np.uint8(np.nanmean(im_mat, axis=2)) 
    Variance = np.uint8(np.nanstd(im_mat, axis=2))
    if save_flag: 
        cv.imwrite('Darkest.jpg', Dark)
        cv.imwrite('Brightest.jpg', Bright)
        cv.imwrite('Timex.jpg', Timex)
        cv.imwrite('Variance.jpg', Variance)

    cv.imshow('Darkest', Dark)
    cv.imshow('Brightest', Bright)
    cv.imshow('Timex', Timex)
    cv.imshow('Variance', Variance)
    cv.waitKey(0)