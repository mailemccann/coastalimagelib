import sys
import imageio
import numpy as np
import cv2 as cv
from math import sin,cos
from scipy.interpolate import RegularGridInterpolator as reg_interp
from scipy.ndimage.morphology import distance_transform_edt
from skimage.exposure import match_histograms

class ImageStats():
    """
    This function generates statistical image products for a given set of 
    images and corresponding extrinsics/intrinsics. The statistical image
    products are the timex, brightest, variance, darkest. If specified,
    rectified imagery for each image/frame can be produced and saved as an
    png file. Rectified images and image products can be produced in world
    or local coordinates if specified in the grid provided by
    D_gridGenExampleRect. This can function can be used for a collection with
    variable (UAS)  and fixed intrinsics in addition to single/multi camera
    capability.


    %  Input:
    oblique images and extrinsics solutions calculated by
    
    rectification grid information

    % Output:
    % 5 Image Products  as well as individual rectified frames if desired saved
    % as pngs. The accompanying metadata will be saved along with grid
    % information in a mat file in the same ouputdirectory as the images. If
    % multi-camera data is selected, the rectified individual frames will share
    % the same name as the fist cam.    

    """
    def __init__(self, xlims, ylims, dx=1, z=0, coords = 'local', origin = 'None', mType = 'CIRN'):

        # Make XYZ grid
        self.X, self.Y = np.meshgrid(np.arange(xlims[0], xlims[1]+dx, dx), 
                                     np.arange(ylims[0], ylims[1]+dx, dx)
                                     )
        self.Z = np.zeros_like(self.X) + z
        x = self.X.copy().T.flatten()
        y = self.Y.copy().T.flatten()
        z = self.Z.copy().T.flatten()
        self.xyz = np.vstack((x, y, z)).T

        # Init other params
        self.coords = coords
        self.origin = origin
        self.mType = mType
        self.s = self.X.shape