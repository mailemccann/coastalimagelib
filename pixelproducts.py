import numpy as np
import cv2 as cv
from scipy.interpolate import RegularGridInterpolator as reg_interp
from corefuncs import Rectifier, CameraData
import imageio
import matplotlib.pyplot as plt


class PixelStack(Rectifier):

    '''
    Class that contains pixel instruments for use in bathymetric inversion,
    surface current, run-up calculations, or other quantitative analyses. 
    Instruments can be made in both world and a local rotated coordinate
    system. However, for ease and accuracy, if planned to use in bathymetric
    inversion, surface current, or run-up applications, it should occur
    in local coordinates.

    Args:
        xlims (ndarray): min and max (inclusive) in the x-direction (e.g. [-250, 500])
        ylims (ndarray): min and max (inclusive) in the y-direction (e.g. [0, 1800])
        dx (float): resolution of grid in x direction (same units as camera calibration)
        dy (float): resolution of grid in y direction (same units as camera calibration)
        z (float): estimated elevation at every point in the x, y grid
        coords (string): 'geo' or 'local'; if 'geo', extrinsics are transformed to local
                but origin is needed to transform
        origin: local origin (x,y,z,angle)
        mType (string): format of intrinsic matrix, 'CIRN' is default, 'DLT' is also supported

    '''   

    def __init__(self, xlims, ylims, dx=1, dy=1, z=0, coords = 'local', origin = 'None', mType = 'CIRN'):

        Rectifier.__init__(self, xlims, ylims, dx, dy, z, coords, origin, mType)

    def pullTransect(self, im_mat, intrinsic_list, extrinsic_list, sample_freq=1, disp_flag=0):        
        '''
        This function generates pixel instruments (timestacks) for a given set of images and
        corresponding extrinsics/intrinsics. 

        Inputs:
            cameras_images (array): N x M x K x num_frames struct of images, one image per camera (K) at the desired timestamp
                            for rectification (N x M is image height x image width, k is number of cameras, num_frames is 
                            number of frames/ timestamps)
            intrinsic_list (list): 1x11xK internal calibration data for each camera
            extrinsic_list (list): 1x6xK external calibration data for each camera
            xyz (ndarray): 3xN array if desired pixels are from different points than the main XYZ grid 
            sample_freq (int): desired frequency at which to grab pixels. This does not factor in camera metadata. User must know frames
                            per second of collection. freq=1 means grab pixels at every frame. freq=2 means grab pixels at every other frame,
                            and so on
            disp_flag (bool): flag that user can use to display output image products
            
        Returns:
            pixels (ndarray): pixel intensities at xyz points (georectified image). Size depends on type of pixels. Axis 3 has the same length
                            as number of frames/ sample rate

        '''  
        
        # Loop for Collecting Pixel Instrument Data.
        num_frames = im_mat.shape[3]

        for curr_frame in range(0,num_frames,sample_freq):
            # Array for pixel values
            self.numcams = len(intrinsic_list)
            IrIndv = np.zeros((self.s[0], self.s[1], self.numcams))
            if disp_flag: fig,axs = plt.subplots(1,self.numcams)

            # Iterate through each camera from to produce single merged frame
            for k, (intrinsics, extrinsics) in enumerate(zip(intrinsic_list, extrinsic_list)):
                image = im_mat[:,:,k,curr_frame]

                # Load individual camera intrinsics, extrinsics, and camera matrices
                self.calib = CameraData(intrinsics, extrinsics, self.origin, self.coords, self.mType)

                # Work in grayscale
                if len(image.shape) > 2:
                    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  

                # Find distorted UV points at each XY location in grid
                if self.mType == 'CIRN':
                    self.Ud, self.Vd, flag = self.xyz2DistUV()
                if self.mType == 'DLT':
                    self.Ud, self.Vd = self.dlt2UV()

                # Grab pixels from image at each position
                ir = self.getPixels(image)
                IrIndv[:,:,k] = ir

                if disp_flag:
                    # Show pixels on image
                    axs[k].xaxis.set_visible(False)
                    axs[k].yaxis.set_visible(False)
                    axs[k].imshow(image.astype(np.uint8), cmap='gray', vmin=0, vmax=255)
                    mask = (self.Ud<image.shape[1]) & (self.Ud>0) & (self.Vd<image.shape[0]) & (self.Vd>0)
                    axs[k].scatter(self.Ud[mask], self.Vd[mask], s=3,c="r")
                
            if disp_flag: plt.show()
                

            # Merge rectifications of multiple cameras
            IrIndv[IrIndv==0] = np.nan

            # NEED BETTER BLENDING HERE
            Ir = np.nanmean(IrIndv,axis=2)

            if curr_frame == 0:   
                pixels = Ir
            else:
                pixels = np.dstack((pixels,Ir))

        # Return pixels
        self.pixels = pixels.astype(np.uint8)
        return self.pixels
