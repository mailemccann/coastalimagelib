import numpy as np
import cv2 as cv
from scipy.interpolate import RegularGridInterpolator as reg_interp
import corefuncs as cf
import imageio
import matplotlib.pyplot as plt
import sys

def pixelStack(grid, im_mat, intrinsic_list, extrinsic_list, sample_freq=1, disp_flag=0, coords = 'local', origin = 'None', mType = 'CIRN'):        
    '''
    Function that creates pixel instruments for use in bathymetric inversion,
    surface current, run-up calculations, or other quantitative analyses. 
    Instruments can be made in both world and a local rotated coordinate
    system. However, for ease and accuracy, if planned to use in bathymetric
    inversion, surface current, or run-up applications, it should occur
    in local coordinates.


    Inputs:
        grid: XYZGrid object of real world coordinates
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
            coords (string): 'geo' or 'local'; if 'geo', extrinsics are transformed to local
            but origin is needed to transform
        origin: local origin (x,y,z,angle)
        mType (string): format of intrinsic matrix, 'CIRN' is default, 'DLT' is also supported
        
    Returns:
        pixels (ndarray): pixel intensities at xyz points (georectified image). Size depends on type of pixels. Axis 3 has the same length
                        as number of frames/ sample rate

    '''  
    
    # Loop for Collecting Pixel Instrument Data.
    num_frames = im_mat.shape[3]
    s = grid.X.shape

    for curr_frame in range(0,num_frames,sample_freq):
        # Array for pixel values
        numcams = len(intrinsic_list)
        IrIndv = np.zeros((s[0], s[1], numcams))
        if disp_flag: fig,axs = plt.subplots(1,numcams)

        # Iterate through each camera from to produce single merged frame
        for k, (intrinsics, extrinsics) in enumerate(zip(intrinsic_list, extrinsic_list)):
            image = im_mat[:,:,k,curr_frame]

            # Load individual camera intrinsics, extrinsics, and camera matrices
            calib = cf.CameraData(intrinsics, extrinsics, origin, coords, mType)

            # Work in grayscale
            if len(image.shape) > 2:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  

            # Find distorted UV points at each XY location in grid
            if mType == 'CIRN':
                Ud, Vd = cf.xyz2DistUV(grid, calib)
            elif mType == 'DLT':
                Ud, Vd = cf.dlt2UV(grid, calib)
            else:
                sys.exit('This intrinsics format is not supported')

            # Grab pixels from image at each position
            ir = cf.getPixels(image, Ud, Vd, s)
            IrIndv[:,:,k] = ir

            if disp_flag:
                # Show pixels on image
                axs[k].xaxis.set_visible(False)
                axs[k].yaxis.set_visible(False)
                axs[k].imshow(image.astype(np.uint8), cmap='gray', vmin=0, vmax=255)
                mask = (Ud<image.shape[1]) & (Ud>0) & (Vd<image.shape[0]) & (Vd>0)
                axs[k].scatter(Ud[mask], Vd[mask], s=3,c="r")
            
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
    pixels = pixels.astype(np.uint8)

    return pixels
