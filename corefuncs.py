import sys
import imageio
import numpy as np
import cv2 as cv
from math import sin,cos
from scipy.interpolate import RegularGridInterpolator as reg_interp
from scipy.ndimage.morphology import distance_transform_edt
from skimage.exposure import match_histograms
from supportfuncs import initFullFileName, localTransformPoints


'''

Core Rectification Functions

Adapted in part from Brittany Bruder and Kate Brodie's CIRN
MATLAB Toolbox and Chris Sherwood's CoastCam github repo

'''

class Rectifier(object):
    """
    Object that contains parameters and functions for a rectification task that may
    involve multiple cameras and multiple frames.

    For multiple rectification tasks with the same desired rectification XYZ grid,
    you only have to initialize this function once, then call mergeRectify for
    each set of camera data and oblique images.

    If local coordinates are desired and inputs are not yet converted to local,
    user can flag coords = 'geo' and input local origin.

    If intrinsics are in a format other than CIRN (currently the only other
    supported format is DLT), user can flag mType = 'DLT'.

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

    """
    def __init__(self, xlims, ylims, dx=1, dy=1, z=0, coords = 'local', origin = 'None', mType = 'CIRN'):

        self.initXYZ(xlims, ylims, dx, dy, z)

        # Init other params
        self.coords = coords
        self.origin = origin
        self.mType = mType
        self.s = self.X.shape

    def initXYZ(self, xlims, ylims, dx, dy, z):
        '''
        Function to initialize the XYZ grid

        xlims and ylims can either be an array of two numbers,
        or one value if a 1D transect is desired
        '''

        if len(xlims)<2:
            xvec = xlims
        else:
            xvec = np.arange(xlims[0], xlims[1]+dx, dx)

        if len(ylims)<2:
            yvec = ylims
        else:
            yvec = np.arange(ylims[0], ylims[1]+dy, dy)

        # Make XYZ grid
        self.X, self.Y = np.meshgrid(xvec, yvec)
        self.Z = np.zeros_like(self.X) + z
        x = self.X.copy().T.flatten()
        y = self.Y.copy().T.flatten()
        z = self.Z.copy().T.flatten()
        self.xyz = np.vstack((x, y, z)).T

    def xyz2DistUV(self):
        '''
        This function computes the distorted UV coordinates (UVd)  that
        correspond to a set of world xyz points for for given camera
        extrinsics and intrinsics. Function also
        produces a flag variable to indicate if the UVd point is valid.

        Returns:
            DU: Nx1 vector of distorted U coordinates for N points
            DV: Nx1 vector of distorted V coordinates for N points

        '''

        # Take Calibration Information, combine it into a singular P matrix
        # containing both intrinsics and extrinsic information, make
        # homogenous
        cal = self.calib
        xyz = np.vstack((self.xyz.T,
                        np.ones((len(self.xyz),))
                        ))
        UV = np.matmul(cal.P, xyz)
        UV = UV / np.tile(UV[2, :], (3, 1))

        # Normalize distances
        u = UV[0, :]
        v = UV[1, :]
        x = (u - cal.c0U)/cal.fx
        y = (v - cal.c0V)/cal.fy

        # Radial distortion
        r2 = x*x + y*y
        fr = 1. + cal.d1*r2 + cal.d2*(r2*r2) + cal.d3*(r2*(r2*r2))

        # Tangential Distortion
        dx = 2.*cal.t1*x*y + cal.t2*(r2+2.*x*x)
        dy = cal.t1*(r2+2.*y*y) + 2.*cal.t2*x*y

        #  Apply Correction
        xd = x*fr + dx
        yd = y*fr + dy
        Ud = xd*cal.fx + cal.c0U
        Vd = yd*cal.fy + cal.c0V
        mask = (Ud<0) | (Ud>cal.NU) | (Vd<0) | (Vd>cal.NV)
        Ud[mask] = 0
        Vd[mask] = 0

        # Calc maximum tangential distortion
        Um = np.array((0, 0, cal.NU, cal.NU))
        Vm = np.array((0, cal.NV, cal.NV, 0))

        # Normalize
        xm = (Um-cal.c0U)/cal.fx
        ym = (Vm-cal.c0V)/cal.fy
        r2m = xm*xm + ym*ym

        # Tangential Distortion
        dxm = 2.*cal.t1*xm*ym + cal.t2*(r2m + 2.*xm*xm)
        dym = cal.t1*(r2m + 2.*ym*ym) + 2.*cal.t2*xm*ym

        # Flag values outside xy limits
        flag = np.ones_like(Ud)
        flag[np.where(np.abs(dy) > np.max(np.abs(dym)))] = 0.
        flag[np.where(np.abs(dx) > np.max(np.abs(dxm)))] = 0.

        DU = Ud.reshape(self.xy_grid.X.shape, order='F')
        DV = Vd.reshape(self.xy_grid.Y.shape, order='F')

        # Flag negative Z values
        UV = np.matmul(cal.P, xyz)
        xyzC = np.matmul(cal.R, np.matmul(cal.IC, xyz))
        flag[np.where(xyzC[2,:] <= 0.)] = 0.
        flag = flag.reshape(self.xy_grid.X.shape, order='F')

        # Apply flag to remove invalid points (set points = 0)
        return DU*flag, DV*flag

    def dlt2UV(self):
        '''
        This function computes the distorted UV coordinates (UVd)  that
        correspond to a set of world xyz points for a given camera m matrix
        for DLT equations

        Input (through self):
            m = the DLT coefficient vector A->L
            X = [N,3] maxtrix (real world coords)

        Attributes:
            DU= Nx1 vector of distorted U coordinates for N points.
            DV= Nx1 vector of distorted V coordinates for N points.

        '''

        m = np.asarray(self.calib.intrinsics)
        m = m.reshape(-1,1)

        #  Carry out the equivalent vectorized calculation of
        #       U = (Ax + By + Cz + D) / (Ex + Fy + Gz + 1)
        #       V = (Hx + Jy + Kz + L) / (Ex + Fy + Gz + 1)

        x2 = np.hstack((self.xyz, np.ones((self.xyz.shape[0], 1))))

        denom = np.matmul(x2,np.vstack((m[4:7], 1)))
        U = np.matmul(x2,m[0:4]) / denom
        V = np.matmul(x2,m[7:11]) / denom

        DU = U.reshape(self.X.shape, order='F')
        DV = V.reshape(self.Y.shape, order='F')

        return DU, DV

    def matchHist(self,image):
        '''

        Trying Chris Sherwood's method if using an RBG image
        Note: usually working with BW for Argus stuff so far


        Matches the histogram of an input image to a reference
        image saved in self in order to better blend seams
        of multiple cameras.

        Args:
            image (ndarray): image to match histogram

        Returns:
            matched (ndarray): modified image with matching histogram

        '''
        if len(image.shape) > 2:
            # Convert to hsv space
            im_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
            ref_hsv = cv.cvtColor(self.ref, cv.COLOR_RGB2HSV)

            # Match histogram to reference hist on only v channel
            matched_v = match_histograms(im_hsv[:,:,2],ref_hsv[:,:,2], multichannel=False)

            # Paste matched values channel into HSV version of source image; convert back to RGB
            matched_hsv = im_hsv.copy()
            matched_hsv[:,:,2] = matched_v
            matched = cv.cvtColor(matched_hsv, cv.COLOR_HSV2RGB)
        else:
            if len(self.ref.shape) > 2:
                ref = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            else: ref = self.ref

            matched = match_histograms(image, ref, multichannel=False)
        return matched

    def getPixels(self, image):

        '''
        Pulls rgb or gray pixel intensities from image at specified
        pixel locations corresponding to X,Y coordinates calculated in either
        xyz2DistUV or dlt2UV.

        Args:
            image (ndarray): image where pixels will be taken from

        Attributes:
            ir (ndarray): pixel intensities

        '''

        # Use regular grid interpolator to grab points
        ir = np.full((self.s[0], self.s[1]), np.nan)
        rgi = reg_interp((np.arange(0, image.shape[0]),
                          np.arange(0, image.shape[1])),
                          image, bounds_error=False, fill_value=np.nan)
        ir = rgi((self.Vd, self.Ud))

        # Mask out values out of range
        mask = (self.Ud>image.shape[1]) & (self.Ud<0) & (self.Vd>image.shape[0]) & (self.Vd<0)
        ir[mask] = np.nan

        return np.uint8(ir)

    def cameraSeamBlend(self, IrIndv):
        """
        This function takes rectifications from different cameras (but same grid)
        and merges them together into a single rectification. To do this, the function
        performs a weighted average where pixels closest to the seams are not
        represented as strongly as those closest to the center of the camera
        rectification.

        Notes:
            - Calculates Euclidean distance for each entry to nearest non-zero pixel value
            - edt: Euclidean Distance Transform

        Args:
            IrIndv (ndarray) A NxMxK matrix where N and M are the grid lengths for the
                rectified image and K is the number of cameras.
                Each k entry is a rectified image from a camera.

        Returns:
            M (ndarray): A NxM uint8 matrix of the greyscale merged rectified image.
                N and M are the grid lengths for the rectified image.

        """

        totalW = np.zeros_like(IrIndv[:,:,0])
        M = np.zeros_like(IrIndv[:,:,0])

        # Iterate through each camera
        for kk in range(self.numcams):
            K = IrIndv[:,:,kk]

            # Make binary matrix from image (1 for Nan, 0 for nonnan)
            K[K==0] = np.nan

            # edt finds euclidean distance from Nan to closest real value
            # Find the nans, then invert
            W = distance_transform_edt(~np.isnan(K))
            if np.isinf(np.max(W)):
                W = np.ones_like(W)
            W = W / np.max(W)

            # Apply weights to image
            K_weighted = K*W

            # Add weights and pixel itensities
            totalW = totalW + W
            K_weighted[np.isnan(K_weighted)] = 0
            M = M + K_weighted

        # Stop divide by 0 warnings
        with np.errstate(invalid='ignore'):
            M = M / totalW

        M[np.isnan(M)]=0

        return M.astype(np.uint8)

    def mergeRectify(self, cameras_frames, intrinsic_list, extrinsic_list):

        """
        This function performs image rectifications at one timestamp given the associated
        extrinsics, intrinsics, and distorted images corresponding to each camera.
        The function utilizes matchHist to match images from each camera
        to the same histogram, then calls xyz2DistUV or dlt2UV to find corresponding
        UVd values to the input grid and pulls the rgb pixel intensity for
        each value using getPixels. If a multi-camera rectification is desired,
        images, intrinsic_list, and extrinsic_list can be input as cell values
        for each camera.

        If local coordinates are desired and inputs are not yet converted to local,
        user can flag coords = 'geo' and input local origin.

        If intrinsics are in a format other than CIRN (currently the only other
        supported format is DLT), user can flag mType = 'DLT'.

        The function calls cameraSeamBlend as a last step to merge the values.

        Inputs:
            cameras_images (list OR ndarray): 1xK list of paths to image files for each camera,
                            OR NxMxK struct of images, one image per camera at the desired timestamp
                            for rectification
            intrinsic_list (list): 1x11xK internal calibration data for each camera
            extrinsic_list (list): 1x6xK external calibration data for each camera
            xyz (ndarray): 3xN array if desired pixels are from different points than the main XYZ grid

        Returns:
            Ir (ndarray): Image intensities at xyz points (georectified image)

        """
        # Array for pixel values
        self.numcams = len(intrinsic_list)
        IrIndv = np.zeros((self.s[0], self.s[1], self.numcams))

        # Iterate through each camera from to produce single merged frame
        for k, (I, intrinsics, extrinsics) in enumerate(zip(cameras_frames, intrinsic_list, extrinsic_list)):

            # Load individual camera intrinsics, extrinsics, and camera matrices
            self.calib = CameraData(intrinsics, extrinsics, self.origin, self.coords, self.mType)

            # User can load filepaths or images, determine which we are working with
            if isinstance(I,str)==1:
                # Load image from current camera
                image = imageio.imread(I)
            else:
                image = cameras_frames[:,:,k]

            # Match histograms
            if k==0:
                self.ref = image
            else:
                image = self.matchHist(image)

            # Work in grayscale
            if len(image.shape) > 2:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # Find distorted UV points at each XY location in grid
            if self.mType == 'CIRN':
                self.Ud, self.Vd = self.xyz2DistUV()
            elif self.mType == 'DLT':
                self.Ud, self.Vd = self.dlt2UV()
            else:
                sys.exit('This intrinsics format is not supported')

            # Grab pixels from image at each position
            ir = self.getPixels(image)
            IrIndv[:,:,k] = ir

        # Merge rectifications of multiple cameras
        Ir = self.cameraSeamBlend(IrIndv)

        # Return merged frame
        return np.flipud(Ir.astype(np.uint8))

    def rectVideos(self, video_list, intrinsic_list, extrinsic_list, numFrames):

        """
        This function performs image rectifications on .avi files, either single or multi- cam,
        and saves a merged and rectified .avi to the user's drive.

        Inputs:
            video_list (list): 1xK list of paths to video files for each camera
            intrinsic_list (list): 1x11xK internal calibration data for each camera
            extrinsic_list (list): 1x6xK external calibration data for each camera

        Returns:
            rect_array: h x w x numFrames array of rectified frames

        """

        caps = []
        # Append each video capture object into structure
        for path in video_list:
            caps.append(cv.VideoCapture(path))

        # Loop through each frame
        for i in range(numFrames):
            for ind, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret:
                    break

                # Add each frame into one object
                if ind==0:
                    frames = np.empty((frame.shape[0],frame.shape[1],len(video_list)))
                frames[:,:,ind] = frame[:,:,0]

            # All frames from one timestamp sent to mergeRectify
            merged = self.mergeRectify(frames, intrinsic_list, extrinsic_list)

            if i ==0:
                # Create videowriter object to save video to drive
                self.outFile = initFullFileName(video_list[0],'video_capture.rect',type='avi')
                result = cv.VideoWriter(self.outFile,cv.VideoWriter_fourcc('M','J','P','G'),
                                        1, (merged.shape[1],merged.shape[0]),0)
                # Initialize array of images
                self.rect_arr = np.empty((merged.shape[0],merged.shape[1],numFrames))

            # Write to drive
            result.write(merged.astype(np.uint8))

            # Add to image array
            self.rect_arr[:,:,i] = merged.astype(np.uint8)

        print('Rectified .avi movie saved to working directory as: ' + self.outFile)
        result.release()

        return self.rect_arr

    def saveNetCDF(self,rect_arr,outfile):
        '''
        Saves the object rect_arr as a netcdf on the user's drive
        rect_arr can be a rectified image, struct of images, pixel instruments, etc
        '''
        import xarray as xr

        # what data types will be used in the netcdf
        encoding = {'merged':{'dtype':'uint8','_FillValue':0}}
        # use xarray to create the netcdf
        ncstruct = xr.Dataset({'merged': (['y', 'x'], rect_arr)},
                                coords={'xyz': self.xyz,
                                        'coord_type': self.coords,
                                        })
        ncstruct.to_netcdf(outfile,encoding=encoding)

class CameraData(object):
    '''
    Object that contains camera matrices in homogenous coordinates from camera
    extrinsics and intrinsics.Must be re-initialized for each new camera
    (dependent on specific camera's intrinsic and extrinsic calibration).

    Arguments:
        intrinsics (list or array): [1 x 11] list of intrinsic values in CIRN format or DLT coefficients
        extrinsics (list or array): [1 x 6] list of extrinsic values [ x y z azimuth tilt swing] of the camera.
            XYZ should be in the same units as xyz points to be converted and azimith,
            tilt, and swing should be in radians.
            (azimuth, tilt, and swing should be defined by CIRN convention- see User Manual)
        origin (list or array): local origin, x, y, and angle
        coords (string): tag to indicate if coordinates are in geo or local
        mTypes (string): tag to indicate if intrinsics are in CIRN or DLT

    Attributes (if mType = 'CIRN'): (in local coords)
        P: [3 x 4] transformation matrix to convert XYZ coordinates to distorted
        UV coordinates.
        K: [3 x 3] K matrix to convert XYZ points to distorted UV coordinates
        R: [3 x 3] Rotation matrix to rotate XYZ world coordinates to camera coordinates
        IC: [4 x 3] Translation matrix to translate XYZ world coordinates to camera coordinates

    '''

    def __init__(self,intrinsics, extrinsics, origin= 'None', coords = 'local', mType = 'CIRN'):

        self.intrinsics = intrinsics
        self.origin = origin
        self.coords = coords

        # If in geo coordinates, convert to local
        self.local_extrinsics = extrinsics
        if (self.coords == 'geo') & (origin == 'None'):
            sys.exit('Local origin is required for a coordinate transform. \
                      If you wish to work in geographical coordinates, \
                      please enter coords = "local", and geo coordinates \
                      will be considered the local working coordinate system.')
        if (self.coords == 'geo') & (origin != 'None'):
            self.local_extrinsics = self.localTranformExtrinsics()
        if mType != 'DLT':
            self.P,self.K,self.R,self.IC = self.getMatrices()

        self.assignCoeffs()

    def getMatrices(self):
        '''
        Part of initializer for mType = 'CIRN', calculates P, K, R, and IC matrices

        Returns:
            P: full camera matrix (3x3)
            K: intrinsic matrix (3x3)
            R: rotation matrix (3x3)
            IC: identity translation matrix (3x3 identity matrix - 3x1 translation matrix)

        '''

        # Define intrinsic coefficients
        fx = self.intrinsics[4]
        fy = self.intrinsics[5]
        c0U = self.intrinsics[2]
        c0V = self.intrinsics[3]

        # Format K matrix
        K = np.array([
                [-fx, 0, c0U],
                [0, -fy, c0V],
                [0,  0,  1]
                ])

        # Format rotation matrix R
        azimuth = self.local_extrinsics[3]
        tilt = self.local_extrinsics[4]
        swing = self.local_extrinsics[5]

        # Calculate R values
        R = np.empty((3,3))
        R[0,0] = -cos(azimuth) * cos(swing) - sin(azimuth) * cos(tilt) * sin(swing)
        R[0,1] = cos(swing) * sin(azimuth) - sin(swing) * cos(tilt) * cos(azimuth)
        R[0,2] = -sin(swing) * sin(tilt)
        R[1,0] = -sin(swing) * cos(azimuth) + cos(swing) * cos(tilt) * sin(azimuth)
        R[1,1] = sin(swing) * sin(azimuth) + cos(swing) * cos(tilt) * cos(azimuth)
        R[1,2] = cos(swing) * sin(tilt)
        R[2,0] = sin(tilt) * sin(azimuth)
        R[2,1] = sin(tilt) * cos(azimuth)
        R[2,2] = -cos(tilt)

        # Format translation matrix IC
        x = self.local_extrinsics[0]
        y = self.local_extrinsics[1]
        z = self.local_extrinsics[2]

        IC = np.array([
            [1, 0, 0, -x],
            [0, 1, 0, -y],
            [0, 0, 1, -z]
            ])

        # Combine K, R, and IC into P
        KR = np.matmul(K,R)
        P = np.matmul(KR,IC)

        # Make homogeneous
        P = P/P[-1,-1]

        return P,K,R,IC

    def localTranformExtrinsics(self):

        """
        Transforms extrinsics in local coordinates to geo, or extrinsics in geo coordinates to local
        Angle should be defined by CIRN convention.

        Returns:
            extrinsics_out (dict): Local or geo extrinsics [x,y,z,a,t,r] (a = azimuth, t = tilr, r = roll)

        """

        # IMPORT SUPPORT FUNCTIONS


        self.origin[2] = np.deg2rad(self.origin[2])

        # This will allow the user to convert to local in the constructor function, but this function can also be used
        # if you ever want to go from local to geo later
        extrinsics_out = self.local_extrinsics.copy()
        if self.coords == 'geo':
            # World to local
            extrinsics_out[0], extrinsics_out[1] = localTransformPoints(self.origin,1,
                                                                        self.local_extrinsics[0],
                                                                        self.local_extrinsics[1])
            extrinsics_out[3] = self.local_extrinsics[3] + self.origin[2]
        else:
            # local to world
            extrinsics_out[0], extrinsics_out[1] = localTransformPoints(self.origin,0,
                                                                        self.local_extrinsics[0],
                                                                        self.local_extrinsics[1])
            extrinsics_out[3] = self.local_extrinsics[3] - self.origin[2]

        return extrinsics_out

    def assignCoeffs(self):

        # Assign Coefficients to Intrinsic Matrix
        self.NU = self.intrinsics[0]
        self.NV = self.intrinsics[1]
        self.c0U = self.intrinsics[2]
        self.c0V = self.intrinsics[3]
        self.fx = self.intrinsics[4]
        self.fy = self.intrinsics[5]
        self.d1 = self.intrinsics[6]
        self.d2 = self.intrinsics[7]
        self.d3 = self.intrinsics[8]
        self.t1 = self.intrinsics[9]
        self.t2 = self.intrinsics[10]
