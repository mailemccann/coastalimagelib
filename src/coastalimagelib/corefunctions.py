import sys
import imageio
import numpy as np
from math import sin, cos
from scipy.interpolate import RegularGridInterpolator as reg_interp
from scipy.ndimage.morphology import distance_transform_edt
from skimage.exposure import match_histograms
from supportfunctions import initFullFileName, localTransformPoints
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
from skimage import color


"""

I. Core Rectification Functions


"""


def xyz2DistUV(grid, cal):

    """
    This function computes the distorted UV coordinates (UVd)  that
    correspond to a set of world xyz points for for given camera
    extrinsics and intrinsics. Function also
    produces a flag variable to indicate if the UVd point is valid.

    Arguments:
        cal: CameraData object containing the DLT coefficient vector A->L
        grid: XYZGrid object containing real world coords

    Returns:
        DU: Nx1 vector of distorted U coordinates for N points
        DV: Nx1 vector of distorted V coordinates for N points
    """

    # Take Calibration Information, combine it into a sigular P matrix
    # containing both intrinsics and extrinsic information, make
    # homogenous
    xyz = np.vstack((grid.xyz.T, np.ones((len(grid.xyz),))))
    UV = np.matmul(cal.P, xyz)
    UV = UV / np.tile(UV[2, :], (3, 1))

    # Normalize distances
    u = UV[0, :]
    v = UV[1, :]
    x = (u - cal.c0U) / cal.fx
    y = (v - cal.c0V) / cal.fy

    # Radial distortion
    r2 = x * x + y * y
    fr = 1.0 + cal.d1 * r2 + cal.d2 * (r2 * r2) + cal.d3 * (r2 * (r2 * r2))

    # Tangential Distortion
    dx = 2.0 * cal.t1 * x * y + cal.t2 * (r2 + 2.0 * x * x)
    dy = cal.t1 * (r2 + 2.0 * y * y) + 2.0 * cal.t2 * x * y

    #  Apply Correction
    xd = x * fr + dx
    yd = y * fr + dy
    Ud = xd * cal.fx + cal.c0U
    Vd = yd * cal.fy + cal.c0V
    mask = (Ud < 0) | (Ud > cal.NU) | (Vd < 0) | (Vd > cal.NV)
    Ud[mask] = 0
    Vd[mask] = 0

    # Calc maximum tangential distortion
    Um = np.array((0, 0, cal.NU, cal.NU))
    Vm = np.array((0, cal.NV, cal.NV, 0))

    # Normalize
    xm = (Um - cal.c0U) / cal.fx
    ym = (Vm - cal.c0V) / cal.fy
    r2m = xm * xm + ym * ym

    # Tangential Distortion
    dxm = 2.0 * cal.t1 * xm * ym + cal.t2 * (r2m + 2.0 * xm * xm)
    dym = cal.t1 * (r2m + 2.0 * ym * ym) + 2.0 * cal.t2 * xm * ym

    # Flag values outside xy limits
    flag = np.ones_like(Ud)
    flag[np.where(np.abs(dy) > np.max(np.abs(dym)))] = 0.0
    flag[np.where(np.abs(dx) > np.max(np.abs(dxm)))] = 0.0

    DU = Ud.reshape(grid.X.shape, order="F")
    DV = Vd.reshape(grid.Y.shape, order="F")

    # Flag negative Z values
    UV = np.matmul(cal.P, xyz)
    xyzC = np.matmul(cal.R, np.matmul(cal.IC, xyz))
    flag[np.where(xyzC[2, :] <= 0.0)] = 0.0
    flag = flag.reshape(grid.X.shape, order="F")

    # Apply flag to remove invalid points (set points = 0)
    return DU * flag, DV * flag


def dlt2UV(grid, cal):

    """
    This function computes the distorted UV coordinates (UVd)  that
    correspond to a set of world xyz points for a given camera m matrix
    for DLT equations

    Arguments:
        cal: CameraData object containing the DLT coefficient vector A->L
        grid: XYZGrid object containing real world coords

    Attributes:
        DU= Nx1 vector of distorted U coordinates for N points.
        DV= Nx1 vector of distorted V coordinates for N points.

    """

    m = np.asarray(cal.intrinsics)
    m = m.reshape(-1, 1)

    #  Carry out the equivalent vectorized calculation of
    #       U = (Ax + By + Cz + D) / (Ex + Fy + Gz + 1)
    #       V = (Hx + Jy + Kz + L) / (Ex + Fy + Gz + 1)

    x2 = np.hstack((grid.xyz, np.ones((grid.xyz.shape[0], 1))))

    denom = np.matmul(x2, np.vstack((m[4:7], 1)))
    U = np.matmul(x2, m[0:4]) / denom
    V = np.matmul(x2, m[7:11]) / denom

    DU = U.reshape(grid.X.shape, order="F")
    DV = V.reshape(grid.Y.shape, order="F")

    return DU, DV


def matchHist(ref, image):
    """

    Chris Sherwood's method if using an RBG image

    Matches the histogram of an input image to a reference
    image saved in self in order to better blend seams
    of multiple cameras.

    Arguments:
        ref (ndarray): reference image
        image (ndarray): image to match histogram

    Returns:
        matched (ndarray): modified image with matching histogram

    """

    if (len(image.shape) > 2) & (image.shape[2] > 1):
        # Convert to hsv space
        im_hsv = color.rgb2hsv(image)
        ref_hsv = color.rgb2hsv(ref)

        # Match histogram to reference hist on only v channel
        matched_v = match_histograms(
            im_hsv[:, :, 2], ref_hsv[:, :, 2], multichannel=False
        )

        # Paste matched values channel into HSV version of source image
        # Then, convert back to RGB
        matched_hsv = im_hsv.copy()
        matched_hsv[:, :, 2] = matched_v
        matched = color.hsv2rgb(matched_hsv)
    else:
        if (len(ref.shape) > 2) & (ref.shape[2] > 1):
            ref = color.rgb2gray(image)

        matched = match_histograms(image, ref, multichannel=False)
    return np.uint8(matched*255)


def getPixels(image, Ud, Vd, s):

    """
    Pulls rgb or gray pixel intensities from image at specified
    pixel locations corresponding to X,Y coordinates calculated in either
    xyz2DistUV or dlt2UV.

    Args:
        image (ndarray): image where pixels will be taken from
        Ud: Nx1 vector of distorted U coordinates for N points
        Vd: Nx1 vector of distorted V coordinates for N points
        s: shape of output image

    Returns:
        ir (ndarray): pixel intensities

    """

    # Use regular grid interpolator to grab points
    im_s = image.shape
    if len(im_s) > 2:
        ir = np.full((s[0], s[1], im_s[2]), np.nan)
        for i in range(im_s[2]):
            rgi = reg_interp(
                (np.arange(0, image.shape[0]), np.arange(0, image.shape[1])),
                image[:, :, i],
                bounds_error=False,
                fill_value=np.nan,
            )
            ir[:, :, i] = rgi((Vd, Ud))
    else:
        ir = np.full((s[0], s[1], 1), np.nan)
        rgi = reg_interp(
            (np.arange(0, image.shape[0]), np.arange(0, image.shape[1])),
            image,
            bounds_error=False,
            fill_value=np.nan,
        )
        ir[:, :, 0] = rgi((Vd, Ud))

    # Mask out values out of range
    with np.errstate(invalid="ignore"):
        mask_u = np.logical_or(Ud <= 1, Ud >= image.shape[1])
        mask_v = np.logical_or(Vd <= 1, Vd >= image.shape[0])
    mask = np.logical_or(mask_u, mask_v)
    if len(im_s) > 2:
        ir[mask, :] = np.nan
    else:
        ir[mask] = np.nan

    return ir


def cameraSeamBlend(IrIndv, numcams, nc):
    """
    This function takes rectifications from different cameras (but same grid)
    and merges them together into a single rectification.

    The function performs a weighted average where pixels closest to the seams
    are not weighted as highly as those closest to the center of the camera
    rectification.

    Notes:
        - Calculates Euclidean distance to nearest non-zero pixel value
        - edt: Euclidean Distance Transform

    Args:
        IrIndv (ndarray): A NxMxK matrix where N and M are the grid lengths
            for the rectified image and K is the number of cameras.
            Each k entry is a rectified image from a camera.

    Returns:
        M (ndarray): A NxM uint8 matrix of the greyscale merged image.
            N and M are the grid lengths for the rectified image.

    """

    totalW = np.zeros_like(IrIndv[:, :, :nc])
    M = np.zeros_like(IrIndv[:, :, :nc])

    # Iterate through each camera
    for kk in range(numcams):
        K = IrIndv[:, :, (kk * nc): (kk * nc + nc)]

        # Make binary matrix from image (1 for Nan, 0 for nonnan)
        K_nans = K[:, :, 0]
        K_nans[np.isnan(K_nans)] = np.nan

        # edt finds euclidean distance from Nan to closest real value
        # Find the nans, then invert
        W = distance_transform_edt(~np.isnan(K_nans))
        if np.isinf(np.max(W)):
            W = np.ones_like(W)
        else:
            W = W / np.max(W)

        # Apply weights to image
        K_weighted = K * W[:, :, np.newaxis]

        # Add weights and pixel itensities
        totalW = totalW + W[:, :, np.newaxis]

        K_weighted[np.isnan(K_weighted)] = 0
        M = M + K_weighted

    # Stop divide by 0 warnings
    with np.errstate(invalid="ignore"):
        M = M / totalW

    M[np.isnan(M)] = 0

    return M.astype(np.uint8)


def mergeRectify(input_frames, cameras, grid):
    """
    This function performs image rectifications at one timestamp given
    the associated extrinsics, intrinsics, and distorted images
    corresponding to each camera contained within the CameraData object.
    The function utilizes matchHist to match images from each camera
    to the same histogram, then calls xyz2DistUV or dlt2UV to find
    corresponding UVd values to the input grid and pulls the rgb pixel
    intensity for each value using getPixels.

    If a multi-camera rectification is desired, images, intrinsic_list,
    and extrinsic_list can be input as cell values for each camera.

    The function calls cameraSeamBlend as a last step to merge the values.

    Inputs:
        input_frames (list OR ndarray): 1xK list of paths to images for
            each camera, OR NxMxK struct of images, one image per camera
            at the desired timestamp for rectification
        cameras (array of CameraData objects): contains:
            intrinsic_list (list): 1x11xK internal calibration data for
                each camera
            extrinsic_list (list): 1x6xK external calibration data for
                each camera
            mType: intrinsic format ('DLT' or 'CIRN')
        xyz (ndarray): XYZ rectification grid

    Returns:
        Ir (ndarray): Image intensities at xyz points (georectified image)

    """
    s = grid.X.shape
    numcams = len(cameras)

    # Iterate through each camera from to produce single merged frame
    for k, (I, calib) in enumerate(zip(input_frames, cameras)):
        nc = calib.nc
        # Determine if the user provided a filepath or image
        if isinstance(I, str):
            # Load image from current camera
            image = imageio.v2.imread(I)
        else:
            image = input_frames[:, :, (k * nc): (k * nc + nc)]

        # Match histograms
        if k == 0:
            ref = image
        else:
            image = matchHist(ref, image)
        if calib.Ud == "None":
            # Find distorted UV points at each XY location in grid
            if calib.mType == "CIRN":
                Ud, Vd = xyz2DistUV(grid, calib)
            elif calib.mType == "DLT":
                Ud, Vd = dlt2UV(grid, calib)
            else:
                sys.exit("This intrinsics format is not supported")
        else:
            Ud = calib.Ud
            Vd = calib.Vd

        # Grab pixels from image at each position
        ir = getPixels(image, Ud, Vd, s)

        # Initialize array for pixel values
        if k == 0:
            IrIndv = np.tile(np.zeros((s[0], s[1], numcams)), (nc,))

        IrIndv[:, :, (k * nc): (k * nc + nc)] = ir

    # Merge rectifications of multiple cameras
    Ir = cameraSeamBlend(IrIndv, numcams, nc)

    # Return merged frame
    return np.flipud(Ir.astype(np.uint8))


def rectVideos(video_list, cameras, grid, numFrames, savefps = 'None'):

    """
    This function performs image rectifications on video files,
    and saves a merged and rectified .avi to the user's drive.

    Inputs:
        video_list (list): 1xK list of paths to video files for each camera
        cameras (array of CameraData objects): 1xK array of CameraData
            intrinsic_list (list): 1x11xK internal calibration data
            extrinsic_list (list): 1x6xK external calibration data
            mType: intrinsic format ('DLT' or 'CIRN')
        xyz (ndarray): XYZ rectification grid
        numFrames (int): number of frames to rectify
        savefps = frames per second to save video at, default is 'None'
            If 'None is specified, video will be saved at the same fps as the first input video

    Returns:
        rect_array: h x w x numFrames array of rectified frames

    """
    import cv2 as cv

    caps = []
    nc = 3
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
            if ind == 0:
                all_frames = frame
                # Save fps at 
                if savefps == 'None':
                    savefps = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            else:
                all_frames = np.append(all_frames, frame, axis=2)

        # All frames from one timestamp sent to mergeRectify
        merged = mergeRectify(all_frames, cameras, grid)

        if i == 0:
            # Create videowriter object to save video to drive
            outFile = initFullFileName(
                video_list[0],
                "video_capture.rect",
                type="avi"
            )
            result = cv.VideoWriter(
                outFile,
                cv.VideoWriter_fourcc("M", "J", "P", "G"),
                savefps,
                (merged.shape[1], merged.shape[0]),
                1,
            )
            # Initialize array of images
            rect_arr = merged.astype(np.uint8)
            IrIndv = np.tile(np.zeros_like(merged), (numFrames,))

        IrIndv[:, :, (i * nc): (i * nc + nc)] = merged.astype(np.uint8)

        # Write to drive
        result.write(merged.astype(np.uint8))

    print("Rectified .avi movie saved to working directory as: " + outFile)
    result.release()

    return rect_arr


"""

II. Pixel and Image Product Functions


"""


def pixelStack(frames, grid, cameras, disp_flag=0):
    """
    Function that creates pixel instruments for use in bathymetric inversion,
    surface current, run-up calculations, or other quantitative analyses.
    Instruments can be made in both world and a local rotated coordinate
    system. However, for ease and accuracy, if planned to use in bathymetric
    inversion, surface current, or run-up applications, it should occur
    in local coordinates.


    Inputs:
        grid: XYZGrid object of desired real world coordinates
        cameras_images (array): N x M x K x num_frames struct of images,
            one image per camera (K) at the desired timestamp
            for rectification (N x M is image height x image width,
            K is number of cameras, num_frames is number of frames)
        cameras (K length array of CameraData objects): contains
            intrinsic_list (list): 1x11xK internal calibration data
            extrinsic_list (list): 1x6xK external calibration data
            origin: local origin (x,y,z,angle)
            mType (string): format of intrinsic matrix,
                'CIRN' is default, 'DLT' is also supported
        sample_freq (int): desired frequency at which to grab pixels.
            This does not factor in camera metadata. User must know frames
            per second of collection. freq=1 means grab pixels at every frame.
            For example, freq=2 means grab pixels at every other frame.
        disp_flag (bool): flag to display output image products
            coords (string): 'geo' or 'local'; if 'geo',
            extrinsics transformed to local but origin is needed to transform

    Returns:
        pixels (ndarray): pixel intensities at xyz points (georectified image).
            Size depends on type of pixels. Axis 3 has the same length
            as number of frames/ sample rate

    """

    # Loop for Collecting Pixel Instrument Data.
    s = grid.X.shape
    nc = cameras[0].nc
    numcams = len(cameras)
    if disp_flag:
        fig, axs = plt.subplots(1, numcams)

    # Iterate through each camera from to produce single merged frame
    for k, (I, calib) in enumerate(zip(frames, cameras)):
        nc = calib.nc
        # Determine if filepath or image was provided
        if isinstance(I, str):
            # Load image from current camera
            image = imageio.imread(I)
        else:
            image = frames[:, :, (k * nc): (k * nc + nc)]

        # Match histograms, init plot
        if k == 0:
            ref = image
        else:
            image = matchHist(ref, image)

        # Find distorted UV points at each XY location in grid
        if calib.Ud == "None":
            if calib.mType == "CIRN":
                Ud, Vd = xyz2DistUV(grid, calib)
            elif calib.mType == "DLT":
                Ud, Vd = dlt2UV(grid, calib)
            else:
                sys.exit("This intrinsics format is not supported")
        else:
            Ud = calib.Ud
            Vd = calib.Vd

        # Grab pixels from image at each position
        ir = getPixels(image, Ud, Vd, s)

        # Initialize array for pixel values
        if k == 0:
            IrIndv = np.tile(np.zeros((s[0], s[1], numcams)), (nc,))

        IrIndv[:, :, (k * nc): (k * nc + nc)] = ir

        if disp_flag:
            # Show pixels on image
            axs[k].xaxis.set_visible(False)
            axs[k].yaxis.set_visible(False)
            axs[k].imshow(
                image.astype(np.uint8),
                cmap="gray",
                vmin=0,
                vmax=255
            )
            mask = ((Ud < image.shape[1])
                    & (Ud > 0)
                    & (Vd < image.shape[0])
                    & (Vd > 0))
            axs[k].scatter(Ud[mask], Vd[mask], s=3, c="r")

    # Merge rectifications of multiple cameras
    pixels = cameraSeamBlend(IrIndv, numcams, nc)
    if disp_flag:
        plt.show()

    # Return pixels
    pixels = pixels.astype(np.uint8)

    return pixels


def imageStats(im_mat, save_flag=0, disp_flag=0):
    """
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

    """

    Dark = np.uint8(np.nanmin(im_mat, axis=2))
    Bright = np.uint8(np.nanmax(im_mat, axis=2))
    Timex = np.uint8(np.nanmean(im_mat, axis=2))
    Variance = np.uint8(np.nanstd(im_mat, axis=2))

    if save_flag:
        imsave("Darkest.jpg", Dark)
        imsave("Brightest.jpg", Bright)
        imsave("Timex.jpg", Timex)
        imsave("Variance.jpg", Variance)

    if disp_flag:
        fig, axs = plt.subplots(1, 4)
        titles = ['Dark','Bright','Timex','Variance']
        for k, im in enumerate([Dark,Bright,Timex,Variance]):
            axs[k].xaxis.set_visible(False)
            axs[k].yaxis.set_visible(False)
            axs[k].set_title(titles[k])
            axs[k].imshow(
                im.astype(np.uint8),
                cmap="gray",
                vmin=0,
                vmax=255
                )
        plt.show()


"""

III. Rectification Grid (XYZGrid) and Camera (CameraData) Classes


"""


class XYZGrid(object):
    """
    Real world XYZ Grid onto which images will be rectified.
    Limits should be given in local coordinates and have the
    same coordinates as camera intrinsic and extrinsic values.

    For multiple rectification tasks with the same desired grid,
    you only have to initialize this class once.
    Then, call mergeRectify for each set of cameras and images.

    If local coordinates are desired and inputs are not yet converted to local,
    user can flag coords = 'geo' and input local origin.

    If intrinsics are in a format other than CIRN (currently the only other
    supported format is DLT), user can flag mType = 'DLT'.

    Args:
        xlims (ndarray): min and max (inclusive) in the x-direction
            (e.g. [-250, 500])
        ylims (ndarray): min and max (inclusive) in the y-direction
            (e.g. [0, 1800])
        dx (float): resolution of grid in x direction
            (same units as camera calibration)
        dy (float): resolution of grid in y direction
            (same units as camera calibration)
        z (float): estimated elevation at every point in the x, y grid
    """

    def __init__(self, xlims, ylims, dx=1, dy=1, z=0):

        self.initXYZ(xlims, ylims, dx, dy, z)

        # Init other params
        self.s = self.X.shape

    def initXYZ(self, xlims, ylims, dx, dy, z):
        """
        Function to initialize the XYZ grid

        xlims and ylims can either be an array of two numbers,
        or one value if a 1D transect is desired.

        """

        if len(xlims) < 2:
            xvec = xlims
        else:
            xvec = np.arange(xlims[0], xlims[1] + dx, dx)

        if len(ylims) < 2:
            yvec = ylims
        else:
            yvec = np.arange(ylims[0], ylims[1] + dy, dy)

        # Make XYZ grid
        self.X, self.Y = np.meshgrid(xvec, yvec)
        self.Z = np.zeros_like(self.X) + z
        x = self.X.copy().T.flatten()
        y = self.Y.copy().T.flatten()
        z = self.Z.copy().T.flatten()
        self.xyz = np.vstack((x, y, z)).T


class CameraData(object):
    """
    Object that contains camera matrices in homogenous coordinates from camera
    extrinsics and intrinsics.Must be re-initialized for each new camera
    (dependent on specific camera's intrinsic and extrinsic calibration).

    If local coordinates are desired and inputs are not yet converted to local,
    user can flag coords = 'geo' and input local origin.

    If intrinsics are in a format other than CIRN (currently the only other
    supported format is DLT), user can flag mType = 'DLT'.

    Arguments:
        intrinsics (list or array): [1 x 11] list of intrinsic values in CIRN
            format or DLT coefficients
        extrinsics (list or array): [1 x 6] list of extrinsic values
            [ x y z azimuth tilt swing] of the camera.
            XYZ should be in the same units as xyz points to be converted.
            Azimith, tilt, and swing should be in radians.
            (azimuth, tilt, and swing should be defined by CIRN convention)
        origin (list or array): local origin, x, y, and angle
        coords (string): tag to indicate if coordinates are in geo or local
        mType (string): tag to indicate if intrinsics are in CIRN or DLT
        nc (int): size of color channel, eg: 1=grayscale, 3=RGB

    Attributes (if mType = 'CIRN'): (in local coords)
        P: [3 x 4] transformation matrix to convert XYZ coordinates to
            distorted UV coordinates.
        K: [3 x 3] K matrix to convert XYZ points to distorted UV coordinates
        R: [3 x 3] Rotation matrix to rotate XYZ world coordinates to camera
            coordinates
        IC: [4 x 3] Translation matrix to translate XYZ world coordinates to
            camera coordinates
        Ud: Distorted u pixel coordinates for current XYZ rectification grid.
            (defaults to "None" until user adds via addUV function)
        Vd: Distorted v pixel coordinates for current XYZ rectification grid.
            (defaults to "None" until user adds via addUV function)

    """

    def __init__(
        self,
        intrinsics,
        extrinsics,
        origin="None",
        coords="local",
        mType="CIRN",
        nc=1
    ):

        self.intrinsics = intrinsics
        self.origin = origin
        self.coords = coords
        self.mType = mType
        self.nc = nc
        self.Ud = "None"
        self.Vd = "None"

        # If in geo coordinates, convert to local
        self.local_extrinsics = extrinsics
        if (self.coords == "geo") & (origin == "None"):
            sys.exit(
                'Local origin is required for a coordinate transform. \
                      If you wish to work in geographical coordinates, \
                      please enter coords = "local", and geo coordinates \
                      will be considered the local working coordinate system.'
            )
        if (self.coords == "geo") & (origin != "None"):
            self.local_extrinsics = self.localTransformExtrinsics()
        if mType != "DLT":
            self.P, self.K, self.R, self.IC = self.getMatrices()

        self.assignCoeffs()

    def getMatrices(self):
        """
        Part of initializer for mType = 'CIRN'.

        Calculates P, K, R, and IC matrices

        Returns:
            P: full camera matrix (3x3)
            K: intrinsic matrix (3x3)
            R: rotation matrix (3x3)
            IC: identity translation matrix
                (3x3 identity matrix - 3x1 translation matrix)

        """

        # Define intrinsic coefficients
        fx = self.intrinsics[4]
        fy = self.intrinsics[5]
        c0U = self.intrinsics[2]
        c0V = self.intrinsics[3]

        # Format K matrix
        K = np.array([[-fx, 0, c0U], [0, -fy, c0V], [0, 0, 1]])

        # Format rotation matrix R
        azimuth = self.local_extrinsics[3]
        tilt = self.local_extrinsics[4]
        swing = self.local_extrinsics[5]

        # Calculate R values
        R = np.empty((3, 3))
        R[0, 0] = (-cos(azimuth) * cos(swing)
                   - sin(azimuth) * cos(tilt) * sin(swing))
        R[0, 1] = (cos(swing) * sin(azimuth)
                   - sin(swing) * cos(tilt) * cos(azimuth))
        R[0, 2] = -sin(swing) * sin(tilt)
        R[1, 0] = (-sin(swing) * cos(azimuth)
                   + cos(swing) * cos(tilt) * sin(azimuth))
        R[1, 1] = (sin(swing) * sin(azimuth)
                   + cos(swing) * cos(tilt) * cos(azimuth))
        R[1, 2] = cos(swing) * sin(tilt)
        R[2, 0] = sin(tilt) * sin(azimuth)
        R[2, 1] = sin(tilt) * cos(azimuth)
        R[2, 2] = -cos(tilt)

        # Format translation matrix IC
        x = self.local_extrinsics[0]
        y = self.local_extrinsics[1]
        z = self.local_extrinsics[2]

        IC = np.array([[1, 0, 0, -x], [0, 1, 0, -y], [0, 0, 1, -z]])

        # Combine K, R, and IC into P
        KR = np.matmul(K, R)
        P = np.matmul(KR, IC)

        # Make homogeneous
        P = P / P[-1, -1]

        return P, K, R, IC

    def localTransformExtrinsics(self):

        """
        Transforms extrinsics in local coordinates to geo, or extrinsics
        in geo coordinates to local.

        Angle should be defined by CIRN convention.

        Returns:
            extrinsics_out (dict): Local or geo extrinsics [x,y,z,a,t,r]
                (a = azimuth, t = tilr, r = roll)

        """

        self.origin[2] = np.deg2rad(self.origin[2])
        extrinsics_out = self.local_extrinsics.copy()
        if self.coords == "geo":
            # World to local
            extrinsics_out[0], extrinsics_out[1] = localTransformPoints(
                self.origin,
                1,
                self.local_extrinsics[0],
                self.local_extrinsics[1]
            )
            extrinsics_out[3] = self.local_extrinsics[3] + self.origin[2]
        else:
            # local to world
            extrinsics_out[0], extrinsics_out[1] = localTransformPoints(
                self.origin,
                0,
                self.local_extrinsics[0],
                self.local_extrinsics[1]
            )
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

    def addUV(self, grid):
        """
        This function precalculates the distorted UV coordinates (UVd)  that
        correspond to a set of world xyz points for the camera matrix contained
        in the self object. 

        This allows the user to save computational time by only having to do 
        this calculation once for one camera-grid pair.

        Arguments:
            self: CameraData object containing the DLT coefficient vector A->L
            grid: XYZGrid object containing real world coords

        Attributes:
            Ud= Nx1 vector of distorted U coordinates for N points.
            Vd= Nx1 vector of distorted V coordinates for N points.

        """

        # Find distorted UV points at each XY location in grid
        if self.mType == "CIRN":
            self.Ud, self.Vd = xyz2DistUV(grid, self)
        elif self.mType == "DLT":
            self.Ud, self.Vd = dlt2UV(grid, self)
        else:
            sys.exit("This intrinsics format is not supported")