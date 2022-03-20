import numpy as np
import cv2 as cv
import os
import datetime as dt

"""
Supporting Functions:

    localTransformPoints(xo, yo, ang, flag, xin, yin)
        - This function transforms points in geo coordinates to local.
            User must provide the local origin in geo coordinates
            and the angle between coordinate systems.

    matIntrinsics2Py(files, tag = 'CIRN')
        - This function takes intrinsics and extrinsics
            stored in a .mat file in CIRN convention and formats
            to work in this toolbox

    getYamlDLT(file,cams)
        - This function takes intrinsics and extrinsics
            stored in a .yaml file in Argus convention and formats
            them to work in this toolbox
        - If you are using coordinates in DLT / Walton- m vector format,
            you must tag the rectifier using mType = 'DLT'

    deBayerArgus(cams,rawPaths,startFrame = 0,savePaths='None')
        - Requires argusIO (import argusIO)
        - This function converts frames at one timestamp for multiple cameras
            stored in .raw files to matrices formatted for use in this toolbox
        - savePaths = '' allows the user to save each debayered frame to their
            drive as .png, .jpg, etc. Save paths must include the full filename
            and file extension

    formatArgusFile(cams,folder,epoch):
        -Generates filenames in Argus convention based on the epoch
            time at the start of collection and provided camera tags.
        -Also generates the name of the merged file of all cameras
            to save after rectification (although this function doesn't
            need to be used solely prior to a rectification task).

    initFullFileName(infile,label,type='avi',return_all=False):
        -Function for creating an outfile name consistent with any
            naming convention, infile name defines the naming style

    saveNetCDF(rect_arr,xyz,coords,outfile):
        -Saves the object rect_arr as a netcdf on the user's drive
        -rect_arr can be a rectified image, struct of images,
            pixel instruments, etc

"""

"""
Transform Functions

"""


def localTransformPoints(origin, x_in, y_in, flag):
    """
    Transforms points either in geographical coordinates to local,
    or in local to geographical.
    This requires the local origin in geographical coordinates, as well as the
    angle between coordinate systems in CIRN angle convention.
    See the WAMFlow user manual for more information on CIRN angle convention.

    Args:
        origin (xo, yo, angle): local origin (0,0) in Geographical coordinates.
                Typically xo is E and yo is N coordinate.
                The angle should be the relative angle
                between the new (local) X axis  and old (Geo)
                X axis, positive counter-clockwise from the old (Geo) X.
        flag = 1 or 0 to indicate transform direction
              Geo-->local (1) or
              local-->Geo (0)
        x_in - Local (X) or Geo (E) coord depending on transform direction
        y_in = Local (Y) or Geo (N) coord depending on direction

    Returns:
        x_out - Local (X) or Geo (E) coord depending on direction
        y_out - Local (Y) or Geo (N) coord depending on direction

    """

    if flag == 1:
        # Geo to Local

        # Translate first
        easp = x_in - origin[0]
        norp = y_in - origin[1]

        # Rotate
        x_out = easp * np.cos(origin[2]) + norp * np.sin(origin[2])
        y_out = norp * np.cos(origin[2]) - easp * np.sin(origin[2])

    if flag == 0:
        # Local to Geo

        # Rotate first
        x_out = x_in * np.cos(origin[2]) - y_in * np.sin(origin[2])
        y_out = y_in * np.cos(origin[2]) + x_in * np.sin(origin[2])

        # Translate
        x_out = x_out + origin[0]
        y_out = y_out + origin[1]

    return x_out, y_out


"""
Intrinsic Functions: Loading and Formatting

"""


def loadMatCIRN(files, tag="CIRN"):
    """
    Requires scipy.io package

    This function reads and formats instrinsics from .mat files, in either
    CIRN camera calibration format or CIRN intrinsics format

    Args:
        files: (string) file paths to .mat intrinsics file
        tag: (string) 'CIRN' or 'CalTech', indicates format of .mat file

        Note: if CIRN, extrinsics are likely included in .mat file

    Returns:
        m: array of lists, length is number of cameras, each list contains
            intrinsics for the corresponding camera
        ex: array of lists, length is number of cameras, each list contains
            the extrinsics for the corresponding camera

    """
    from scipy.io import loadmat

    m = np.empty(len(files), object)
    ex = np.empty(len(files), object)

    # Loop through each camera
    for ind in range(len(files)):
        data = loadmat(files[ind])
        camID = files[ind]
        if tag == "CIRN":
            m[ind] = list(np.ravel(data["intrinsics"]))
            ex[ind] = list(np.ravel(data["extrinsics"]))
        else:
            mi = np.empty(11)
            mi[0] = data.nx
            # Number of pixel columns
            mi[1] = data.ny
            # Number of pixel rows
            mi[2] = data.cc[0]
            # U component of principal point
            mi[3] = data.cc[1]
            # V component of principal point
            mi[4] = data.fc[0]
            # U components of focal lengths (in pixels)
            mi[5] = data.fc[1]
            # V components of focal lengths (in pixels)
            mi[6] = data.kc[0]
            # Radial distortion coefficient
            mi[7] = data.kc[1]
            # Radial distortion coefficient
            mi[8] = data.kc[4]
            # Radial distortion coefficient
            mi[9] = data.kc[2]
            # Tangential distortion coefficients
            mi[10] = data.kc[3]
            # Tangential distortion coefficients
            m[ind] = list(np.ravel(mi))

    return m, ex


def loadYamlDLT(file, cams):
    """
    Requires yaml package

    This function reads and formats extrinsics and instrinsics in DLT format
    from .yaml files containing camera data from one or more pre-calibrated
    cameras that are labeled with a camera ID

    Args:
        file: (string) .yaml file path that contains camera data
        cams: (list of strings) camera labels used in .yaml file

    Returns:
        m: array of lists, length is number of cameras, each list contains
            the DLT vector for the corresponding camera
        ex: array of lists, length is number of cameras, each list contains
            the extrinsics for the corresponding camera
    """
    import yaml

    with open(file, "r") as f:
        cameraData = yaml.load(f, Loader=yaml.FullLoader)
    m = np.empty(len(cams), object)
    ex = np.empty(len(cams), object)
    for ind in range(len(cams)):
        camID = cams[ind]
        m[ind] = cameraData[camID]["m"]
        x = cameraData[camID]["x"]
        y = cameraData[camID]["y"]
        z = cameraData[camID]["z"]
        a = cameraData[camID]["azimuth"]
        t = cameraData[camID]["tilt"]
        r = cameraData[camID]["roll"]
        ex[ind] = list([x, y, z, a, t, r])

    return m, ex


def loadJson(jsonfile):
    """
    Reads a .json file into a python dictionary.

    Requires json package.

    """
    import json

    with open(jsonfile, "r") as data:
        dictname = json.loads(data.read())
    return dictname


"""
Argus Functions

"""


def deBayerArgus(cams, rawPaths, frame=0, numFrames=0):
    """
    Requires argusIO

    This function converts frames at one timestamp for multiple cameras
    stored in .raw files to matrices formatted for use in this toolbox

    Args:
        rawPaths (list of strings): paths from which to load .raw files
        frame (int): start frame in .raw file to deBayer
        savePaths (list of strings): optional, paths to save each debayered
            frame to drive as .png, .jpg, etc if desired
            Save paths must include the full filename and file extension
        numFrames: number of frames to deBayer from start frame ("frame"),
            (optional- leave out if only debayering one frame)

    Returns:
        outmat (ndarray): NxMxK matrix of deBayered images, NxM is height
            and width of frame, K is number of cameras

    Based on argusIO code written by Dylan Anderson.

    """
    import argusIO_v2

    cameras = dict()
    frames = dict()

    for p in range(len(cams)):
        # how many raw frames to skip
        cameras[cams[p]] = argusIO_v2.cameraIO(
            cameraID=cams[p],
            rawPath=rawPaths[p],
            startFrame=frame,
            nFrames=numFrames
        )
        cameras[cams[p]].readRaw()
        cameras[cams[p]].deBayer()
        del cameras[cams[p]].raw

        frames[cams[p]] = cameras[cams[p]].imGrayCV

    if numFrames > 1:
        s = frames[cams[0]][:, :, 0].shape
        outmats = np.zeros(
            (s[0], s[1], len(cams), cameras["c1"].nFrames), dtype=np.uint8
        )

        for f in range(cameras["c1"].nFrames):
            for p in range(len(cams)):
                outmats[:, :, p, f] = frames[cams[p]][:, :, f].astype(np.uint8)
    else:
        s = frames[cams[0]].shape
        outmats = np.zeros((s[0], s[1], len(cams)), dtype=np.uint8)

        for p in range(len(cams)):
            outmats[:, :, p] = frames[cams[p]].astype(np.uint8)

    return outmats


def deBayerParallel(i, cams, rawPaths, frame=0, numFrames=0):
    """
    Requires argusIO

    This function converts frames at one timestamp for multiple cameras
    stored in .raw files to matrices formatted for use in this toolbox

    Args:
        rawPaths (list of strings): paths from which to load .raw files
        frame (int): start frame in .raw file to deBayer
        savePaths (list of strings): optional, paths to save each debayered
            frame to drive as .png, .jpg, etc if desired
            Save paths must include the full filename and file extension
        numFrames: number of frames to deBayer from start frame ("frame"),
            (optional- leave out if only debayering one frame)

    Returns:
        outmat (ndarray): NxMxK matrix of deBayered images, NxM is height
            and width of frame, K is number of cameras

    Based on argusIO code written by Dylan Anderson.

    """
    import argusIO_v2
    import multiprocessing as mp

    cameras = dict()
    frames = dict()

    pool = mp.Pool(mp.cpu_count())
    results = []

    for p in range(len(cams)):
        # how many raw frames to skip
        cameras[cams[p]] = argusIO_v2.cameraIO(
            cameraID=cams[p],
            rawPath=rawPaths[p],
            startFrame=frame,
            nFrames=numFrames
        )
        pool.apply_async(cameras[cams[p]].readRaw())

    pool.close()
    pool.join()

    for p in range(len(cams)):
        cameras[cams[p]].deBayer()
        del cameras[cams[p]].raw
        frames[cams[p]] = cameras[cams[p]].imGrayCV

    s = frames[cams[0]][:, :, 0].shape
    outmats = np.zeros((s[0], s[1], len(cams), numFrames))

    for p in range(len(cams)):
        outmats[:, :, p, :] = frames[cams[p]][:, :, :].astype(np.uint8)

    return outmats


def formatArgusFile(cams, folder, epoch, folder_tag = True, file_ext = '.raw', **kwargs):
    """
    Generates filenames in Argus convention based on the epoch
    time at the start of collection and provided camera tags.
    Also generates the name of the merged file of all cameras
    to save after rectification (although this function doesn't
    need to be used solely prior to a rectification task).

    Args:
        cams (list of strings): camera tag for each camera
        epoch (int): epoch/ UTC time at start of collection,
            MUST be an integer
        folder (string): folder where files are/ will be located.
    Keywork Argument:
        'outFileBase'(str): this defines the output file name base,
            default is input file name without camera info
                - overwrites the automatic argus file naming convention
                - add full file name and file extension
        'folder_tag'(bool): set to 'False' if folders
            should not be included in full file path
    Returns:
        paths (list of strings): string of fullfile names in Argus
            convention for each camera
        outFile (string): out filename of the merged file of all cameras;
            used if going into a rectification task

    Note: day_folder (eg: '304_Oct.31/') should not be included in the
        "folder" string, as it changes depending on the epoch time.
        The day_folder will be generated in this function.

    """

    t = dt.datetime.utcfromtimestamp(int(epoch))

    year_start = dt.datetime(t.year, 1, 1)

    jul_str = str((t - year_start).days + 1).zfill(3)
    day_str = t.strftime("%a")
    mon_str = t.strftime("%b")

    day_folder = os.path.join((jul_str
                              + "_"
                              + mon_str
                              + "."
                              + str(t.day).zfill(2)),
                              "")
    file = (
        str(epoch)
        + "."
        + day_str
        + "."
        + mon_str
        + "."
        + str(t.day).zfill(2)
        + "_"
        + str(t.hour).zfill(2)
        + "_"
        + str(t.minute).zfill(2)
        + "_"
        + str(t.second).zfill(2)
        + ".GMT."
        + str(t.year)
        + ".argus02b."
    )

    if folder_tag:
        paths = [
            os.path.join(folder, cams[i], (day_folder + file + cams[i] + file_ext))
            for i in range(len(cams))
        ]
    else:
        paths = [
            os.path.join((file + cams[i] + file_ext))
            for i in range(len(cams))
        ]

    out_path = kwargs.get("outFileBase", "")
    outFile = (
        out_path
        + str(t.day).zfill(2)
        + "_"
        + str(t.hour).zfill(2)
        + "_"
        + str(t.minute).zfill(2)
        + "_merged.avi"
    )

    return paths, outFile


def getThreddsTideTp(t):
    """
    Get peak period and water level from the FRF thredds server

    Tries EOP first for water level, if doesn't work, uses 8m array

    Args:
        t: time in epoch (int)

    Returns:
        Tp: Peak period (float)
        WL: water level (float)
    """
    import datetime as dt
    from netCDF4 import Dataset

    time_obj = dt.datetime.utcfromtimestamp(int(t))
    hr = time_obj.hour
    yr = time_obj.year
    mon_str = str(str(time_obj.month).zfill(2))

    frf_base = "https://chlthredds.erdc.dren.mil/thredds/dodsC/frf/"
    # Peak period Dataset
    ds = Dataset(frf_base
                 + "oceanography/waves/8m-array/"
                 + str(yr)
                 + "/FRF-ocean_waves_8m-array_"
                 + str(yr)
                 + mon_str
                 + ".nc",
                 "r",
    )
    wave_Tp = ds.variables["waveTp"][:]
    thredds_time_Tp = np.asarray(ds.variables["time"][:])

    # Water Level Dataset
    try:
        # Try EOP
        ds2 = Dataset(frf_base
                      + "oceanography/waterlevel/eopNoaaTide/"
                      + str(yr)
                      + "/FRF-ocean_waterlevel_eopNoaaTide_"
                      + str(yr)
                      + mon_str
                      + ".nc",
                      "r",
        )
        waterlevel = ds2.variables["waterLevel"][:]
        thredds_time_WL = np.asarray(ds2.variables["time"][:])
        print("Water level sourced from EOP")
    except:
        # If no EOP, grab from 8m array
        waterlevel = ds.variables["waterLevel"][:]
        thredds_time_WL = np.asarray(ds.variables["time"][:])
        print("Water level sourced from 8m array")

    ind_WL = np.abs(thredds_time_WL - t).argmin()
    ind_Tp = np.abs(thredds_time_Tp - t).argmin()

    # Peak period
    Tp = int(np.ceil(wave_Tp[ind_Tp]))

    # Water level
    WL = round(waterlevel[ind_WL], 2)

    return Tp, WL


"""
Misc. Functions

"""


def initFullFileName(infile, label, type="avi", return_all=False):
    """
    Function for conveniently creating an outfile name consistent with any
    naming convention, given an infile name that contains the naming style
    """

    front = infile.split("/")
    name = ".".join(front[-1].split(".")[:-1])
    folder = ""

    if len(front) > 1:
        folder = "/".join(front[0:-1]) + "/"

    outstr = folder + name + "." + label + "." + type

    if return_all:
        return outstr, name, folder
    else:
        return outstr


def saveNetCDF(rect_arr, xyz, coords, outfile):

    """
    Saves the object rect_arr as a netcdf on the user's drive;
    rect_arr can be a rectified image, struct of images,
    pixel instruments, etc.

    """
    import xarray as xr

    # what data types will be used in the netcdf
    encoding = {"merged": {"dtype": "uint8", "_FillValue": 0}}
    # use xarray to create the netcdf
    ncstruct = xr.Dataset(
        {"merged": (["y", "x"], rect_arr)}, 
        coords={"xyz": xyz, "coord_type": coords, }
    )
    ncstruct.to_netcdf(outfile, encoding=encoding)


def estSharpness(img):
    """
    Estimate image sharpness and contrast

    Input:
        img - np.ndarray representing image
              img read as skimage.io.imread( 'imagefilename.jpg' )
    Returns:
        s,c - sharpness and contrast estimates

    https://stackoverflow.com/questions/6646371/detect-which-image-is-sharper

    """
    array = np.asarray(cv.cvtColor(img, cv.COLOR_BGR2GRAY), dtype=np.int32)
    contrast = array.std()
    gy, gx = np.gradient(array)
    gnorm = np.sqrt(gx ** 2 + gy ** 2)
    sharpness = np.average(gnorm)

    return sharpness, contrast


def avgColor(img):
    """
    Calculate the average pixel intensity of an image

    Input:
        img - np.ndarray representing image
              img read as skimage.io.imread( 'imagefilename.jpg' )
    Returns:
        av - av (np.array of average r, g, b values), 
        avall - average of r,g,b

    """
    av = img.mean(axis=0).mean(axis=0)
    avall = av.mean(axis=0)

    return av, avall