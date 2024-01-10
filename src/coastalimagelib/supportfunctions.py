import numpy as np

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

    initFullFileName(infile,label,type='avi',return_all=False):
        -Function for creating an outfile name consistent with any
            naming convention, infile name defines the naming style

    saveNetCDF(rect_arr,xyz,coords,outfile):
        -Saves the object rect_arr as a netcdf on the user's drive
        -rect_arr can be a rectified image, struct of images,
            pixel instruments, etc
    
    estSharpness

    avgColor

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
    import cv2 as cv

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
