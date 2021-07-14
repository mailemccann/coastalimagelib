import numpy as np
import cv2 as cv


'''
Supporting Functions:

    localTransformPoints(xo, yo, ang, flag, xin, yin)
        - Use this function to transform points in geo coordinates to local
            if you have the local origin in geo coordinates and angle 
            between coordinate systems
        - Similar to Brittany, Chris codes

    matIntrinsics2Py(files,tag = 'CIRN')
        - This function is useful for taking intrinsics and extrinsics 
            stored in a .mat file in CIRN convention and formatting 
            them to work in this toolbox

    getYamlDLT(file,cams)
        - This function is useful for taking intrinsics and extrinsics 
            stored in a .yaml file in Argus convention and formatting 
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
        -Function for conveniently creating an outfile name consistent with any
            naming convention, given an infile name that contains the naming style

    saveNetCDF(rect_arr,xyz,coords,outfile):
        -Saves the object rect_arr as a netcdf on the user's drive
        -rect_arr can be a rectified image, struct of images, pixel instruments, etc

'''

def localTransformPoints(origin, x_in, y_in, flag):
    '''
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
        x_in - Local (X) or Geo (E) coord depending on transformation direction
        y_in = Local (Y) or Geo (N) coord depending on transformation direction
    
    Returns:
        x_out - Local (X) or Geo (E) coord depending on transformation direction
        y_out - Local (Y) or Geo (N) coord depending on transformation direction
    
    '''

    if flag == 1:
        # Geo to Local

        # Translate first
        easp = x_in-origin[0]
        norp = y_in-origin[1]

        # Rotate
        x_out = easp*np.cos(origin[2])+norp*np.sin(origin[2])
        y_out = norp*np.cos(origin[2])-easp*np.sin(origin[2])

    if flag == 0:
        # Local to Geo

        # Rotate first
        x_out = x_in*np.cos(origin[2])-y_in*np.sin(origin[2])
        y_out = y_in*np.cos(origin[2])+x_in*np.sin(origin[2])

        # Translate
        x_out = x_out+origin[0]
        y_out = y_out+origin[1]

    return x_out, y_out
    
def matIntrinsics2Py(files,tag = 'CIRN'):
    ''' 
    Requires scipy.io package

    This function reads and formats instrinsics from .mat files, in either
    CIRN camera calibration format or CIRN intrinsics format

    Args:
        files: (string) file paths to .mat intrinsics file
        tag: (string) 'CIRN' or 'CalTech', indicates format of .mat file

        Note: if CIRN, extrinsics are likely included in .mat file, so I have them in there as well
    
    Returns:
        m: array of lists, length is number of cameras, each list contains the intrinsics 
            for the corresponding camera
        ex: array of lists, length is number of cameras, each list contains the extrinsics 
            for the corresponding camera

    '''
    from scipy.io import loadmat

    m = np.empty(len(files),object)
    ex = np.empty(len(files),object)

    # Loop through each camera
    for ind in range(len(files)):
        data = loadmat(files[ind])
        camID = files[ind]
        if tag == 'CIRN':
            m[ind] = list(np.ravel(data['intrinsics']))
            ex[ind] = list(np.ravel(data['extrinsics']))
        else:
            mi = np.empty(11)
            mi[0] = data.nx;            # Number of pixel columns
            mi[1] = data.ny;            # Number of pixel rows
            mi[2] = data.cc[0];         # U component of principal point
            mi[3]= data.cc[1];          # V component of principal point
            mi[4] = data.fc[0];         # U components of focal lengths (in pixels)
            mi[5] = data.fc[1];         # V components of focal lengths (in pixels)
            mi[6] = data.kc[0];         # Radial distortion coefficient
            mi[7] = data.kc[1];         # Radial distortion coefficient
            mi[8] = data.kc[4];         # Radial distortion coefficient
            mi[9] = data.kc[2];        # Tangential distortion coefficients
            mi[10] = data.kc[3];        # Tangential distortion coefficients
            m[ind] = list(np.ravel(mi))

    return m,ex

def getYamlDLT(file,cams):
    ''' 
    Requires yaml package

    This function reads and formats extrinsics and instrinsics in DLT format from 
    .yaml files containing camera data from one or more pre-calibrated
    cameras that are labeled with a camera ID

    Args:
        file: (string) .yaml file path that contains camera data
        cams: (list of strings) camera labels used in .yaml file

    Returns:
        m: array of lists, length is number of cameras, each list contains the DLT vector 
            for the corresponding camera
        ex: array of lists, length is number of cameras, each list contains the extrinsics 
            for the corresponding camera
    '''
    import yaml 

    with open(file, 'r') as f:
        cameraData = yaml.load(f, Loader=yaml.FullLoader)
    m = np.empty(len(cams),object)
    ex = np.empty(len(cams),object)
    for ind in range(len(cams)):
        camID = cams[ind]
        m[ind] = cameraData[camID]['m']
        x = cameraData[camID]['x']
        y = cameraData[camID]['y']
        z = cameraData[camID]['z']
        a = cameraData[camID]['azimuth']
        t = cameraData[camID]['tilt']
        r = cameraData[camID]['roll']
        ex[ind] = list([x,y,z,a,t,r])

    return m,ex

def deBayerArgus(cams,rawPaths,startFrame = 0,savePaths='None'):
    ''' 
    Requires argusIO

    This function converts frames at one timestamp for multiple cameras
    stored in .raw files to matrices formatted for use in this toolbox

    Args:
        rawPaths (list of strings): paths from which to load .raw files
        startFrame (int): frame in .raw file to deBayer
        savePaths (list of strings): optional, paths to save each debayered
            frame to drive as .png, .jpg, etc if desired
            Save paths must include the full filename and file extension

    Returns: 
        outmat (ndarray): NxMxK matrix of deBayered images, NxM is height 
            and width of frame, K is number of cameras

    Based on argusIO code written by Dylan Anderson. 

    '''
    import argusIO
    
    cameras = dict()
    for p in range(len(cams)):

        # how many raw frames to skip
        cameras[cams[p]] = argusIO.cameraIO(cameraID=cams[p], rawPath=rawPaths[p],skip=startFrame)
        cameras[cams[p]].readRaw()
        cameras[cams[p]].deBayerRawFrameOpenCV()
        del cameras[cams[p]].raw

        if p==0:
            s = cameras[cams[p]].imGrayCV.shape
            outmat = np.zeros((s[0], s[1], len(cams)))

        outmat[:,:,p] = cameras[cams[p]].imGrayCV
        del cameras[cams[p]].imGrayCV

        # Save to drive
        if savePaths != 'None':
            cv.imwrite((savePaths[p] + cams[p] + '.jpg'),outmat[:,:,p])  

    return outmat

def formatArgusFile(cams,folder,epoch):
    '''
    Generates filenames in Argus convention based on the epoch
    time at the start of collection and provided camera tags.
    Also generates the name of the merged file of all cameras
    to save after rectification (although this function doesn't
    need to be used solely prior to a rectification task). 

    Args:
        cams (list of strings): camera tag for each camera 
        epoch (int): epoch/ UTC time at start of collection, MUST be an integer
        folder (string): folder where files are/ will be located.

    Returns:
        paths (list of strings): string of fullfile names in Argus convention for each camera
        outFile (string): out filename of the merged file of all cameras; used if going 
            into a rectification task

    Note: day_folder (eg: '304_Oct.31/') should not be included in the "folder" string, 
        as it changes depending on the epoch time. 
        The day_folder will be generated in this function.
    '''

    import datetime as dt 

    t = dt.datetime.utcfromtimestamp(int(epoch))
    year_start = dt.datetime(t.year, 1, 1)
    jul_str = str((t-year_start).days).zfill(3)

    day_str = t.strftime('%a')
    mon_str = t.strftime('%b')

    day_folder = jul_str + '_' + mon_str + '.' + str(t.day).zfill(2) + '/'
    file = str(epoch)+ '.'+ day_str+ '.'+ mon_str+ '.'+ str(t.day).zfill(2)+ '_'+str(t.hour).zfill(2)+ '_00_00.GMT.'+ str(t.year)+ '.argus02b.'
    paths = [(folder + day_folder + file + cams[i] + '.raw') for i in range(len(cams))]
    outFile = folder + file + 'merged.avi'

    return paths, outFile

def initFullFileName(infile,label,type='avi',return_all=False):
    ''' 
    Function for conveniently creating an outfile name consistent with any
    naming convention, given an infile name that contains the naming style
    '''

    front = infile.split('/')
    name = '.'.join(front[-1].split('.')[:-1])
    folder =''

    if len(front) > 1: 
        folder = '/'.join(front[0:-1]) + '/'

    outstr = folder + name + '.' + label + '.' + type

    if return_all:
        return outstr, name, folder
    else:
        return outstr

def saveNetCDF(rect_arr,xyz,coords,outfile):
    '''
    Saves the object rect_arr as a netcdf on the user's drive
    rect_arr can be a rectified image, struct of images, pixel instruments, etc
    '''
    import xarray as xr

    # what data types will be used in the netcdf
    encoding = {'merged':{'dtype':'uint8','_FillValue':0}}
    # use xarray to create the netcdf
    ncstruct = xr.Dataset({'merged': (['y', 'x'], rect_arr)},
                            coords={'xyz': xyz,
                                    'coord_type': coords,
                                    })
    ncstruct.to_netcdf(outfile,encoding=encoding)