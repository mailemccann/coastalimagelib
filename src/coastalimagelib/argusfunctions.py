import numpy as np
import cv2 as cv

'''
Argus specific class and class functions to work with *.raw Argus sensor data

DeBayering workflow:
    1. Initialize cameraIO object
    2. Call readRaw()
    3. Call deBayer()
        - Grayscale frame now saved in self.imGray
                          OR
       Call deBayer(rgb=True) if RBG desired
        - RGB components saved in self.imR, self.imG, self.imB
'''

class cameraIO():
    '''
    Class containing camera data and functions'''

    def __init__(self, **kwargs):

        self.cameraID = kwargs.get('cameraID', 'c1')
        self.rawPath = kwargs.get('rawPath')
        self.nFrames = kwargs.get('nFrames', 1)
        self.startFrame = kwargs.get('startFrame', 0)

    def readRaw(self):
        '''
        This function is utilized for opening *.raw Argus files prior to a debayering task,
        and adds the

        Must call this function before calling any debayering functions
        '''
        with open(self.rawPath, "rb") as my_file:
            self.fh = my_file
            cameraIO.readHeaderLines(self)
            cameraIO.readAIIIrawFullFrame(self)

    def readHeaderLines(self):
        '''
        Reads the header lines of a .raw file to obtain metadata to
        feed into deBayering function

        '''

        separated = dict()
        lines = dict()
        for i in range(25):
            lines[i] = self.fh.readline().decode("utf-8").rstrip()

            temp = lines[i].split(':')
            if len(temp) != 1:
                temp1 = temp[0]
                temp2 = float(temp[1])
                separated[temp1] = temp2
            else:
                temp = lines[i].split(';')
                if len(temp) != 1:
                    temp1 = temp[0]
                    temp2 = temp[1]
                    separated[temp1] = temp2

        self.header = separated
        self.w = int(separated['cameraWidth'])
        self.h = int(separated['cameraHeight'])

    def readAIIIrawFullFrame(self):

        '''
        This function reads AIII raw files and populates the self.raw object
        by reading the *.raw file frame by frame and adding each to the multi-
        dimensional array without cropping the frames

        Attributes:
            self.raw: (ndarray) contains all raw sensor data from one camera,
                read from self.rawPath
        
        '''

        skipoffset = self.startFrame*32 + self.startFrame*self.w*self.h
        self.fh.seek(30, 1)
        if skipoffset > 0:
            for i in range(self.nFrames):

                if i == 0:
                    binary = np.fromfile(file=self.fh, dtype=np.uint8, count=self.w * self.h, offset=skipoffset)
                else:
                    binary = np.fromfile(file=self.fh, dtype=np.uint8, count=self.w * self.h, offset=32)

                data = np.uint8(binary)
                del binary

                if i == 0:
                    I = data.reshape((self.h, self.w))
                else:
                    I = np.dstack((I, data.reshape((self.h, self.w))))

                del data
            self.raw = I

        else:

            for i in range(self.nFrames):
                if i == 0:
                    binary = np.fromfile(file=self.fh, dtype=np.uint8, count=self.w * self.h)
                else:
                    binary = np.fromfile(file=self.fh, dtype=np.uint8, count=self.w * self.h, offset=32)

                data = np.uint8(binary)  # i think this is redundant to above with dtype argument of np.uint8
                del binary
                
                if len(data) == 0:
                    # here adjust our frame count to what is accurate then break out of the loop
                    # and continue processing as normal
                    self.nFrames = i
                    break

                if i == 0:
                    I = data.reshape((self.h, self.w))
                else:
                    I = np.dstack((I, data.reshape((self.h, self.w))))

                del data
            self.raw = I

    def deBayer(self, rgb = False, parallel = False):
        '''
        deBayers a raw frame utilizing the package opencv
        
        Use this function for debayering into an RGB or gray image output

        Deletes from memory the frame that you don't need (eg deletes rgb if rgb =False)
        
        '''

        if rgb:
            imR = np.zeros(np.shape(self.raw), dtype='uint8')
            imB = np.zeros(np.shape(self.raw), dtype='uint8')
            imG = np.zeros(np.shape(self.raw), dtype='uint8')
        else:
            self.imGrayCV = np.zeros(np.shape(self.raw), dtype='uint8')

        if self.nFrames == 1:
            frame = self.raw
            framecopy = np.uint8(frame)
            rgbArray = cv.cvtColor(framecopy, cv.COLOR_BayerGB2BGR)

            if rgb:
                imR[:, :] = rgbArray[:, :, 2]
                imB[:, :] = rgbArray[:, :, 0]
                imG[:, :] = rgbArray[:, :, 1]
            
                self.imR = imR
                self.imG = imG
                self.imB = imB

            else:
                imGrayscale = cv.cvtColor(rgbArray, cv.COLOR_BGR2GRAY)
                self.imGrayCV[:, :] = np.uint8(imGrayscale)

        else:
            for ib in range(self.nFrames):

                frame = self.raw[:, :, ib]
                framecopy = np.uint8(frame)
                rgbArray = cv.cvtColor(framecopy, cv.COLOR_BayerGB2BGR)

                if rgb:
                    imR[:, :, ib] = rgbArray[:, :, 2]
                    imB[:, :, ib] = rgbArray[:, :, 0]
                    imG[:, :, ib] = rgbArray[:, :, 1]
                
                else:
                    imGrayscale = cv.cvtColor(rgbArray, cv.COLOR_BGR2GRAY)
                    self.imGrayCV[:, :, ib] = np.uint8(imGrayscale)

            if rgb:
                self.imR = imR
                self.imG = imG
                self.imB = imB

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

    cameras = dict()
    frames = dict()

    for p in range(len(cams)):
        # how many raw frames to skip
        cameras[cams[p]] = cameraIO(
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
