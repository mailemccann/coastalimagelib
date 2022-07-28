import numpy as np
import cv2

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

        import cv2 as cv

        if rgb:
            imR = np.zeros(np.shape(self.raw), dtype='uint8')
            imB = np.zeros(np.shape(self.raw), dtype='uint8')
            imG = np.zeros(np.shape(self.raw), dtype='uint8')
        else:
            self.imGrayCV = np.zeros(np.shape(self.raw), dtype='uint8')

        if self.nFrames == 1:
            frame = self.raw
            framecopy = np.uint8(frame)
            rgbArray = cv2.cvtColor(framecopy, cv2.COLOR_BayerGB2BGR)

            if rgb:
                imR[:, :] = rgbArray[:, :, 2]
                imB[:, :] = rgbArray[:, :, 0]
                imG[:, :] = rgbArray[:, :, 1]
            
                self.imR = imR
                self.imG = imG
                self.imB = imB

            else:
                imGrayscale = cv2.cvtColor(rgbArray, cv2.COLOR_BGR2GRAY)
                self.imGrayCV[:, :] = np.uint8(imGrayscale)

        else:
            for ib in range(self.nFrames):

                frame = self.raw[:, :, ib]
                framecopy = np.uint8(frame)
                rgbArray = cv2.cvtColor(framecopy, cv2.COLOR_BayerGB2BGR)

                if rgb:
                    imR[:, :, ib] = rgbArray[:, :, 2]
                    imB[:, :, ib] = rgbArray[:, :, 0]
                    imG[:, :, ib] = rgbArray[:, :, 1]
                
                else:
                    imGrayscale = cv2.cvtColor(rgbArray, cv2.COLOR_BGR2GRAY)
                    self.imGrayCV[:, :, ib] = np.uint8(imGrayscale)

            if rgb:
                self.imR = imR
                self.imG = imG
                self.imB = imB