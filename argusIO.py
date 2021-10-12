import yaml
import numpy as np
import cv2
import scipy.spatial.qhull as qhull

'''
Argus specific class and class functions to work with *.raw Argus sensor data

DeBayering workflow:
    1. Initialize cameraIO object
    2. Call readRaw()
    3. Call deBayerRawFrameOpenCV() if grayscale desired
        - Grayscale frame now saved in self.imGray
                          OR
       Call deBayerRawFrameOpenCVForColor() if RBG desired
        - RGB components saved in self.imR, self.imG, self.imB
'''

class cameraIO():
    '''
    Class containing camera data and functions'''

    def __init__(self, **kwargs):

        self.cameraID = kwargs.get('cameraID', 'c1')
        self.rawPath = kwargs.get('rawPath')
        self.yamlPath = kwargs.get('yamlPath', 'D:/WAMToolbox/cameraData.yml')
        self.nFrames = kwargs.get('nFrames', 1)
        self.xMin = kwargs.get('xMin', 0)
        self.xMax = kwargs.get('xMax', 500)
        self.yMin = kwargs.get('yMin', 0)
        self.yMax = kwargs.get('yMax', 1200)
        self.dx = kwargs.get('dx', 1)
        self.skip = kwargs.get('skip', 0)
        self.verbose = kwargs.get('verbose', True)  # turn on/off print statements
        self.parallel = kwargs.get('parallel', True)
    def getCameraData(self):

        '''
        Loads camera data from yamlPath file containing camera intrinsic and extrinsic values
        
        '''
        with open(self.yamlPath, 'r') as f:
            self.cameraData = yaml.load(f, Loader=yaml.FullLoader)

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

    def readAIIIrawCropped(self):
        
        '''
        This function reads AIII raw files and populates the self.raw object
        by reading the *.raw file frame by frame and adding each to the multi-
        dimensional array and cropping the frames based on the user defined
        umin, umax, vmin, vmax

        Attributes:
            self.raw: (ndarray) contains all raw sensor data from one camera,
                read from self.rawPath
        
        '''
        self.umin = self.cameraData[self.cameraID]['umin']
        self.umax = self.cameraData[self.cameraID]['umax']
        self.vmin = self.cameraData[self.cameraID]['vmin']
        self.vmax = self.cameraData[self.cameraID]['vmax']
        skipoffset = self.skip*32 + self.skip*self.w*self.h
        self.fh.seek(30, 1)

        print("working on camera", self.cameraID)

        if skipoffset > 0:
            for i in range(self.nFrames):
                print("Reading frame {}".format(i))
                if i == 0:
                    binary = np.fromfile(file=self.fh, dtype=np.uint8, count=self.w * self.h, offset=skipoffset)
                else:
                    binary = np.fromfile(file=self.fh, dtype=np.uint8, count=self.w * self.h, offset=32)

                data = np.uint8(binary)
                del binary
                if i == 0:
                    crop = data.reshape((self.h, self.w))
                    del data
                    I = crop[self.vmin:(self.vmax), self.umin:(self.umax)]
                else:
                    crop = data.reshape((self.h, self.w))
                    del data
                    I = np.dstack((I, crop[self.vmin:(self.vmax), self.umin:(self.umax)]))

                del crop

        else:

            for i in range(self.nFrames):
                print("Reading frame {}".format(i))
                if i == 0:
                    binary = np.fromfile(file=self.fh, dtype=np.uint8, count=self.w * self.h)
                else:
                    binary = np.fromfile(file=self.fh, dtype=np.uint8, count=self.w * self.h, offset=32)

                data = np.uint8(binary)
                del binary
                if i == 0:
                    crop = data.reshape((self.h, self.w))
                    del data
                    I = crop[self.vmin:(self.vmax), self.umin:(self.umax)]
                else:
                    crop = data.reshape((self.h, self.w))
                    del data
                    I = np.dstack((I, crop[self.vmin:(self.vmax), self.umin:(self.umax)]))

                del crop

        self.raw = I

    def readAIIIrawFullFrame(self):

        '''
        This function reads AIII raw files and populates the self.raw object
        by reading the *.raw file frame by frame and adding each to the multi-
        dimensional array without cropping the frames

        Attributes:
            self.raw: (ndarray) contains all raw sensor data from one camera,
                read from self.rawPath
        
        '''

        skipoffset = self.skip*32 + self.skip*self.w*self.h
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
                if i%20 == 0 and self.verbose==True: print(f"Reading rawFile {i/self.nFrames*100:.1f} %")
                if i == 0:
                    binary = np.fromfile(file=self.fh, dtype=np.uint8, count=self.w * self.h)
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

    def deBayerRawFrame(self):
        '''
        deBayers a raw frame using Argus framework, adapted from
        original Matlab code from John Stanley

        Attributes:
            self.imGray:
        
        '''

        im = np.zeros(np.shape(self.raw))
        imGray = np.zeros(np.shape(self.raw))
        imR = np.zeros(np.shape(self.raw))
        imB = np.zeros(np.shape(self.raw))
        imG = np.zeros(np.shape(self.raw))

        for ib in range(self.nFrames):

            print('DeBayering frame {}'.format(ib))
            frame = self.raw[:, :, ib]

            # python code adpoted from Argus framework
            Bmin = [1, 3]
            Rmin = [2, 2]
            Gmin = [2, 3, 3, 2]

            B = frame.copy()
            xx = np.arange(Bmin[0], self.h, 2)
            for j in range(Bmin[1], self.w - 1, 2):
                B[xx - 1, j - 1] = np.mean((B[xx - 1, j - 2], B[xx - 1, j]), axis=0)

            for i in range(Bmin[0] + 1, self.h - 1, 2):
                B[i - 1, :] = np.mean((B[i - 2, :], B[i, :]), axis=0)

            R = frame.copy()
            xx = np.arange(Rmin[0], self.h - 1, 2)
            for j in range(Rmin[1], self.w - 1, 2):
                R[xx - 1, j - 1] = np.mean((R[xx - 1, j - 2], R[xx - 1, j]), axis=0)
            for i in range(Rmin[0] + 1, self.h - 1, 2):
                R[i - 1, :] = np.mean((R[i - 2, :], R[i, :]), axis=0)

            G = frame.copy()
            for i in range(Gmin[0], self.h - 1, 2):
                for j in range(Gmin[1], self.w - 1, 2):
                    G[i - 1, j - 1] = np.mean((G[i - 2, j - 1], G[i, j - 1], G[i - 1, j], G[i - 1, j - 2]), axis=0)

            for i in range(Gmin[2], self.h - 1, 2):
                for j in range(Gmin[3], self.w - 1, 2):
                    G[i - 1, j - 1] = np.mean((G[i - 2, j - 1], G[i, j - 1], G[i - 1, j], G[i - 1, j - 2]), axis=0)

            rgbArray = np.zeros((2048, 2448, 3), 'uint8')
            rgbArray[..., 0] = R
            rgbArray[..., 1] = G
            rgbArray[..., 2] = B
            imGrayscale = cv2.cvtColor(rgbArray, cv2.COLOR_BGR2GRAY)

            imGray[:, :, ib] = imGrayscale

            self.imGray = imGray

    def deBayerRawFrameOpenCVForColor(self):
        '''
        deBayers a raw frame utilizing the package opencv
        
        Use this function for debayering into an RGB image output
        
        '''

        im = np.zeros(np.shape(self.raw), dtype='uint8')

        imR = np.zeros(np.shape(self.raw), dtype='uint8')
        imB = np.zeros(np.shape(self.raw), dtype='uint8')
        imG = np.zeros(np.shape(self.raw), dtype='uint8')

        for ib in range(self.nFrames):

            print('DeBayering frame {}'.format(ib))
            frame = self.raw[:, :, ib]
            framecopy = np.uint8(frame)
            rgbArray = cv2.cvtColor(framecopy, cv2.COLOR_BayerGB2BGR)

            imR[:, :, ib] = rgbArray[:, :, 2]
            imB[:, :, ib] = rgbArray[:, :, 0]
            imG[:, :, ib] = rgbArray[:, :, 1]
        
            self.imR = imR
            self.imG = imG
            self.imB = imB

    def deBayerRawFrameOpenCV(self):

        '''
        deBayers a raw frame utilizing the package opencv
        
        Use this function for debayering into an grayscale
        image output
        
        '''

        im = np.zeros(np.shape(self.raw), dtype='uint8')
        self.imGrayCV = np.zeros(np.shape(self.raw), dtype='uint8')
        imR = np.zeros(np.shape(self.raw), dtype='uint8')
        imB = np.zeros(np.shape(self.raw), dtype='uint8')
        imG = np.zeros(np.shape(self.raw), dtype='uint8')

        if self.nFrames == 1:  # used for limited memory operations
            frame = self.raw
            framecopy = np.uint8(frame)
            rgbArray = cv2.cvtColor(framecopy, cv2.COLOR_BayerGB2BGR)

            imGrayscale = cv2.cvtColor(rgbArray, cv2.COLOR_BGR2GRAY)
            self.imGrayCV[:, :] = np.uint8(imGrayscale)
        elif self.parallel is True:
            import multiprocessing as mp
            pool = mp.Pool(mp.cpu_count() - 1)
            # https://www.machinelearningplus.com/python/parallel-processing-python/
            imGrayscale = pool.starmap_async(cv2.cvtColor, [(np.uint8(self.raw[:,:,ib]), cv2.COLOR_BayerGB2GRAY)
                                                        for ib in range(self.nFrames)]).get()
            pool.close()
            self.imGrayCV = np.array(imGrayscale, dtype=np.uint8)

        else:
            for ib in range(self.nFrames):
                if ib % 20 == 0 and self.verbose==True:
                    print(f'debayering frame {ib}/{self.nFrames}')
                frame = self.raw[:, :, ib]
                framecopy = np.uint8(frame)
                rgbArray = cv2.cvtColor(framecopy, cv2.COLOR_BayerGB2BGR)
                imGrayscale = cv2.cvtColor(rgbArray, cv2.COLOR_BGR2GRAY)

                self.imGrayCV[:, :, ib] = np.uint8(imGrayscale)
        
    '''

    The following functions are utilized in rectification tasks specific to Argus,
    however they are no longer supported and maintained. For further photogrammetry
    tasks, refer to the corefunctions module within CoastalImageLib.
    
    '''

    def uvToXY(self):
        # Do we trust the edges?
        self.umin = self.cameraData[self.cameraID]['umin']
        self.umax = self.cameraData[self.cameraID]['umax']
        self.vmin = self.cameraData[self.cameraID]['vmin']
        self.vmax = self.cameraData[self.cameraID]['vmax']
        self.m = self.cameraData[self.cameraID]['m']
        Uu, Vv = np.meshgrid(np.arange(self.umin, self.umax, 1), np.arange(self.vmin, self.vmax, 1))

        U = Uu.ravel()
        V = Vv.ravel()
        # Change from Walton m-vector notation to DLT notation so don't have to use subscripts
        A = self.m[0]
        B = self.m[1]
        C = self.m[2]
        D = self.m[3]
        E = self.m[4]
        F = self.m[5]
        G = self.m[6]
        H = self.m[7]
        J = self.m[8]
        K = self.m[9]
        L = self.m[10]

        # Assign variable names to coefficients derived in solving for x,y, or z
        M = (E * U - A)
        N = (F * U - B)
        O = (G * U - C)
        P = (D - U)
        Q = (E * V - H)
        R = (F * V - J)
        S = (G * V - K)
        T = (L - V)

        Z = 0 * np.ones(len(U), )
        X = np.divide((np.multiply(np.multiply(N, S) - np.multiply(R, O), Z) + (np.multiply(R, P) - np.multiply(N, T))),
                      (np.multiply(R, M) - np.multiply(N, Q)))
        Y = np.divide((np.multiply(np.multiply(M, S) - np.multiply(Q, O), Z) + (np.multiply(Q, P) - np.multiply(M, T))),
                      (np.multiply(Q, N) - np.multiply(M, R)))

        self.Xx = np.reshape(X, [(self.vmax - self.vmin), (self.umax - self.umin)])
        self.Yy = np.reshape(Y, [(self.vmax - self.vmin), (self.umax - self.umin)])
        self.X = X
        self.Y = Y

    def cropFrames(self):

        '''
        Function called to crop a debayered frame
        
        '''

        self.gray = np.zeros(((self.vmax - self.vmin), (self.umax - self.umin), self.nFrames), dtype='uint8')

        if self.nFrames == 1:
            self.gray = self.imGrayCV[self.vmin:(self.vmax), self.umin:(self.umax)]
        else:
            for i in range(self.nFrames):
                self.gray[:, :, i] = self.imGrayCV[self.vmin:(self.vmax), self.umin:(self.umax), i]

    def frameInterp(self):

        self.xy = np.zeros([len(self.X), 2])
        self.xy[:, 0] = self.Y.flatten()
        self.xy[:, 1] = self.X.flatten()

        self.xnew = np.arange(self.xMin, self.xMax, self.dx)
        self.ynew = np.arange(self.yMin, self.yMax, self.dx)

        xgrid, ygrid = np.meshgrid(self.xnew, self.ynew)
        self.uv = np.zeros([xgrid.shape[0] * xgrid.shape[1], 2])
        self.uv[:, 0] = ygrid.flatten()
        self.uv[:, 1] = xgrid.flatten()

        cameraIO.interpWeights(self)

        self.grayInterp = np.zeros((len(self.ynew), len(self.xnew), self.nFrames), dtype='uint8')
        print(np.shape(self.grayInterp))

        if self.nFrames == 1:
            returned = cameraIO.interpolate(self, self.gray)
            self.grayInterp = returned.reshape(xgrid.shape[0], xgrid.shape[1])
        else:
            for i in range(self.nFrames):
                #print("interpolating frame {}".format(i))
                returned = cameraIO.interpolate(self, self.gray[:, :, i])
                self.grayInterp[:, :, i] = returned.reshape(xgrid.shape[0], xgrid.shape[1])
        
        self.xgrid = xgrid
        self.ygrid = ygrid

    def interpWeights(self):
        #def interp_weights(xy, uv,d=2):
        #print("calculating Delauny for camera", self.cameraID)
        tri = qhull.Delaunay(self.xy)
        simplex = tri.find_simplex(self.uv)
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = self.uv - temp[:, 2]
        bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
        self.vtx = vertices
        self.wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    def interpolate(self, values):
        ret = np.einsum('nj,nj->n', np.take(values, self.vtx), self.wts)
        ret[np.any(self.wts < 0, axis=1)] = np.nan
        return ret