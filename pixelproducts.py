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
        images and extrinsics/intrinsics calculated by
        C_singleExtrinsicSolution or F_variableExtrinsicSolution. Section 3 will
        require information concerning the pixel instruments such as type,
        coordinate systems, and dimensions.

    Attributes:
        # A .mat file app ed with _pixInst with the pixel instruments as well as
        # images with the instruments plotted as well as instrument stack images.
        # Rectification metadata will be included in the matfile.

    '''   

    def __init__(self, xlims, ylims, dx=1, dy=1, z=0, coords = 'local', origin = 'None', mType = 'CIRN'):

        Rectifier.__init__(self, xlims, ylims, dx, dy, z, coords, origin, mType)

    def getTimestack(self, im_mat, intrinsic_list, extrinsic_list, sample_rate=1,disp_flag=0):        
        '''
        This function generates pixel instruments (timestacks) for a given set of images and
        corresponding extrinsics/intrinsics. 

        '''  
        '''
        # Loop for Collecting Pixel Instrument Data.
        num_frames = im_mat.shape[3]

        for curr_frame in range(0,num_frames,sample_rate):

        '''
        
        curr_frame=0
        # Array for pixel values
        self.numcams = len(intrinsic_list)
        IrIndv = np.zeros((self.s[0], self.s[1], self.numcams))
        if disp_flag: fig,axs = plt.subplots(1,self.numcams)

        # Iterate through each camera from to produce single merged frame
        for k, (intrinsics, extrinsics) in enumerate(zip(intrinsic_list, extrinsic_list)):
            image = im_mat[:,:,k]

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
        Ir = np.nanmean(IrIndv,axis=2)

        # Return merged frame
        return (Ir.astype(np.uint8))


    def getTimestackFromMovie(self, im_mat, intrinsic_list, extrinsic_list, sample_rate=1):        
        '''
        This function generates pixel instruments (timestacks) for a given set of images and
        corresponding extrinsics/intrinsics. 

        '''  

        # Loop for Collecting Pixel Instrument Data.
        num_frames = im_mat.shape[3]

        for j in range(0,num_frames,sample_rate):
            images = im_mat[:,:,:,j]
            Ir = self.mergeRectify(images, intrinsic_list, extrinsic_list)
            cv.imshow('test',Ir.astype(np.uint8))

        '''
             
                
                # If not First frame, tack on as last dimension (time).
                if j~=1
                    s=size(Igray) 
                    nDim= length(find(s~=1))   # Finds number of actual dimension is =1 (transects) or 2 (grid)nDim
                    
                    # For Gray Scale it is straight forward
                    pixInst(p).Igray=cat(nDim+1,pixInst(p).Igray,Igray)   # Add on last dimension (third if Grid, second if transects)
                    
                    # For RGB it is trickier since MATLAB always likes rgb values in
                    # third dimension.
                    # If a GridGrid Add in the fourth dimension
                    if nDim==2
                        pixInst(p).Irgb=cat(nDim+2,pixInst(p).Irgb,Irgb)   # Add on last dimension (Always Fourth)
                    
                    # If a Transect Grid Add in the second dimension
                    if nDim==1
                        pixInst(p).Irgb=cat(2,pixInst(p).Irgb,Irgb) 
        '''
'''
class pixGrid(PixelInsts):

    #  Example Grid
    #  Note, dx and dy do not need to be equal.

    def __init__(self, **kwargs):
        dx = kwargs.get('dx',self.dx)
        dy = kwargs.get('dy',self.dy)
        xlim = kwargs.get('xlim',self.xlims)
        ylim = kwargs.get('ylim',self.ylims)
        z = kwargs.get('z',self.z)

        X, Y = np.meshgrid(np.arange(xlim[0], xlim[1]+dx, dx), 
                        np.arange(ylim[0], ylim[1]+dx, dy))
        Z = np.zeros_like(X) + z

class (PixelInsts):

    #  VBar (Alongshore/ y Transects)

    def __init__(self,**kwargs):
        x = kwargs.get('x',200)
        dy = kwargs.get('dy',self.dx)
        ylim = kwargs.get('ylim',self.ylims)
        z = kwargs.get('z',self.z)

        Y = np.arange(ylim[0], ylim[1]+dy, dy)
        X = np.zeros_like(Y) + x
        Z = np.zeros_like(X) + z

class runupGrid(PixelInsts):

    #  Runup (Cross-shore Transects)

    def __init__(self,**kwargs):
        y = kwargs.get('y',600)
        dx = kwargs.get('dy',self.dx)
        xlim = kwargs.get('xlim',self.xlims)
        z = kwargs.get('z',self.z)

        X = np.arange(xlim[0], xlim[1]+dx, dx)
        Y = np.zeros_like(X) + y
        Z = np.zeros_like(X) + z
'''