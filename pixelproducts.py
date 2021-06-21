import numpy as np
import cv2 as cv
from scipy.interpolate import RegularGridInterpolator as reg_interp
from corefuncs import Rectifier


class PixelInsts(Rectifier):

    '''
    Class that contains pixel instruments for use in bathymetric inversion,
    surface current, or run-up calculations. Instruments can be made in
    both world and a local rotated coordinate system. However, for ease and
    accuracy, if planned to use in bathymetric inversion, surface current,
    or run-up applications, it should occur in local coordinates.

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

    def __init__(self, rect, save_flag=0):

        Rectifier.__init__(self, rect.xlims, rect.ylims, rect.dx, rect.z, rect.coords, rect.origin, rect.mType)
        self.save_flag = save_flag

        self.ims = []
        self.num_images = 0

    def createInstruments(self,grids):        
        '''
        This function generates pixel instruments for a given set of images and
        corresponding extrinsics/intrinsics. 

        '''  

        # Loop for Collecting Pixel Instrument Data.
        I = []
        '''
        for j in self.num_images:
            
            # For Each Camera
            for k in len(self.cams):
                # Load Image
                I[k] = 1 # Add image!!
            
            #  Loop for Each Pixel Instrument
            for p in grids:
                
                # Check if a time varying Z was specified. If not, wil just use constant Z
                # Add this functionality
                
                #Pull Correct Extrinsics out, Corresponding In time
                for k=1:camnum
                    extrinsics{k}=Extrinsics{k}(j,:) 
                
                intrinsics=Intrinsics 
                
                # Pull RGB Pixel Intensities
                [Ir]= mergeRectify(I,intrinsics,extrinsics,p.X,p.Y,p.Z,0)
                
                
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

class vBarGrid(PixelInsts):

    #  VBar (Alongshore/ y Transects)

    def __init__(self,**kwargs):
        x = kwargs.get('x',200)
        dy = kwargs.get('dy',self.dx)
        ylim = kwargs.get('ylim',self.ylims)
        z = kwargs.get('z',self.z)

        Y = np.arange(ylim[0], ylim[1]+dy, dy)
        X = np.zeros_like(Y) + x
        Z = np.zeros_like(X) + z

def runupGrid(PixelInsts):

    #  Runup (Cross-shore Transects)

    def __init__(self,**kwargs):
        y = kwargs.get('y',600)
        dx = kwargs.get('dy',self.dx)
        xlim = kwargs.get('xlim',self.xlims)
        z = kwargs.get('z',self.z)

        X = np.arange(xlim[0], xlim[1]+dx, dx)
        Y = np.zeros_like(X) + y
        Z = np.zeros_like(X) + z