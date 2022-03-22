# CoastalImageLib

CoastalImageLib is a Python- based library that produces common coastal image products intended for quantitative analysis of coastal environments. This library contains functions to georectify and merge multiple oblique camera views, produce statistical image products for a given set of images, create subsampled pixel instruments for use in bathymetric inversion, surface current estimation, run-up calculations, and other quantitative analyses. This library also contains support functions to format camera intrinsic values from various input file formats and convert extrinsic values from geographical to user defined local coordinates. This package intends to be an open- source broadly generalizable front end to future coastal imaging applications, ultimately expanding user accessibility to optical remote sensing of coastal environments. This package was developed and tested on data collected from the Argus Tower, a 43 m tall observation structure in Duck, North Carolina at the US Army Corps of Engineers Field Research Facility that holds six stationary cameras which collect twice- hourly image products of the dry beach and surf zone. Thus, CoastalImageLib also contains functions designed to interface with the file storage and collection system implemented at the Argus Tower. 

## Software Architecture

The following list depicts the library structure for CoastalImageLib, expressed in terms of a hierarchical filesystem. Any classes contained in each .py file are included in italics. Modules, as well as the functions they contain, do not require a specific order in which to be implemented. Suggested workflows are included in the Illustrative Example Script Jupyter notebook contained in the CoastalImageLib repository. Detailed descriptions of each module can be found in the following Software Functionalities section.

    **CoastalImageLib**
        corefunctions.py
            *class XYZGrid*
            *class CameraData*
        supportfunctions.py
        argusIO.py
    
The main user- interactive module is **corefunctions.py**. Two classes are contained in this module: XYZGrid() and CameraData(). These classes bundle data and functionality vital to the rectification process. XYZGrid() holds the real world target grid on which rectification or pixel subsampling will take place. CameraData() holds camera calibration values, and contains a function for extrinsic value transforms, and a function for calculating camera matrices. Users must initialize instances of these classes for each desired rectification grid, and each calibrated camera.

### corefunctions.py

#### Georectification and Merging Multiple Camera Views

The module **corefunctions.py** contains a series of functions that implement fundamental photogrammetry calculations to georectify oblique imagery onto a user- defined real world XYZ grid and merge multiple camera views, for both grayscale and color images. Additionally, this module contains functions to generate statistical image products for a given set of images and corresponding camera extrinsic and intrinsic values, as well as functions to generate pixel instruments for use in bathymetric inversion, surface current, or run-up calculations. For rectification tasks, the user first initializes an XYZGrid() object. The user specifies x and y limits and resolution of the real- world grid in x and y directions. The value given for z should be the estimated water level at the time of data collection relative to the local vertical datum used in specifying extrinsic information. 

Next, the user initializes a CameraData() object for each calibrated camera being utilized. Each instance of this class requires all camera intrinsic and extrinsic values unique to that device. For cameras that have not yet been calibrated and the intrinsic values are not known, the user is directed to the CalTech Camera Calibration library , or other relevant calibration libraries such as the calibration functions contained on OpenCV. Intrinsic values are accepted in the CIRN convention or in the direct linear transform coefficient notation. See the CoastalImageLib User Manual for detailed information on calibration and intrinsic value formatting. The user can also optionally specify the coordinate system being utilized, with the further option of providing the local origin for a coordinate transform. 

If oblique imagery was captured using a non-stationary camera, for example an unmanned aerial vehicle mounted camera, the user is directed to the CIRN Quantitative Coastal Imaging library for calibration and stabilization \cite{CIRN}. Note that this library requires stationary ground control points (GCPs) and stabilization control points (SCPs). See the CIRN Quantitative Coastal Imaging library User Manual \cite{CIRN} for detailed information on GCPs and SCPs.
    
The **corefunctions.py** function mergeRectify() is designed to merge and rectify one or more cameras at one timestamp into a single frame. For multiple subsequent frames, the user can either loop through mergeRectify() and rectify each desired frame on the same XYZ grid, or call the function rectVideos() to merge and rectify frames from one or more cameras provided in video format, sampled at the same time and frame rate. Merging of multiple cameras includes a histogram matching step, where the histogram of the first camera view is used as the reference histogram for balancing subsequent camera views. This step helps remove visible camera seams and improves the congruity of illumination.

#### Statistical Image Products}

The **corefunctions.py** module also contains the function imageStats() to generate statistical image products for a given set of stationary oblique or georectified images contained in a three dimensional array, in either grayscale or color. All image product calculations are taken from the Argus video monitoring convention. The products and their descriptions are as follows:

    1. Brightest: These images are the composite of all the brightest pixel intensities at each pixel location throughout the entire collection.
    2. Darkest: These images are the composite of all the darkest pixel intensities at each pixel location throughout the entire collection.
    3. Timex: Time- exposure (timex) images represent the mathematical time- mean of all the frames captured over the period of sampling. 
    4. Variance: Variance images are found from the variance of image intensities of all the frames captured over the period of sampling. 

#### Pixel Products

The **corefunctions.py** module also contains the function pixelStack() to create subsampled pixel timestacks in either grayscale or color for use in algorithms such as bathymetric inversion, surface current estimation, or run-up calculations. Pixel timestacks show variations in pixel intensity over time. The main pixel products are included below, however additional instruments can be created from these main classes. For example, a single pixel, which may be useful for estimating wave period, can be generated by creating an alongshore transect of length 1.

    1. Grid (also known as Bathy Array in Holman and Stanley 2007 and other publications that reference Argus image products): This is a 2D array of pixels covering the entire nearshore, which can be utilized in bathymetry estimation algorithms. 
    2.  Alongshore/ Y Transect (sometimes referred to as Vbar): This product is commonly utilized in estimating longshore currents.
    3. Cross- shore/ X Transect (sometimes referred to as Runup Array): Cross- shore transects can be utilized in estimating wave runup.


### **supportfunctions.py**

This module contains functions independent of any overarching class or specific workflow, which serve to assist the user in utilizing the core functions. **supportfunctions.py** contains supporting functions to format intrinsic files, convert extrinsic coordinates to and from geographical and local coordinate systems, calculate extrinsic values, and other steps necessary to utilize the core functions of the **CoastalImageLib** library. Additionally, **supportfunctions.py** contains functions that interface with Argus technology, including functions to create Argus compatible filenames from UTC timestamps, and convert raw Argus files into delivery files collected from the Argus tower **Argus** or other mini- Argus systems. Converting raw Argus data utilizes functions contained in **argusIO.py**.
