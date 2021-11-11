import multiprocessing as mp

import numpy as np
import sys
sys.path.append('C:/Users/mccan/Documents/GitHub/CoastalImageLib')
import supportfunctions as sf
import corefunctions as cf
import cv2 as cv
import time
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == '__main__':
    # Location of our calibration data, stored in a .yaml file 
    calibration_loc = "C:/Users/mccan/Documents/Projects/FRF/photogrammetry/cameraData.yml"

    # Current order of cameras
    cams = ['c1','c2','c3','c4','c5','c6']

    # Call a support function to format .yaml files into CIRN convention
    m, ex = sf.loadYamlDLT(calibration_loc,cams)
    cameras = np.empty(len(cams),dtype=object)
    for i in range(len(cams)):
        cameras[i] = cf.CameraData(m[i], ex[i], coords = 'local', origin= 'None', mType = 'DLT',nc=1)

    # Current camera tags (camera 4 is camera 0 here)
    cams = ['1','2','3','4','5','6']
    folder = 'D:/WAMToolbox/304_Oct.31/'
    rawPaths = [folder + '1604152800.Sat.Oct.31_14_00_00.GMT.2020.argus02b.c' + i + '.raw' for i in cams]
    vidfile = folder + '.'.join(rawPaths[0].split('/')[-1].split('.')[:-2]) + '.cx.avi'

    # Grab epoch time
    t = int(rawPaths[0].split('/')[-1].split('.')[0])
    
    # Water level/ rectification z value
    Tp, z = sf.getThreddsTideTp(t)

    # Grid boundaries
    xMin = 0
    xMax = 500
    yMin = -500
    yMax = 1200
    dy = 1
    dx = 1
    grid = cf.XYZGrid([xMin,xMax], [yMin,yMax], dx, dy, z)

    # RECTIFY
    nframes = 12 #17*60*2

    ts = time.time()
    outmats = sf.deBayerArgus(cams, rawPaths, frame = 0, numFrames = nframes)
    print('Time to debayer: ' + str(time.time() - ts))

    ts = time.time()
    pool = mp.Pool(mp.cpu_count())
    results = []

    # Define callback function to collect the output in `results`
    def collect_result(result):
        global results
        results.append(result)

    # Use loop to parallelize
    for i in range(nframes):
        pool.apply_async(cf.rectifyParallel, args=(i, outmats[:,:,:,i], cameras, grid), callback=collect_result)

    # Close Pool and let all the processes complete    
    pool.close()
    pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

    print('Time to rectify: '+ str(time.time() - ts))
    
    # Sort results
    ts = time.time()
    results.sort(key=lambda x: x[0])
    for i, r in results:
        if i ==0:
            out = cv.VideoWriter(vidfile,cv.VideoWriter_fourcc('M','J','P','G'), 2, (np.shape(r[:,:,0])[1],np.shape(r[:,:,0])[0]),0)
        out.write(r[:,:,0].astype(np.uint8))
    out.release()
    print('Time to sort and save to drive: '+ str(time.time() - ts))


