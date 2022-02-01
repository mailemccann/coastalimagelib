import multiprocessing as mp
import os
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
    # RAW image locations
    folder = "C:/Users/mccan/Documents/Projects/FRF/wamflow/MiscWAM/304_Oct.31/" # '/mnt/gaia/peeler/argus/argus02bFullFrame/2021/'

    # traverse root directory, and list directories as dirs and files as files
    epoch = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if os.path.splitext(file)[1] == '.raw':
                file_splt = file.split('.')
                e = int(file_splt[0])
                epoch.append(e)
    
    # Make sure at least two cameras were collecting at that time
    final_epoch = []
    for e_ind, e_val in enumerate(epoch):
        e_count = epoch.count(e_val)
        if e_count > 1:
            if e_val not in final_epoch:
                final_epoch.append(e_val)

    print(sorted(final_epoch))

    cams = ['c1','c2','c3','c4','c5','c6']

    numFrames = 120*17  # 2 fps for 17 min

    # Grid boundaries
    xMin = 0
    xMax = 500
    yMin = -200
    yMax = 1600
    yamlLoc = folder + 'cameraData.yml'

    # Call a support function to format .yaml files into CIRN convention
    m, ex = sf.loadYamlDLT(yamlLoc,cams)
    cameras = np.empty(len(cams),dtype=object)
    for i in range(len(cams)):
        cameras[i] = cf.CameraData(m[i], ex[i], coords = 'local', origin= 'None', mType = 'DLT',nc=1)

    for currep in final_epoch:
        paths, outFile = sf.formatArgusFile(cams,folder,currep)
        print(outFile)
        paths = [folder + '1604152800.Sat.Oct.31_14_00_00.GMT.2020.argus02b.' + c + '.raw' for c in cams]

        # Water level/ rectification z value
        Tp, z = sf.getThreddsTideTp(currep)

        # Grid boundaries
        xMin = 0
        xMax = 500
        yMin = -500
        yMax = 1200
        dy = 1
        dx = 1
        grid = cf.XYZGrid([xMin,xMax], [yMin,yMax], dx, dy, z)

        # RECTIFY
        nframes = 5 #17*60*2

        ts = time.time()
        outmats = sf.deBayerArgus(cams, paths, frame = 0, numFrames = nframes)
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
                out = cv.VideoWriter(folder + 'test.avi', cv.VideoWriter_fourcc(*'RGBA'), 2, (np.shape(r[:,:,0])[1],np.shape(r[:,:,0])[0]),0)
            out.write(r[:,:,0].astype(np.uint8))
        out.release()
        #print('Time to sort and save to drive: '+ str(time.time() - ts))


