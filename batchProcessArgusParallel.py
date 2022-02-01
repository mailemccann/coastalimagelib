import multiprocessing as mp
import os
import numpy as np
import sys
sys.path.append('C:/Users/mccan/Documents/GitHub/CoastalImageLib')
import supportfunctions as sf
import corefunctions as cf
import cv2 as cv
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == '__main__':
    # RAW image locations
    folder = '/mnt/gaia/peeler/argus/argus02bFullFrame/2021/'

    # traverse root directory, and list directories as dirs and files as files
    epoch = []
    cam_tag = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if os.path.splitext(file)[1] == '.raw':
                file_splt = file.split('.')
                cam_tag.append(file_splt[7])
                epoch.append(int(file_splt[0]))
    
    # Make camera array from the number/ tags of cameras available during that collect
    final_epoch = []
    cam_array = []
    epoch = np.asarray(epoch)
    cam_tag = np.asarray(cam_tag)
    for e_ind, e_val in enumerate(epoch):
        if e_val not in final_epoch:
            final_epoch.append(e_val)
            cam_array.append(cam_tag[epoch==e_val])

    nframes = 120*17  # 2 fps for 17 min

    # Grid boundaries
    xMin = 0
    xMax = 500
    yMin = -500
    yMax = 1200
    dy = 1
    dx = 1
    yamlLoc = '/home/spike/maile/cameraData.yml'

    for ind, val in enumerate(final_epoch):
        
        # Call a support function to format .yaml files into CIRN convention
        cams = cam_array[ind]
        m, ex = sf.loadYamlDLT(yamlLoc,cams)
        cameras = np.empty(len(cams),dtype=object)
        for i in range(len(cams)):
            cameras[i] = cf.CameraData(m[i], ex[i], coords = 'local', origin= 'None', mType = 'DLT',nc=1)
        
        paths, outFile = sf.formatArgusFile(cams,folder,val)
        print('Rectified movie will be saved to: ' + outFile)
        
        # Water level/ rectification z value
        Tp, z = sf.getThreddsTideTp(val)
        grid = cf.XYZGrid([xMin,xMax], [yMin,yMax], dx, dy, z)

        # RECTIFY
        outmats = sf.deBayerArgus(cams, paths, frame = 0, numFrames = nframes)
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

        # Sort results
        results.sort(key=lambda x: x[0])
        for i, r in results:
            if i ==0:
                out = cv.VideoWriter(outFile,cv.VideoWriter_fourcc('M','J','P','G'), 2, (np.shape(r[:,:,0])[1],np.shape(r[:,:,0])[0]),0)
            out.write(r[:,:,0].astype(np.uint8))
        out.release()


