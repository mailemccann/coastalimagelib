import pixelproducts as pf
import numpy as np
import cv2 as cv
import supportfuncs as sf
import corefuncs as cf
import matplotlib.pyplot as plt

# RAW image locations
folder = 'C:/Users/mccan/Desktop/WAM_Bungo/304_Oct.31/'
cams = ['c1','c2','c3','c4','c5','c6']
yamlLoc = 'C:/Users/mccan/Desktop/cameraData.yml'
m, ex = sf.getYamlDLT(yamlLoc,cams) 

numFrames = 120 # 2 fps for 17 min
video_list = [folder + '1604152800.Sat.Oct.31_14_00_00.GMT.2020.' + i + '.avi' for i in cams]

# Grid boundaries
xMin = 60
xMax = 100
yMin = 300
yMax = 700

#rect_object = cf.Rectifier([xMin,xMax], [yMin,yMax], dx=1, dy=1, z=0, mType = 'DLT')

caps = []
# Append each video capture object into structure
for path in video_list:
    caps.append(cv.VideoCapture(path))


pix_object = pf.PixelStack([200], [500,700], dx=.5, dy=2, z=0, mType='DLT')
# Loop through each frame
for i in range(numFrames):
    for ind, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            break

        # Add each frame into one object
        if ind==0:
            frames = np.empty((frame.shape[0],frame.shape[1],len(video_list)))
        frames[:,:,ind] = frame[:,:,0]
        

    if i ==0:   
        trans = pix_object.getTimestack(frames, m, ex, disp_flag=1)
        fig,axs = plt.subplots()
        axs.imshow(np.flipud(trans.astype(np.uint8)),cmap='gray', vmin=0, vmax=255)
        plt.show()
    else:
        merged = pix_object.getTimestack(frames, m, ex)
        trans = np.vstack((trans,merged))

cv.imwrite('testxtransect.jpg',trans.astype(np.uint8))
