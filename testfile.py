import numpy as np
import time
from datetime import datetime, timedelta

rawPath = "C:/Users/mccan/Documents/Projects/FRF/wamflow/MiscWAM/304_Oct.31/1604152800.Sat.Oct.31_14_00_00.GMT.2020.argus02b.c1.raw"
with open(rawPath, "rb") as my_file:
    separated = dict()
    lines = dict()
    for i in range(25):
        lines[i] = my_file.readline().decode("utf-8").rstrip()

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
    fileHeader = separated
    print(fileHeader['frameCount'])
    w = int(separated['cameraWidth'])
    h = int(separated['cameraHeight'])

    print(time.gmtime(fileHeader['begin']))
    for i in range(10):
        if i == 0:
            time = np.fromfile(file=my_file, dtype=np.uint32, count=1)
            binary = np.fromfile(file=my_file, dtype=np.uint8, count=w * h, offset=28)
            timePhoto = datetime.fromtimestamp(time[0])
            timeZone = timedelta(hours=4)
            print((timePhoto+timeZone).strftime('%Y-%m-%d %H:%M:%S'))

        else:
            time = np.fromfile(file=my_file, dtype=np.uint32, count=1)
            binary = np.fromfile(file=my_file, dtype=np.uint8, count=w * h, offset=28)
            timePhoto = datetime.fromtimestamp(time[0])
            timeZone = timedelta(hours=4)
            print((timePhoto+timeZone).strftime('%Y-%m-%d %H:%M:%S'))