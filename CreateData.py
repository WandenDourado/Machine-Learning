import os
import numpy as np


def CrossValidation(sample_size):    
    position = 0
    position_missing = 0
    missing = False
    path = 'img/lfw/'
    entries = os.listdir(path)
    #data = np.empty([len(entries),3], dtype="str")
    data = []
    count = 0
    print(len(entries))
    for entry in range(sample_size):
        count = count + 1
        subpath = path + entries[entry] + '/'

        faces = os.listdir(subpath)
        faces_size = len(faces)
        if(faces_size > 1):
            data.append([subpath+faces[0], subpath+faces[1], 1])
            #data[position][0] = subpath+faces[0]
            #data[position][1] = subpath+faces[1]
            #data[position][2] = 1
            position = position + 1
        else:
            if(missing):
                data[position_missing][1] = subpath+faces[0]
                missing = False
            else:
                data.append([subpath+faces[0], '', 0])
                #data[position][0] = subpath+faces[0]
                #data[position][2] = 0
                missing = True
                position_missing = position
                position = position + 1
        if(range(sample_size) == count+1):
            if(missing):
                data[position_missing][1] = subpath+faces[0]
    return data