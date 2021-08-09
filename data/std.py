#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# noramlization task
import time
import os, sys
import numpy as np
import pickle

os.chdir('./data/img_patches')
import numpy as np
import pickle

def update_progress(job_title, progress):
    length = 50 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()


arr_dict = dict()

for rt, directs, files in os.walk(os.getcwd(), topdown = True):
    start = time.time()
    #if rt == os.getcwd():
    print(rt)
    if directs != []:
        for direct in directs:
            arr_dict[direct] = {}

    if directs == []:
        l = len(files)
        print('num of files:', len(files))
        sum_arr = 0
        sum_sqr_arr = 0

        for i, file in enumerate(files):
            if i% 20 == 0:
                #print('current number of files dealt: {}'.format(i), end='\r')
                time.sleep(0.1)
                update_progress("Current loading progress:", i*1.0/l)

            #temp_list.append(np.load(os.path.join(rt,file)))
            arr = np.load(os.path.join(rt, file))
            sum_arr += np.sum(arr, axis=(0,1,2))
            sum_sqr_arr += np.sum(np.square(arr), axis=(0,1,2))
        dim = arr.shape
        mean_arr = sum_arr/((i+1)*dim[0]*dim[1]*dim[2])
        std_arr = np.sqrt((sum_sqr_arr - np.square(mean_arr)*(i+1)*dim[0]*dim[1]*dim[2])/((i+1)*dim[0]*dim[1]*dim[2]-1))


        update_progress("Current loading progress:", 1)
        #print('tmp_array: ', tmp_array.shape)
        #mean, std = np.empty(tmp_array.shape[-1]), np.empty(tmp_array.shape[-1])

        #for idx in range(tmp_array.shape[-1]):
        #    mean[idx] = np.mean(tmp_array[:,:,:,:,idx])
        #    std[idx] = np.std(tmp_array[:,:,:,:,idx])
        print(mean_arr, std_arr)
        print(time.time() - start)

        for key in arr_dict.keys():
            if key in rt:
                arr_dict[key] = [mean_arr, std_arr]

        #del temp_list
        #del tmp_array


print(arr_dict)

with open(os.path.join(os.getcwd(), 'mean_std.pkl'), 'wb') as f:
    pickle.dump(arr_dict, f)

#arr_dict = pickle.load(open(os.path.join(os.getcwd(), 'mean_std.pkl'), 'rb'))

i = 1
for d in os.listdir():
    if os.path.isdir(d) and any([d in x for x in ['42_42_3', '48_48_5', '64_64_5', '96_96_7']]):
        os.chdir(d)
        mean, std = arr_dict[d][0], arr_dict[d][1]
        print(mean)
        print(std)

        for file in os.listdir():
            np_arr = np.load(os.path.join(os.getcwd(), file))
            #np_arr = (np_arr - mean)/std
            np_arr = (np_arr + 1)/2    
            if not os.path.exists(os.getcwd()+'_std'):
                os.mkdir(os.getcwd()+'_std')
            if i%100 == 0:
              print(os.path.join(os.getcwd()+'_std', file), ': saving standardized np array file: no:', i)
            np.save(os.path.join(os.getcwd()+'_std', file), np_arr)  ## change the directory
            i += 1
        print(os.path.join(os.getcwd()+'_std', file), ': saving standardized np array file: no:', i)
        os.chdir('..')
