#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:29:22 2019

@author: rccuser
"""
import os
import pickle
import numpy as np
import pandas as pd

def singlePatch(patch, arr_dir, ProxID, patch_dim):
    (st_d, ed_d, st_h, ed_h, st_w, ed_w) = patch[0]  # coord info list of each image mode; the 0-th element is always the patch coords.
    label = patch[-2]   # boolean
    # connect with coresponding resized image
    rz_img = np.load(os.path.join(arr_dir, ProxID)+'.npy')  # resized image filename
    patch_img = rz_img[st_d:ed_d,st_h:ed_h,st_w:ed_w]
    assert (patch_img.shape == patch_dim), 'patched images have the mismatched dimensions'
    patch_img = np.moveaxis(patch_img, 0, -1)
     
    return patch_img, label

def singlePatch_multiMod(patch, arr_dir, patch_dim):
    LABEL = None
    patch_img_assem = [[],[],[]]
    for mod in patch.keys():
        (st_d, ed_d, st_h, ed_h, st_w, ed_w) = patch[mod][0]  # coord info list of each image mode; the 0-th element is always the patch coords.
        label = patch[mod][-2]   # boolean
        if LABEL is not None:
            #assert LABEL == label, print(patch,': label mismatch across the mods')
            if LABEL != label:
                print(patch,': label mismatch across the mods')
                return None, LABEL
        LABEL = label
        # connect with coresponding resized image
        rz_img = np.load(os.path.join(arr_dir, mod)+'.npy')  # resized image filename
        patch_img = rz_img[st_d:ed_d,st_h:ed_h,st_w:ed_w]
        assert (patch_img.shape == patch_dim), 'patched images have the mismatched dimensions'
        patch_img = np.moveaxis(patch_img, 0, -1)
        if 't2tse' in mod:
            patch_img_assem[0] = patch_img
        elif 'ADC' in mod:
            patch_img_assem[1] = patch_img
        elif 'BVAL' in mod:
            patch_img_assem[2] = patch_img

    patch_assem = np.moveaxis(np.array(patch_img_assem), 0, -1)
    return patch_assem, LABEL


def extract_imgPatchCoord(ID, arr_path, ProxID, dest_path, pd_frame, patch_dim, ktrans = False):
    print(ProxID)        
    for patch in ID.keys(): # number of patches in individual patch_size
    #print(patch)
        filepath = os.path.join(dest_path,ProxID+'_'+patch)
        if ktrans:
            img_arrays, label = singlePatch(ID[patch], arr_path, ProxID, patch_dim)
        else:
            img_arrays, label = singlePatch_multiMod(ID[patch], arr_path, patch_dim)
            
        if img_arrays is not None:
            np.save(filepath, img_arrays)
            ## save file info into pd.DataFrame
            pd_frame = pd_frame.append({'File_Path': filepath, 'ProxID': ProxID,'Patch_name': patch, \
                                    'File name':ProxID+'_'+patch+'.npy', 'label': label, 'Image file shape': img_arrays.shape}, ignore_index = True) # original path

    return pd_frame

#%%
if __name__ == '__main__':
    # Load DICOM image patch coordinate dictionary
    patch_coord_dict = pickle.load(open(os.path.join(os.getcwd(), 'patch_coord.pkl'), 'rb'))
    resize_img_dir = os.path.join(os.getcwd(), 'resized')
    #patch_dim = [(7, 96, 96), (5, 64, 64)] # D typically is an odd number unless dealing with inconsistent slice thickness later on
    #img_patches = pd.DataFrame([])

    for patch_dim in patch_coord_dict.keys():  # number of patch_size depends on the number of patch sizes generated
        img_patches = pd.DataFrame([])
        for ProxID in patch_coord_dict[patch_dim].keys():
            resize_img_path = os.path.join(resize_img_dir, ProxID)
            patch_dest_path = os.path.join(os.getcwd(), 'img_patches', str(patch_dim[1])+'_'+str(patch_dim[2])+'_'+str(patch_dim[0]))
            if not os.path.exists(patch_dest_path):
                os.mkdir(patch_dest_path)
            img_patches = extract_imgPatchCoord(patch_coord_dict[patch_dim][ProxID], resize_img_path, ProxID, patch_dest_path, img_patches, patch_dim)
        #img_patches.to_csv(os.path.join(os.path.dirname(patch_dest_path),'img_patches.csv')
        img_patches.to_pickle(os.path.join(os.path.dirname(patch_dest_path),str(patch_dim[1])+'_'+str(patch_dim[2])+'_'+str(patch_dim[0])+'_img_patches.pkl'))
    ########### ktrans image patch saving
    patch_coord_ktrans_dict = pickle.load(open(os.path.join(os.getcwd(), 'patch_coord_ktrans.pkl'), 'rb'))
    resize_ktrans_img_dir = os.path.join(os.getcwd(), 'resized_ktrans')

    for patch_dim in patch_coord_ktrans_dict.keys():  # number of patch_size depends on the number of patch sizes generated
        img_ktrans_patches = pd.DataFrame([])
        for ProxID in patch_coord_ktrans_dict[patch_dim].keys():
            resize_ktrans_img_path = os.path.join(resize_ktrans_img_dir, ProxID)
            patch_ktrans_dest_path = os.path.join(os.getcwd(), 'img_ktrans_patches', str(patch_dim[1])+'_'+str(patch_dim[2])+'_'+str(patch_dim[0]))
            if not os.path.exists(patch_ktrans_dest_path):
                os.mkdir(patch_ktrans_dest_path)

            img_ktrans_patches = extract_imgPatchCoord(patch_coord_ktrans_dict[patch_dim][ProxID], resize_ktrans_img_path, ProxID, patch_ktrans_dest_path, img_ktrans_patches, patch_dim, ktrans = True)
        #img
        img_ktrans_patches.to_pickle(os.path.join(os.path.dirname(patch_ktrans_dest_path),str(patch_dim[1])+'_'+str(patch_dim[2])+'_'+str(patch_dim[0])+'_img_ktrans_patches.pkl'))
