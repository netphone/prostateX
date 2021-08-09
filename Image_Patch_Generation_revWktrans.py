#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:38:42 2019

@author: rccuser
"""

### extract finding patches from resized images
### extact image patches excluding findings and partially including findings from resize images

### dataset must be resized and findings must be localized after resizing
import os
import pandas as pd
import numpy as np
import pickle
import argparse
#from visualization import visualize as vis



class ImagePatches(object):
    def __init__(self, patch_dim, clinFindings, overlap, including_ktrans = True):
        self.include_ktrans = including_ktrans
        self.patch_dim = patch_dim
        self.findings_lbl = clinFindings
        self.overlap = overlap
        self.patch_assem = None
        self.patch_assem_ktrans = None
        self.IJK_Shift = {}
        self.rand_patch = {}
        self.MOD = {}
        self.imgs_dims = {}

    def overlap_invest(self, patch_dict, Mod, delta = 8):
        FID = list(patch_dict.keys())
        ClinSig = []
        for c1 in range(len(FID)):
            if 'fid' in FID[c1]:
                for mod in Mod:
                    (i, j, k) = patch_dict[FID[c1]][mod][2]
                    for c2 in range(len(FID)):
                        if 'fid' not in FID[c2]:
                        #if c2 < len(FID):
                        #if (patch_dict[FID[c2]][mod][-2] == True and patch_dict[FID[c1]][mod][-2] == False) or \
                            #(patch_dict[FID[c2]][mod][-2] == False and patch_dict[FID[c1]][mod][-2] == True):  ## only True and False needs to examine
                            ## check overlap if patient_findings[curr_fid] in patch_dict[pre_fid][mod][0]
                            #(i, j, k) = patient_findings[FID[c1]]

                            (start_D, end_D, start_H, end_H, start_W, end_W) = patch_dict[FID[c2]][mod][0]
                            if (i-delta-start_H)*(end_H-i-delta) > 0 and (j-delta-start_W)*(end_W-j-delta) > 0 and (k-start_D)*(end_D-k) >= 0:
                                if patch_dict[FID[c1]][mod][-2] == True:
                                    #print(FID[c1], FID[c2])
                                    #print(patch_dict[FID[c2]][mod][-2])
                                    patch_dict[FID[c2]][mod][-2] = True
                                    #print(patch_dict[FID[c2]][mod][-2])

                            ClinSig.append(patch_dict[FID[c2]][mod][-2])

            # temporary block the line below due to inconsistent fid in ProstateX 193 fid4
            #assert len(set(ClinSig)) == 1 or len(set(ClinSig)) == 0, print("the ClinSig finding not consisitent in", FID[c1])
        ############' Have to consider the case of random choice
        return patch_dict


    #def dcm_patch_setup(self, img_assem, resize_finding):
    def dcm_patch_setup(self, resize_img, resize_finding):

        patch_coord = {}
        fid_label = self.findings_lbl
        '''
        for img_bundle in img_assem:
            for img in img_bundle:
                ProxID, __, img_mod = img[0].split('/')[7:]
                if ProxID not in patch_coord.keys():
                    patch_coord[ProxID] = {}

                mod = '-'.join(img_mod.split('-')[:2])
                findings = resize_finding[ProxID][mod]
                D, H, W = img[1].shape
        '''
        del_findings = []
        for dim in self.patch_dim:
            patch_coord[dim] = {}
            for __, row in resize_img.iterrows():
                ProxID, img_mod = row.ProxID, row.img_folder
                if ProxID not in patch_coord[dim].keys():
                    patch_coord[dim][ProxID] = {}

                mod = '-'.join(img_mod.split('-')[:2])
                findings = resize_finding[ProxID][mod]
                D, H, W = row.cropped_img_dim

                assert (dim[0] <= D) and (dim[1] <= H) and (dim[2] <= W), "Error: patch size is larger than the image size"
                # ProstateX 140 has inverse k findings.
                ind_patient_findings = {}
                for fid in findings:
                    (i,j,k) = findings[fid]['resize_ijk'] # resize ijk at fid
                    ## ProstateX 193 fid4 has inconsistent i,j,k in k
                    if any(np.array([i,j,k])<0):
                        print(ProxID, mod, fid, (i,j,k), ' is smaller than 0 and will not be included.')
                        del_findings.append([dim, ProxID, fid])
                    else:
                        ind_patient_findings[fid] = (i,j,k)
                        pos = findings[fid]['pos']
                        if fid not in patch_coord[dim][ProxID].keys():
                            patch_coord[dim][ProxID][fid] = {}
                        clinSig = fid_label[fid_label.ProxID == ProxID][fid_label.pos == pos].ClinSig.item()
                        # first, ensure the patch boundary is inside the image

                        start_D, end_D = k - int(dim[0]/2), k + int((dim[0]+1)/2)
                        start_H, end_H = i - int(dim[1]/2), i + int((dim[1]+1)/2)
                        start_W, end_W = j - int(dim[2]/2), j + int((dim[2]+1)/2)

                        if start_D <= 0:
                            start_D = 0; end_D = dim[0]
                        if end_D >= D:
                            start_D = D - dim[0]; end_D = D
                        if start_H <= 0:
                            start_H = 0; end_H = dim[1]
                        if end_H >= H:
                            start_H = H - dim[1]; end_H = H
                        if start_W <= 0:
                            start_W = 0; end_W = dim[2]
                        if end_W >= W:
                            start_W = W - dim[2]; end_W = W

                        if k < start_D or k >= end_D:
                            print('k: {}, start_D: {}, end_D: {}'.format(k, start_D, end_D))
                        patch_coord[dim][ProxID][fid][mod] = [[start_D, end_D, start_H, end_H, start_W, end_W], [k-start_D, end_D-k], (i, j, k), (D, H, W), clinSig, pos]
                    #print(patch_coord[dim][ProxID][fid][mod])
                    ## only 1 example of H, D differnce in findings of different image mode being greater than 4.
                    ## only 8 example of H, D differnce in findings of different image mode being greater than 3.
                    ## overlap investigation to ensure ClinSig
                    ##########################################

                ##### ClinSig check may need to move to dcm_patchCatch ####
        for item in del_findings:
            if item[2] in patch_coord[item[0]][item[1]].keys():
                del patch_coord[item[0]][item[1]][item[2]]

        self.patch_assem = patch_coord


    def dcm_patchCatch(self, random_patch = True):
        # to ensure the slices cover the same volume
        patch = self.patch_assem
        # dim --> ProxID --> fid --> mod
        for i, dim in enumerate(self.patch_dim):
            for ProxID in patch[dim].keys():
                for fid in patch[dim][ProxID].keys():

                    shift_slice = False
                    Diff = {}
                    Mod = []
                    for idx, mod in enumerate(patch[dim][ProxID][fid].keys()):
                        Mod.append(mod)
                        ## difference from finding slice
                        diff = (patch[dim][ProxID][fid][mod][1][1]-1)-patch[dim][ProxID][fid][mod][1][0]
                        if diff != 0:
                            Diff[diff] = [patch[dim][ProxID][fid][mod][1][0], patch[dim][ProxID][fid][mod][1][1]-1]
                            shift_slice = True
                        ################ Consider a solution later ############
                        # negative k in ProstateX-0154
                        if ProxID == 'ProstateX-0025':
                            continue
                        #######################################################
                        if shift_slice and idx == 2:
                            shift = Diff[max(Diff.keys())]
                            # check all 3 mods to update the slice location
                            for m in Mod:
                                patch[dim][ProxID][fid][m] = cood_adjust(patch[dim][ProxID][fid][m], shift)

                print(dim, ProxID)

                if random_patch:
                    #img_D, img_H, img_W = patch[dim][ProxID][fid][mod][3]
                    ## a proper number of random selection is critical
                    #ijk_shift, ran_patches= self.dcm_randomCatch(patch[dim][ProxID], ProxID, dim, (img_D, img_H, img_W), 15, i)
                    ijk_shift, ran_patches= self.dcm_randomCatch(patch[dim][ProxID], ProxID, dim, 100, i)

                    if i == 0:
                        self.IJK_Shift[ProxID] = ijk_shift
                        self.rand_patch[ProxID] = ran_patches
                        self.MOD[ProxID] = Mod
                    for ran_id in ran_patches.keys():  # contacting random patch into patch
                        patch[dim][ProxID][ran_id] = ran_patches[ran_id]

                ClinSig_status = self.overlap_invest(patch[dim][ProxID], Mod, delta = self.overlap) # maynot necessary: fid, clinSig)
                for finding in ClinSig_status.keys():
                    for mod in ClinSig_status[finding].keys():
                        if patch[dim][ProxID][finding][mod][-2] != ClinSig_status[finding][mod][-2]:
                            print(dim, ProxID, finding, mod, " ==> ClinSig being True")
                        patch[dim][ProxID][finding][mod][-2] == ClinSig_status[finding][mod][-2]

                #######################
                # remove it later
                #vis(patch[ProxID][dim], ProxID, dim, img_assem, self.patch_assem[ProxID][dim])
                #######################
        self.patch_assem = patch



    def dcm_randomCatch(self, patch, ProxID, patch_dim, num_in_random_patch, index):
        ## from each pactch fid --> mod
        # patch[fid][mod]: [[start_D, end_D, start_H, end_H, start_W, end_W], [k-start_D, end_D-k], (i, j, k), (D, H, W), clinSig, pos]
        IJK_AvgShift = None
        random_patches = {}
        #(D, H, W) = dim  ### it should be dependent on 'mod' <-- will be dismissed
        (d, h, w) = patch_dim
        IJK_shifts = []
        MOD = []

        if index == 0:
            for fid in patch.keys():
                IJK_tmp = []
                for mod in patch[fid].keys():
                    i, j, k = patch[fid][mod][2]
                    IJK_tmp.append((i,j,k))
                    MOD.append(mod)
                    #print(mod, (i,j,k))
                tmp = list(dict.fromkeys(IJK_tmp))
                if len(tmp) == 1:
                    IJK_diff = np.array([[0,0,0]])
                else:
                    IJK_diff = np.diff(tmp, axis = 0)

                #if IJK_diff != []:
                IJK_shifts.append(IJK_diff)

            MOD = list(dict.fromkeys(MOD))
            IJK_AvgShift = np.mean(IJK_shifts, axis = 0, dtype=int)  ## <--- try to solve the issue
            #############################
            #if index == 0:
            #    print(IJK_AvgShift)
            #############################
            count = 1

            img_dims = {}
            while (count <= num_in_random_patch):
                #for count in range(num_in_random_patch):
                # random numbers with the constraint
                for mod in MOD:
                    img_dims[mod] =patch[fid][mod][3]

                min_DHW = np.min([img_dims[key] for key in img_dims.keys()], axis = 0)
                str_Coods = [np.random.randint(0, min_DHW[0]-d+1), np.random.randint(0, min_DHW[1]-h+1), np.random.randint(0, min_DHW[2]-w+1)]

                #assert (start_d >=0 and start_h >=0 and start_w >=0), 'random number cannot be smaller than 0'
                assert np.min(str_Coods) >= 0, 'random number cannot be smaller than 0'
                #start_d, start_h, start_w = randomStarts(D,H,W,d,h,w)
                random_patches['radm'+str(count)] = {}
                # check every fid in patch
                # shift in each mod is different, particularly for differnet fid; how to match them --> average out the shifts (done above)
                random_patches['radm'+str(count)] = rand_patches(random_patches['radm'+str(count)], MOD, str_Coods, patch_dim, img_dims, IJK_AvgShift[0])

                if random_patches['radm'+str(count)] == True:
                    continue

                ## Need to complish:
                ## 1.  check if the current random patch overlaps with the previous patches less than 90%
                ### setting the overlap of 70% and trying to accumulate 20 random selected patches would be impractical
                ### here are several trials: 70% & 5 randoms, 90% & 20,
                if count >= 2:
                    ref_rands, rand = list(random_patches.keys())[:-1], list(random_patches.keys())[-1]
                    for ref_rand in ref_rands:
                        # only use the last mod for comparison
                        ref_cood = np.array(random_patches[ref_rand][mod][0])[2:]
                        tgt_cood = np.array(random_patches[rand][mod][0])[2:,]
                        u_startH = max(ref_cood[0], tgt_cood[0])
                        u_endH = min(ref_cood[1], tgt_cood[1])
                        if u_endH <= u_startH:
                            continue
                        u_startW = max(ref_cood[2], tgt_cood[2])
                        u_endW = min(ref_cood[3], tgt_cood[3])
                        if u_endW <= u_startW:
                            continue
                        u_area = (u_endH - u_startH)*(u_endW - u_startW)
                        if (u_area >= (ref_cood[1] - ref_cood[0]) * (ref_cood[3] - ref_cood[2])* 0.81):
                            count -= 1
                            break
                        ## 2.  update the ClinSig Status --> done in dcm_patchCatch
                count += 1
            self.imgs_dims[ProxID] = img_dims

        else:
            #### load random patches from index = 1 because of containing the biggest shifts across different modalities
            ref_patch = self.rand_patch[ProxID]
            ref_ijk_shift = self.IJK_Shift[ProxID][0]
            ref_mod = self.MOD[ProxID]

            for key in ref_patch.keys():  # fid and radm patches
                if 'radm' in key:
                    ref_cood = ref_patch[key][ref_mod[0]][0]
                    random_patches[key]={}
                    random_patches[key]=rand_patches(random_patches[key], ref_mod, \
                                  [int((ref_cood[0]+ref_cood[1]-patch_dim[0])/2),int((ref_cood[2]+ref_cood[3]-patch_dim[1])/2),int((ref_cood[4]+ref_cood[5]-patch_dim[2])/2)], \
                                  patch_dim, self.imgs_dims[ProxID], ref_ijk_shift)
                    #random_patches[key]=rand_patches(random_patches[key], ref_mod, \
                                  #str_Coods, patch_dim, self.imgs_dims[ProxID], ref_ijk_shift)

        return IJK_AvgShift, random_patches


    #def ktrans_patch_setup(self, img_assem_ktrans, resize_finding_ktrans):
    def ktrans_patch_setup(self, resize_img_ktrans, resize_finding_ktrans):
        patch_coord_ktrans = {}
        fid_label = self.findings_lbl
        '''
        for img in img_assem_ktrans:
            #mod = 'ktrans'
            ProxID = img[0][0].split('/')[7]
            if ProxID not in patch_coord_ktrans.keys():
                patch_coord_ktrans[ProxID] = {}

                findings = resize_finding_ktrans[ProxID]
                D, H, W = img[0][1].shape
        '''
        for dim in self.patch_dim:
            patch_coord_ktrans[dim] = {}
            for ind, row in resize_img_ktrans.iterrows():
                ProxID = row.ProxID
                if ProxID not in patch_coord_ktrans.keys():
                    patch_coord_ktrans[dim][ProxID] = {}

                    findings = resize_finding_ktrans[ProxID]
                    D, H, W = row.cropped_img_dim

                    assert (dim[0] <= D) and (dim[1] <= H) and (dim[2] <= W), "Error: patch size is larger than the image size"

                    # ProstateX 140 has inverse k findings still?
                    for fid in findings:
                        (i, j, k) = findings[fid]['resize_ijk']  #resize ijk at fid
                        if any(np.array([i,j,k]) < 0):
                            print(ProxID, fid, (i,j,k), ' is smaller than 0 and will not be included.')
                        else:

                            pos = findings[fid]['pos']
                            if fid not in patch_coord_ktrans[dim][ProxID].keys():
                                patch_coord_ktrans[dim][ProxID][fid] = {}
                            clinSig = fid_label[fid_label.ProxID == ProxID][fid_label.pos == pos].ClinSig.item()
                            # first, ensure the patch boundary is inside the image

                            start_D, end_D = k - int(dim[0]/2), k + int((dim[0]+1)/2)
                            start_H, end_H = i - int(dim[1]/2), i + int((dim[1]+1)/2)
                            start_W, end_W = j - int(dim[2]/2), j + int((dim[2]+1)/2)

                            if start_D <= 0:
                                start_D = 0; end_D = dim[0]
                            if end_D >= D:
                                start_D = D - dim[0]; end_D = D
                            if start_H <= 0:
                                start_H = 0; end_H = dim[1]
                            if end_H >= H:
                                start_H = H - dim[1]; end_H = H
                            if start_W <= 0:
                                start_W = 0; end_W = dim[2]
                            if end_W >= W:
                                start_W = W - dim[2]; end_W = W

                            if k < start_D or k >= end_D:
                                print('k: {}, start_D: {}, end_D: {}'.format(k, start_D, end_D))

                            patch_coord_ktrans[dim][ProxID][fid] = [[start_D, end_D, start_H, end_H, start_W, end_W], [k-start_D, end_D-k], (i, j, k), (D, H, W), clinSig, pos]
                        ## only 1 example of H, D differnce in findings of different image mode being greater than 4.
                        ## only 8 example of H, D differnce in findings of different image mode being greater than 3.

        self.patch_assem_ktrans = patch_coord_ktrans


    def ktrans_patchCatch(self, random_patch = True):
        dcm_patch = self.patch_assem
        patch = self.patch_assem_ktrans
        for i, dim in enumerate(patch.keys()):
            for ProxID in patch[dim].keys():
                ktrans_IJK = []
                dicom_IJK = []
                for fid in patch[dim][ProxID].keys():
                    # find current ktrans fid's position
                    ktrans_pos = patch[dim][ProxID][fid][-1]

                    #Diff = {}
                    #diff = (patch[dim][ProxID][fid][1][1]-1)-patch[dim][ProxID][fid][1][0]
                    if ProxID in dcm_patch[dim].keys():
                        for FID in dcm_patch[dim][ProxID].keys():
                            # find position
                            if 'fid' in FID and all([dcm_patch[dim][ProxID][FID][mod][-1] == ktrans_pos for mod in dcm_patch[dim][ProxID][FID].keys()]):
                                tgt_shift = list(dcm_patch[dim][ProxID][FID].values())[0][1]
                                ktrans_IJK.append(patch[dim][ProxID][fid][2])
                                mod_list = list(dcm_patch[dim][ProxID][FID].keys())
                                dicom_IJK.append(dcm_patch[dim][ProxID][FID][mod_list[0]][2])

                                if patch[dim][ProxID][fid][1] != tgt_shift:
                                    print('#######', ProxID, fid, ' goes to cood_adjust -->', 'dicom shift:', tgt_shift)
                                    patch[dim][ProxID][fid] = cood_adjust(patch[dim][ProxID][fid], [tgt_shift[0], tgt_shift[1]-1])
                                    print(ProxID, ' adjusted ktrans diff shift:', patch[dim][ProxID][fid][:2])
                                break

                            #Diff[abs((dcm_patch[dim][ProxID][fid][mod][1][1]-1)-dcm_patch[dim][ProxID][fid][mod][1][0])] = [dcm_patch[dim][ProxID][fid][mod][1][0], dcm_patch[dim][ProxID][fid][mod][1][1]-1]

                ktrans_shift_IJK =  np.mean(np.array(ktrans_IJK) - np.array(dicom_IJK), axis=0, dtype=int)
                print(dim, ProxID)
                if random_patch and ProxID in dcm_patch[dim].keys():
                    ### need adjust the avg ijk shift from dicom patch
                    #for fid in patch[dim][ProxID].keys():
                    #    print('ktran ijk:', patch[dim][ProxID][fid][2]) #(i, j, k)
                    #    print('dicom avg ijk:', self.IJK_Shift[ProxID])
                    for rand in dcm_patch[dim][ProxID].keys():
                        if 'radm' in rand:
                            mod_list = list(dcm_patch[dim][ProxID][rand].keys())
                            patch[dim][ProxID][rand] = dcm_patch[dim][ProxID][rand][mod_list[0]]
                            for i in range(6):
                                patch[dim][ProxID][rand][0][i] += np.roll(ktrans_shift_IJK, 1)[int(i/2)]

                            if (any(np.array(patch[dim][ProxID][rand][0]) < 0) or patch[dim][ProxID][rand][0][1] >= patch[dim][ProxID]['fid1'][3][0]\
                                or patch[dim][ProxID][rand][0][3] >= patch[dim][ProxID]['fid1'][3][1]\
                                or patch[dim][ProxID][rand][0][5] >= patch[dim][ProxID]['fid1'][3][2]): #D
                                del patch[dim][ProxID][rand]

        self.patch_assem_ktrans = patch

##############################################################
## random selection of patches (non repeat selection)
## how to lable thsee random selections.
##############################################################

def cood_adjust(co_list, shift):  #finding coordinate mismatching acrossing different image modalities exceed image boundaries
    # not considering checking image height and width coordinate mismatching due to the finding being located far from the edges of image plane
    # co_list: [[start_D, end_D, start_H, end_H, start_W, end_W], [k-start_D, end_D-k], (i, j, k), (D, H, W), clinSig, pos]
    if co_list[1][0] != shift[0]:
        if co_list[2][2] - shift[0] < 0:
            print("start_D index is smaller than 0") # k - shift[0] >= 0
            co_list[0][0] = 0 # co_list[2][2] - shift[0]
            co_list[0][1] = shift[0] + shift[1] + 1 #co_list[2][2] + shift[1]+1
            co_list[1] = [co_list[2][2] - co_list[0][0], co_list[0][1] - co_list[2][2]]

        elif co_list[2][2] + shift[1] >= co_list[3][0]:   #D
            print("end_D index is larger than 'D'")
            co_list[0][1] = co_list[3][0]
            co_list[0][0] = co_list[3][0] - (shift[0] + shift[1]+1)
            co_list[1] = [co_list[2][2] - co_list[0][0], co_list[0][1] - co_list[2][2]]
        # ProstateX 121 has the error below
        #assert co_list[2][2] + shift[1] <= co_list[3][0]-1, print(dim, ProxID, "end_D index exceeds slice number") # k + shift[1] <= D
        else:
            co_list[0][0] = co_list[2][2] - shift[0]
            co_list[0][1] = co_list[2][2] + shift[1]+1
            co_list[1] = [shift[0], shift[1]+1]

    return co_list


def rand_patches(patch_dic, MOD, starts, dim, img_dims, AvShift, mod_tmp = None):
    for mod in MOD:
        if not bool(patch_dic):
            patch_dic[mod] = [[starts[0], starts[0]+dim[0], starts[1], starts[1]+dim[1], starts[2], starts[2]+dim[2]], False, '0']
            mod_tmp = mod
        else:
            if max(AvShift) == min(AvShift):
                patch_dic[mod] = [[starts[0], starts[0]+dim[0], starts[1], starts[1]+dim[1], starts[2], starts[2]+dim[2]], False, '0']
            else:
                if 't2' in mod_tmp:
                    patch_dic[mod] = [[starts[0]+AvShift[2], starts[0]+dim[0]+AvShift[2], starts[1]+AvShift[0], starts[1]+dim[1]+AvShift[0], \
                                      starts[2]+AvShift[1], starts[2]+dim[2]+AvShift[1]], False, '0']
                else:
                    if 't2' in mod:
                        patch_dic[mod] = [[starts[0]+AvShift[2], starts[0]+dim[0]+AvShift[2], starts[1]+AvShift[0], starts[1]+dim[1]+AvShift[0], \
                                      starts[2]+AvShift[1], starts[2]+dim[2]+AvShift[1]], False, '0']
                    else:
                        patch_dic[mod] = [[starts[0], starts[0]+dim[0], starts[1], starts[1]+dim[1], starts[2], starts[2]+dim[2]], False, '0']


                #print('radm'+str(count), mod, random_patches['radm'+str(count)][mod][0])
        if min(patch_dic[mod][0]) < 0 or patch_dic[mod][0][1] > img_dims[mod][0] or patch_dic[mod][0][3] > img_dims[mod][1] or patch_dic[mod][0][5] > img_dims[mod][2]:
            #print('coord number cannot be smaller than 0')
            #del random_patches['radm'+str(count)]
            dismiss = True
            return dismiss

    return patch_dic#, dismiss


#%%
def main():
    # a list of image patch dimensions (D, H, W)
    ## patch dimension (, 196, 196): 27 exceedings (9 patients)
    def patch_list(p):
        try:
            x,y,z = map(int, p.lstrip('(').rstrip(')').split(','))
            return (x,y,z)
        except:
            raise argparse.ArgumentTypeError('Patch input must be (x,y,z)')

    parser = argparse.ArgumentParser()
    parser.add_argument('--working-folder', help='working directory; should contain ', default='./data')
    parser.add_argument('-p', '--patch-dims', type=patch_list, nargs='+') #command line: -p '(x,y,z)' '(a,b,c)' '(i,j,k)' ...
    parser.add_argument('--overlap', help='number of pixels overlapping to define ClinSig', type=int, default=6)

    args = parser.parse_args()

    curdir = args.working_folder
    patch_dim = args.patch_dims #[(7, 96, 96), (5, 64, 64)] # D typically is an odd number unless dealing with inconsistent slice thickness later on
    findings_csv_file = os.path.join(curdir, r'ProstateX-TrainingLesionInformationv2/ProstateX-Findings-Train.csv')
    findings_csv = pd.read_csv(findings_csv_file)
    extract_findings = findings_csv[['ProxID', 'fid', 'pos', 'ClinSig']]

    Img_patch = ImagePatches(patch_dim, extract_findings, overlap=args.overlap)

    resize_finding = pickle.load(open(os.path.join(curdir, 'resize_finding.pkl'), 'rb'))
    # pandas DataFrame
    resize_img = pickle.load(open(os.path.join(curdir, 'resized_dicom_imgs.pkl'), 'rb'))
    #Img_patch.dcm_patch_setup(img_assem, resize_finding) # <-- from dataset_exam.py
    Img_patch.dcm_patch_setup(resize_img, resize_finding)
    Img_patch.dcm_patchCatch(random_patch = False)
    patch_coord_dict = Img_patch.patch_assem
    ###### remove later after examining###########
    #for dim in patch_coord_dict.keys():
    #    for ProxID in patch_coord_dict[dim].keys():
    #        for fid in patch_coord_dict[dim][ProxID].keys():
    #            for mod in patch_coord_dict[dim][ProxID][fid].keys():
    #                 if any(np.array(patch_coord_dict[dim][ProxID][fid][mod][0])<0):
    #                     print(dim, ProxID, fid, 'coords less than 0')
    #
    ##############################################

    with open(os.path.join(curdir, 'patch_coord.pkl'), 'wb') as f:
        pickle.dump(patch_coord_dict,f)
    #####################
    resize_finding_ktrans = pickle.load(open(os.path.join(curdir, 'resize_finding_ktrans.pkl'), 'rb'))
    # pandas DataFrame
    resize_img_ktrans = pickle.load(open(os.path.join(curdir, 'resized_ktrans_imgs.pkl'), 'rb'))
    #Img_patch.ktrans_patch_setup(img_assem_ktrans, resize_finding_ktrans) # <-- from dataset_exam.py
    ##############################################
    ## need to revise the associated codes beloww
    Img_patch.ktrans_patch_setup(resize_img_ktrans, resize_finding_ktrans)
    Img_patch.ktrans_patchCatch(random_patch = False)

    patch_coord_ktrans_dict = Img_patch.patch_assem_ktrans

    ############ remove after examining ###############################
    #for dim in patch_coord_ktrans_dict.keys():
    #    for ProxID in patch_coord_ktrans_dict[dim].keys():
    #        for fid in patch_coord_ktrans_dict[dim][ProxID].keys():
    #            if any(np.array(patch_coord_ktrans_dict[dim][ProxID][fid][0])<0):
    #                print(dim, ProxID, fid, 'coords less than 0')
    #############################################################################

    with open(curdir+'/'+'patch_coord_ktrans.pkl', 'wb') as f:
        pickle.dump(patch_coord_ktrans_dict,f)


if __name__ == '__main__':
    main()
