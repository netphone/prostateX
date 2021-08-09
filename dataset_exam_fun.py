# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import glob
import pydicom
import numpy as np
import scipy.ndimage.interpolation
import SimpleITK as sitk
import pandas as pd
import pickle
import argparse
#import matplotlib.pyplot as plt


def load_dcmScan(path, size_HWD):
    H, W, D = size_HWD
    slices = np.zeros((D, H, W))
    for s in os.listdir(path):
        ds = pydicom.dcmread(path + '/' + s)
        InstNum = ds.InstanceNumber
        try:
            slices[InstNum-1, :, :] = ds.pixel_array.T
        except:
            print("An exception occurrs at ", path+'/'+s)
            print("Proposed number of slices: ", D, "Received Instance Number: ", InstNum)

    return slices


def load_ktrans(path):
    ## need instanceNumber as well?
    for s in os.listdir(path):
        if 'mhd' in s:
            imgInfo = sitk.ReadImage(path+'/'+s)
            #ijk =imgInfo.TransformPhysicalPointToIndex(posKTRANS)
            imgArr = sitk.GetArrayFromImage(imgInfo)
            resol = np.array(list(reversed(imgInfo.GetSpacing())))
            imgArr = np.swapaxes(imgArr, 1, 2)
    return imgArr[::-1], resol




class InvestigateData(object):
    def __init__(self, dicom_dir, ktrans_dir):
        self.dir = dicom_dir
        self.ktrans_dir = ktrans_dir
        self.file_paths = None
        self.ktrans_file_paths = None
        self.dcm_sel = None
        self.ktrans_sel = None
        self.num_sizes= {}
        self.num_thkness = {}
        self.sp_res = {}
        self.ktrans_res = {}
        self.assemble_bundle = None
        self.assemble_bundle_ktrans = None
        self.resize_finding_loc = {}
        self.resize_finding_loc_ktrans = {}

        self.df = None
        self.df_ktrans = None

    def get_dicom_file_paths(self,):
        dicom_fps = glob.glob(self.dir+'/'+'*.dcm')
        self.file_paths = list(set(dicom_fps))


    def get_KTrans_file_paths(self,):
        ktrans_fps = glob.glob(self.ktrans_dir+'/'+'*.mhd')
        self.ktrans_file_paths = list(set(ktrans_fps))

    ####################################################

    def selected_dicom_files(self, demanded_img_mods, folder, train_mode = True, dicom_files_sel = dict()):
        for rt, directs, files in os.walk(self.dir, topdown = True):
            if train_mode and rt.endswith(folder):
                directs[:] = [d for d in directs if int(d.split('-')[1]) < 204]
            elif not train_mode and rt.endswith(folder):
                directs[:] = [d for d in directs if int(d.split('-')[1]) >= 204]

            for img_mod in demanded_img_mods:
                if img_mod in rt:
                    fp_split = rt.split('/')
                    scan_path = '/'.join(fp_split[:-1])

                    if not scan_path in dicom_files_sel.keys():
                        dicom_files_sel[scan_path] = {}

                    if img_mod in dicom_files_sel[scan_path].keys():
                        if int(dicom_files_sel[scan_path][img_mod].split('-')[0]) < int(fp_split[-1].split('-')[0]):
                            dicom_files_sel[scan_path][img_mod] = fp_split[-1]
                    else:
                        dicom_files_sel[scan_path][img_mod]= fp_split[-1]

        self.dcm_sel = dicom_files_sel


    def selected_ktrans_files(self, ktrans_files_sel = []):
        for rt, direct, files in os.walk(self.ktrans_dir):
            for f in files:
                if '.mhd' in f:
                    ktrans_files_sel.append(rt)
                    continue

        self.ktrans_sel = ktrans_files_sel


    def read_dicom_folder(self):
        #num_sizes = {}
        #num_thick = {}
        #sp_res = []

        for key in self.dcm_sel.keys():
            f_values = self.dcm_sel[key].values()

            for value in f_values:

                ds = pydicom.dcmread('/'.join([key, value, '000000.dcm']))
                num_files = len(os.listdir('/'.join([key, value])))
                ########################################################################
                '''
                size = int(ds.SliceThickness)
                #size = [ds.Columns, ds.Rows, int(ds.SliceThickness)]#[0x0018, 0x0050])
                if temp_size == []:
                    temp_size = size
                else:
                    if temp_size != size:
                        print(key, value, temp_size, size)
                '''
                ########################################################################
                size, resol, thick = (ds.Columns, ds.Rows, num_files), tuple([float(x) for x in ds.PixelSpacing]), float(ds.SliceThickness)
                self.sp_res.append([key, value, size, resol, thick])

                if size in self.num_sizes.keys():
                    self.num_sizes[size] += 1
                else:
                    self.num_sizes[size] = 1

                if thick in self.num_thkness.keys():
                    self.num_thkness[thick] += 1
                else:
                    self.num_thkness[thick] = 1


    def read_ktrans_files(self):
        ktrans_info = []
        for file in self.ktrans_sel:
            ImgArr, resol = load_ktrans(file)
            ktrans_info.append([file, ImgArr.shape, resol])
        return ktrans_info


    def examine_patient_scan(self, path):
        # There is an inconsistency in ProstateX--0038 that the number of slices is 18 in the folder of
        # 7-ep2ddifftraDYNDISTADC-06300 while the max value of Instance Number is 19. Compared to the folder
        # of BVal, the folder of *ADC-06300 misses 1 slice.
        ds = pydicom.dcmread('/'.join([path, '000000.dcm']))
        num_dcm_files = len(os.listdir(path))
        size_HWD, resol_HWD = (ds.Columns, ds.Rows, num_dcm_files), \
        tuple([float(x) for x in ds.PixelSpacing]) + tuple([float(ds.SliceThickness)])

        return size_HWD, resol_HWD


    def resample_array(self, patient_dcm):
        lstDCMfiles = []
        for dcm_path, size_HWD, resize_shape in patient_dcm:
            dcm_slices = load_dcmScan(dcm_path, size_HWD)
            resampled_dcm = scipy.ndimage.interpolation.zoom(dcm_slices, np.insert(np.array(resize_shape)/dcm_slices.shape[1:], 0, 1.), order=1)
            #lstDCMfiles.append([dcm_slices.shape, resampled_dcm.shape])
            lstDCMfiles.append([dcm_path, resampled_dcm])
        return lstDCMfiles


    def slice_array(self, img_arrays, tgt_size = (320, 320)):
        target_arrays = []
        for img_arr in img_arrays:
            __, W, H = img_arr[1].shape
            crop_W, crop_H = int((W - tgt_size[0])/ 2), int((H - tgt_size[1])/ 2)
            target_img = img_arr[1][:, crop_W:crop_W+tgt_size[0], crop_H:crop_H+tgt_size[1]]
            target_arrays.append([img_arr[0], target_img])

        return target_arrays


    def resize_dcm(self, resize_dict, return_imgs = True):
        assemble_img_bundles = []
        df = pd.DataFrame([])
        img_df = pd.DataFrame([])

        if not os.path.exists(os.path.join(os.path.dirname(self.dir), 'resized')):
            os.mkdir(os.path.join(os.path.dirname(self.dir), 'resized'))

        for key in self.dcm_sel.keys():
            dcm_dim = dict()
            ind_patient_dcm_scans = []

            for value in self.dcm_sel[key].values():
                dcm_path = '/'.join([key, value])
                size_HWD, resol_HWD = self.examine_patient_scan(dcm_path)
                # compare values in dcm_dim if some keys existing already
                if bool(dcm_dim) and (resol_HWD[-1] != tuple(dcm_dim.values())[-1][-1][-1]):
                    print(key)     # <-- will be replaced by another function
                    print('slice thickness does not match!')
                    dcm_dim[value] = ['slice thickness no match', size_HWD, resol_HWD]
                else:
                    # resize the dcm image using the output ds
                    plane_area = tuple(int(x) for x in np.array(size_HWD[:-1])*np.array(resol_HWD[:-1]))
                    dcm_dim[value] = [plane_area, size_HWD, resol_HWD]
                    # required output size:
                    resize_shape = resize_dict[plane_area][size_HWD[:-1]]
                    ind_patient_dcm_scans.append((dcm_path, size_HWD, resize_shape))
                    df =df.append({'File_Path': dcm_path, 'img_folder': value, 'plane_area(ori)': plane_area,
                                   'Img_dim': size_HWD, 'resize_dim': resize_shape}, ignore_index = True)
            self.sp_res[key] = dcm_dim

            if len(ind_patient_dcm_scans) == 3:  #(or == 4)
                img_bundle = self.resample_array(ind_patient_dcm_scans)
                # add a step of resize the image bundles (crop the slices and remove some slices)
                # image cropping for each slice
                img_bundle = self.slice_array(img_bundle, (320,320))
                # save img_bundle to disk
                for img_mod in img_bundle:
                    saved_path = os.path.join(os.path.dirname(self.dir), 'resized', img_mod[0].split('/')[-3])

                    if not os.path.exists(saved_path):
                        os.mkdir(saved_path)

                    np.save(os.path.join(saved_path, '-'.join(img_mod[0].split('/')[-1].split('-')[:2])), img_mod[1])
                    img_df = img_df.append({'File_Path': img_mod[0], 'ProxID': img_mod[0].split('/')[-3] ,
                                            'cropped_img_dim': img_mod[1].shape, 'saved_file': saved_path+'/'+'-'.join(img_mod[0].split('/')[-1].split('-')[:2])+'.npy'}, ignore_index = True) # original path

                if return_imgs:
                    assemble_img_bundles.append(img_bundle)

                print(img_bundle[0][1].shape, img_bundle[1][1].shape, img_bundle[2][1].shape)
        self.df = pd.merge(df, img_df, on='File_Path')

        if return_imgs:
            self.assemble_bundle = assemble_img_bundles  ## <<-- this occupies a lot of RAM space




    def resize_ktrans(self, resize_dict_ktrans, return_imgs = True):
        assemble_img_bundles = []
        kdf = pd.DataFrame([])
        k_img_df = pd.DataFrame([])

        if not os.path.exists(os.path.join(os.path.dirname(self.ktrans_dir), 'resized_ktrans')):
            os.mkdir(os.path.join(os.path.dirname(self.ktrans_dir), 'resized_ktrans'))


        for file in self.ktrans_sel:
            ImgArr, resol = load_ktrans(file) #size_HWD, resol_HWD
            plane_area = tuple([int(x*y) for x, y in zip(ImgArr.shape[1:], resol[1:])])
            self.ktrans_res[file] =[plane_area, (ImgArr.shape[1], ImgArr.shape[2], ImgArr.shape[0]), (resol[1], resol[2], resol[0])]
            resize_shape = resize_dict_ktrans[plane_area]

            # plane area: 192x192 (most), 230x230, and 249x249
            # target area: 160x160 with 320x320 pixels
            # 192x192: 128x128(img) --> resize to 384x384 --> crop to 320x320
            # 230x230: 128x128(img) --> resize to 460x460 --> crop to 320x320
            # 249x249: 128x128(img) --> resize to 498x498 --> crop to 320x320
            kdf =kdf.append({'File_Path': file, 'plane_area(ori)': plane_area,
                                   'Img_dim': ImgArr.shape, 'resize_dim': resize_shape}, ignore_index = True)

            resampled_ImgArr = scipy.ndimage.interpolation.zoom(ImgArr, np.insert(np.array(resize_shape)/ImgArr.shape[1:], 0, 1.), order=1)
            img_bundle = self.slice_array([[file , resampled_ImgArr]], (320,320))
            if return_imgs:
                assemble_img_bundles.append(img_bundle)
            print(img_bundle[0][1].shape)

            saved_path = os.path.join(os.path.dirname(self.ktrans_dir), 'resized_ktrans', file.split('/')[-1])

            if not os.path.exists(saved_path):
                os.mkdir(saved_path)

            np.save(os.path.join(saved_path, file.split('/')[-1]), img_bundle[0][1])
            k_img_df = k_img_df.append({'File_Path': file, 'ProxID': file.split('/')[-1],
                                        'cropped_img_dim': img_bundle[0][1].shape, 'saved_file': saved_path+'/'+file.split('/')[-1]+'.npy'}, ignore_index = True) # original path

        self.df_ktrans = pd.merge(kdf, k_img_df, on='File_Path')

        if return_imgs:
            self.assemble_bundle_ktrans = assemble_img_bundles  ## <<-- this occupies a lot of RAM space


    def finding_loc(self, label_csv_file, resize_dict):

        label_csv = pd.read_csv(label_csv_file)
        extract_label = label_csv[['ProxID', 'fid', 'pos', 'ijk', 'Dim', 'DCMSerDescr', 'DCMSerNum']]
        #resize_finding_loc = dict()
        spRes = self.sp_res
        '''
        dcm_assem_img_bundles = self.assemble_bundle

        for img_bundle in dcm_assem_img_bundles:
            # for training set
            ProxID = img_bundle[0][0].split('/')[-3]  ## instead of using 7
            self.resize_finding_loc[ProxID] = dict()
            for img_mod in img_bundle:
                img_path_info = img_mod[0].split('/')
        '''
        for file in self.df['File_Path']:
            ProxID = file.split('/')[-3]
            if ProxID not in self.resize_finding_loc.keys():
                self.resize_finding_loc[ProxID] = dict()

            img_path_info = file.split('/')
            [DCMSerNum, ImgMod] = img_path_info[-1].split('-')[0:2]
            img_mod_info = extract_label[extract_label.ProxID == ProxID][extract_label.DCMSerNum == int(DCMSerNum)]
            [plane_area, size_HWD, resol_HWD] = spRes['/'.join(img_path_info[:-1])][img_path_info[-1]]

            resize = resize_dict[plane_area][size_HWD[:-1]]
            scale = resize[0]/size_HWD[0], resize[1]/size_HWD[1]
            # it possibly has several rows beceuase of multiple findings
            temp_dict  = {}

            for idx, (pos, ijk_list, fid) in enumerate(zip(img_mod_info['pos'].tolist(), img_mod_info['ijk'].tolist(), img_mod_info['fid'].tolist())):
                i, j, k = ijk_list.split(' ')
                rescale_i, rescale_j = int(int(i)*scale[0]), int(int(j)*scale[1])
                # each resize_i, resize_j, k associated with a finding
                resize_i, resize_j = rescale_i - int((resize[0]-320)/2), rescale_j - int((resize[1]-320)/2)
                temp_dict['fid'+str(idx+1)] = {}#fid
                temp_dict['fid'+str(idx+1)]['ori_ijk'] = (int(i),int(j),size_HWD[2]-int(k)-1)  ## shape[0]-k-1
                temp_dict['fid'+str(idx+1)]['resize_ijk'] = (resize_i,resize_j,size_HWD[2]-int(k)-1)  ## shape[0]-k-1
                temp_dict['fid'+str(idx+1)]['pos'] = pos
            self.resize_finding_loc[ProxID][DCMSerNum+'-'+ImgMod] = temp_dict


    def finding_loc_ktrans(self, ktrans_label_csv_file, resize_dict_ktrans):
        ktrans_csv = pd.read_csv(ktrans_label_csv_file)
        extract_label = ktrans_csv[['ProxID', 'fid', 'pos', 'ijk']]
        '''
        for img_bundle in self.assemble_bundle_ktrans:
            img_path_info = img_bundle[0][0]
            ProxID = img_path_info.split('/')[7]
        '''

        for img_path_info in self.df_ktrans['File_Path']:
            ProxID = img_path_info.split('/')[-1]
            self.resize_finding_loc_ktrans[ProxID] = dict()

            img_info = extract_label[extract_label.ProxID == ProxID]
            plane_area = self.ktrans_res[img_path_info][0]
            size_HWD = self.ktrans_res[img_path_info][1]

            resize =resize_dict_ktrans[plane_area]
            scale = resize[0]/size_HWD[0], resize[1]/size_HWD[1]
            temp_dict = {}

            for idx, (pos, ijk_list, fid) in enumerate(zip(img_info['pos'].tolist(), img_info['ijk'].tolist(), img_info['fid'].tolist())):
                i, j, k = ijk_list.split(' ')
                rescale_i, rescale_j = int(int(i)*scale[0]), int((int(j))*scale[1])
                resize_i, resize_j = rescale_i - int((resize[0]-320)/2), rescale_j - int((resize[1]-320)/2)
                temp_dict['fid'+str(idx+1)] = {}
                temp_dict['fid'+str(idx+1)]['ori_ijk'] = (int(i),int(j),size_HWD[2]-int(k)-1)
                temp_dict['fid'+str(idx+1)]['resize_ijk'] = (resize_i,resize_j, size_HWD[2]-int(k)-1)
                temp_dict['fid'+str(idx+1)]['pos'] = pos
            self.resize_finding_loc_ktrans[ProxID] = temp_dict


    ############################################

#%%
def namestr(obj, namespace):
        return [name for name in namespace if namespace[name] is obj]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', help='ProstateX image path', default='.')
    parser.add_argument('--dicom-dir', help='dicom image folder path', default='DICOM')
    parser.add_argument('--ktrans-dir', help='ktrans image folder path', default='ProstateX-Ktrans')
    parser.add_argument('--label_csv', help='path of label csv file', default='ProstateX-TrainingLesionInformationv2/ProstateX-Images-Train.csv')
    parser.add_argument('--ktrans_label_csv', help='paht of ktrans label csv file', default='ProstateX-TrainingLesionInformationv2/ProstateX-Images-KTrans-Train.csv')
    args = parser.parse_args()

    dicom_dir = os.path.join(os.path.abspath(args.data_path), args.dicom_dir)
    ktrans_dir = os.path.join(os.path.abspath(args.data_path), args.ktrans_dir)
    prostateX = InvestigateData(dicom_dir, ktrans_dir)
    prostateX.get_dicom_file_paths()

    ## --> image_fps = prostateX.file_paths  # <---  verification only??
    #img_mod_count = dict()
    demanded_img_mods = ["t2tsetra", "ADC", "BVAL"]
    # collect all the file paths that contains a file of T2W_trans, ADC or BVAL
    prostateX.selected_dicom_files(demanded_img_mods, folder = args.dicom_dir , train_mode = True)
    ## --> dicom_collect = prostateX.dcm_sel  # <-- verification only?
    prostateX.selected_ktrans_files()
    # resize dictionary
    resize_dict = {(180,180):{(320,320):(360,360), (256,256):(360,360)}, (192,192):{(384,384):(384,384),
               (320,320):(384,384), (512,512):(384,384), (640,640):(384,384)}, (200,200):{(320,320):(400,400)},
                (260,211):{(160,130):(520,422)}, (168,256):{(84,128):(336,512)}, (172,256):{(86,128):(344,512)},
                (176,256):{(88,128):(352,512)}, (240,256):{(120,128):(480,512)}, (256,212):{(128,106):(512,424)}}

    resize_dict_ktrans = {(192, 192):(384, 384), (230, 230):(460, 460), (249, 249):(498, 498)}
    # one in ktrans test folder contains (20,160,160) x (3.6,1.625,1.625) = plan area (260,260)


    ### the line below is for test only
    #prostateX.resize_ktrans()


    prostateX.resize_dcm(resize_dict, return_imgs = False)
    sp_res = prostateX.sp_res
    df = prostateX.df
    df.to_pickle(os.path.join(os.path.dirname(dicom_dir),'resized_dicom_imgs.pkl'))

    #########
    prostateX.resize_ktrans(resize_dict_ktrans, return_imgs = False)
    ktrans_res = prostateX.ktrans_res
    df_ktrans = prostateX.df_ktrans
    df_ktrans.to_pickle(os.path.join(os.path.dirname(ktrans_dir), 'resized_ktrans_imgs.pkl'))

    label_csv_file = os.path.join(os.path.abspath(args.data_path), args.label_csv) #'/Users/rccuser/Desktop/ProstateX/data/ProstateX-TrainingLesionInformationv2/ProstateX-Images-Train.csv'
    prostateX.finding_loc(label_csv_file, resize_dict)
    resize_finding = prostateX.resize_finding_loc
    ktrans_label_csv_file = os.path.join(os.path.abspath(args.data_path), args.ktrans_label_csv) #'/Users/rccuser/Desktop/ProstateX/data/ProstateX-TrainingLesionInformationv2/ProstateX-Images-KTrans-Train.csv'
    prostateX.finding_loc_ktrans(ktrans_label_csv_file, resize_dict_ktrans)
    resize_finding_ktrans = prostateX.resize_finding_loc_ktrans



    for dic in ['sp_res', 'ktrans_res', 'resize_finding', 'resize_finding_ktrans']:
        f = open(os.path.join(os.path.dirname(dicom_dir), dic+'.pkl'), 'wb')
        pickle.dump(locals()[dic],f)


    #img_assem = prostateX.assemble_bundle
    #img_assem_ktrans = prostateX.assemble_bundle_ktrans
if __name__ == '__main__':
    main()
