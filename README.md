## PCa-finder model training pipeline
### Steps of training a model using prostateX dataset

#### 1. executing dataset_exam_fun.py to collect resized image info
- save resized images into 'resized' and 'resized_ktrans' folders
- save 'resized_dicom_imgs.pkl' and 'resized_ktrans_imgs.pkl'
- save four files named: <b><I>sp_res.pkl, ktrans_res.pkl, resize_finding.pkl, and resize_finding_ktrans.pkl</I></b>
```
python dataset_exam_fun.py --data-path='./raw data' --dicom-dir='DICOM' --ktrans-dir='ProstateX-Ktrans'
```
#### 2. executing Image_Patch_Generation_revWktrans.py 
to generate finding and randomly-picked image patches with different sizes 
- save 'patch_coord.pkl'
- save 'patch_coord_ktrans.pkl'
```
python Image_Patch_Generation_revWktrans.py --working-folder='./raw data' -p '(7, 96, 96)' '(5, 64, 64)' '(5, 48, 48)' '(3, 42, 42)' --overlap=4
```
#### 3. Execute save_img_patches.py (or save_img_patchesWktrans.py) 
to create subfolders associated with patch sizes under img_patches (and img_ktrans_patches) folders
- save associated patch-sized .npy files and corrospoinding patch-sized info in the particular patch-size subfolder.
```
python ./data/save_img_patches.py
```
#### 4. Calculate averages and stds, save standarized data and recheck the numbers
```
python ./data/std.py
python ./data/std_ktrans.py
```
#### 5. Integrate lossse.py (for focal loss function) and transforms.py (for image augmentation) into train.py
- Integrate DenseNet model and training procedure
- read arguments externally.
```
python train.py args
```
#### Model Architecture
- Aligning MRI images retrieved from T2W, ADC, DWI, and Ktrans modalities according to the clinical lesions’ coordinates, followed by image stacking.

![Model Architecture](https://github.com/netphone/prostateX/blob/main/architecture.jpg?raw=true)
