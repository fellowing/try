## ADC 一般是 (160,160,20) 

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset

# import nibabel as nib
from scipy import ndimage

import SimpleITK as sitk

# neuroimage = sitk.ReadImage("F:/DSC_LCBM_nii/1047-NIE_LAN/17.02.20/ADC/ADC.nii.gz")
# print(neuroimage.GetSize()) # (160,160,20)

# neuroimage = sitk.ReadImage("F:/DSC_LCBM_nii/1047-NIE_LAN/17.02.20/T1_MPRAGE/T1_MPRAGE.nii.gz")
# print(neuroimage.GetSize()) #(256, 256, 192)
# # t1_mprange 需要调整 pool
# neuroimage = sitk.ReadImage("F:/DSC_LCBM_nii/1047-NIE_LAN/17.02.20/T1WI/T1WI.nii.gz")
# print(neuroimage.GetSize()) # (640, 640, 20)

# neuroimage = sitk.ReadImage("F:/DSC_LCBM_nii/1047-NIE_LAN/17.02.20/T1WI+C/T1WI+C.nii.gz")
# print(neuroimage.GetSize()) # (320, 320, 20)

# neuroimage = sitk.ReadImage("F:/DSC_LCBM_nii/1047-NIE_LAN/17.02.20/T2WI/T2WI.nii.gz")
# print(neuroimage.GetSize()) # (640, 640, 20)

# neuroimage = sitk.ReadImage("F:/DSC_LCBM_nii/1047-NIE_LAN/17.02.20/T2_FLAIR/T2_FLAIR.nii.gz")
# print(neuroimage.GetSize()) # (320, 320, 20)

# neuroimage = sitk.ReadImage("F:/DSC_LCBM_nii/1047-NIE_LAN/17.02.20/CBV/CBV.nii.gz")
# print(neuroimage.GetSize()) # (220, 220, 21)

# nda_copy = sitk.GetArrayFromImage(neuroimage)
# print(nda_copy.shape)
# nda_copy[1,1,1]
# sitk.Show(neuroimage)

# def a function to get the mri path from the data list

## 我们对每种模态都建立模型，定义一个可以对


def get_path(patienti, modal = 'ADC'):
    modal_path = []
    for i in range(len(patienti[1])):
        timei = patienti[1][i]
        def if_modal(s):
            torf = (s.split('/')[-1] == modal or s.split('/')[-2] == modal)
            return torf
        modal_inx = [timei[1][x] for x in range(len(timei[1])) if if_modal(timei[1][x])]
        modal_path = modal_path + modal_inx

    return modal_path


# pats = pickle.load(open('../data/train_data.pkl', "rb"))

# labs = torch.tensor([x[-1] for x in pats], dtype = torch.float32)

def get_single(data_list, modal):
    modal_time1 = []
    modal_time2 = []
    modal_time3 = []
    for j in range(len(data_list)):

        pat_times = len(data_list[j][1])
        single_modal = get_path(data_list[j], modal)
        try:
            modal_time1 += [single_modal[0]]
        except:
            modal_time1 += ['None']

        try:
            modal_time2 += [single_modal[1]]
        except:
            modal_time2 += ['None']

        try:
            modal_time3 += [single_modal[2]]
        except:
            modal_time3 += ['None']
    modal_time1 = [x + '/' + modal + '.nii.gz' for x in modal_time1]
    modal_time2 = [x + '/' + modal + '.nii.gz' for x in modal_time2]
    modal_time3 = [x + '/' + modal + '.nii.gz' for x in modal_time3]
    three_times = [modal_time1, modal_time2, modal_time3]
    return three_times

# try1 = get_single(pats, 'ADC')

# len(try1[1])

## 将数据中的路径替换成 ../data


def zoom(image1, standard_dim):
    image_data = sitk.GetArrayFromImage(image1).T # Retrieves array data
    image_data = image_data.astype(float)
    # we need to Transpose because the ITK get (z,y,x) not (x,y,z)
    # Resize and interpolate image
    image_size = image_data.shape # Store dimensions of N-D array
    current_dim1 = image_size[0]
    current_dim2 = image_size[1]
    current_dim3 = image_size[2]
    # Calculate scale factor for each direction
    #  standard_dim = self.standard_dim
    scale_factor1 = standard_dim[0] / float(current_dim1)
    scale_factor2 = standard_dim[1] / float(current_dim2)
    scale_factor3 = standard_dim[2] / float(current_dim3)
    # Resize image (spline interpolation)
    image_data = ndimage.zoom(image_data, (scale_factor1, scale_factor2, scale_factor3))
    return image_data



class MRIData(Dataset):

    def __init__(self, data_array, standard_dim, max_num, modal):

        self.data_array = data_array      # get the data array
        self.standard_dim = standard_dim  # get the dim we want to be
        self.max_num = max_num            # get the max num
        self.modal = modal                # get the  modal 

        self.data = get_single(self.data_array, self.modal)
        self.data_list1 = self.data[0]
        self.data_list2 = self.data[1]
        self.data_list3 = self.data[2]

        data_list1 = []
        data_list2 = []
        data_list3 = []

        self.patient_label = torch.tensor([x[-1] for x in self.data_array], dtype = torch.float32)
        for i in range(len(self.data_array)):
            try:
                image1 = sitk.ReadImage(self.data[0][i]) # Loads proxy image

                # Resize image (spline interpolation)
                image_data = zoom(image1, self.standard_dim)
                ## 对影像进行差值处理，从而实现改变图片大小
                # Convert image data to a tensor
                image_data_tensor1 = torch.Tensor(image_data)
                image_data_tensor1 = image_data_tensor1.view(1,self.standard_dim[0],self.standard_dim[1],self.standard_dim[2])

            except:
                image_data_tensor1 = torch.zeros(1,self.standard_dim[0], self.standard_dim[1], self.standard_dim[2])

            try:
                image2 = sitk.ReadImage(self.data[0][i]) # Loads proxy image

                # Resize image (spline interpolation)
                image_data = zoom(image2, self.standard_dim)
                ## 对影像进行差值处理，从而实现改变图片大小
                # Convert image data to a tensor
                image_data_tensor2 = torch.Tensor(image_data)
                image_data_tensor2 = image_data_tensor2.view(1,self.standard_dim[0],self.standard_dim[1],self.standard_dim[2])

            except:
                image_data_tensor2 = torch.zeros(1,self.standard_dim[0], self.standard_dim[1], self.standard_dim[2])
            try:
                image3 = sitk.ReadImage(self.data[0][i]) # Loads proxy image

                # Resize image (spline interpolation)
                image_data = zoom(image3, self.standard_dim)
                ## 对影像进行差值处理，从而实现改变图片大小
                # Convert image data to a tensor
                image_data_tensor3 = torch.Tensor(image_data)
                image_data_tensor3 = image_data_tensor3.view(1,self.standard_dim[0],self.standard_dim[1],self.standard_dim[2])

            except:

                image_data_tensor3 = torch.zeros(1,self.standard_dim[0], self.standard_dim[1], self.standard_dim[2])
            
            data_list1.append(image_data_tensor1)
            data_list2.append(image_data_tensor2)
            data_list3.append(image_data_tensor3)

        self.images_tensor1 = torch.stack(data_list1,dim=0)
        self.images_tensor2 = torch.stack(data_list2,dim=0)
        self.images_tensor3 = torch.stack(data_list3,dim=0)

    def __len__(self):
        """
        Returns length of dataset       (required by DataLoader)
        """
        return len(self.data_array) # the number of patients in the dataset

    def __getitem__(self, index):
        """
        Allows indexing of dataset      (required by DataLoader)
        Returns a tensor that contains the patient's MRI neuroimages and their diagnoses (AD or MCI)
        """
        
        return [self.images_tensor1[index], self.images_tensor2[index], self.images_tensor3[index]], self.patient_label[index]



# x1 = torch.randn(1,160,160,20)
# x2 = torch.randn(1,160,160,20)
# x3 = torch.randn(1,160,160,20)

# x = []
# x.append(x1)
# x.append(x2)
# x.append(x3)
# images_tensor = torch.stack(x,dim=0)
# len(images_tensor)
# images_tensor.shape


# images_tensor.shape
# images_tensor[0][0].shape

