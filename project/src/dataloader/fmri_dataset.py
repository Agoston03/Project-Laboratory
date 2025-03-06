import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from nilearn.image import load_img

class FMRIDataset(Dataset):
    def __init__(self, csv_file, fmri_dir):
        """
        Args:
            csv_file (str): Path to the CSV file with metadata
                            containing file IDs and time information.
            fmri_dir (str): Directory where the fMRI data is stored, 
                            organized by subject ID.
        """
        self.data_frame = pd.read_csv(csv_file, sep=';')
        self.fmri_dir = fmri_dir
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        """
        Retrieves a single sample (fMRI data) from the dataset.
        """
        subject_id = self.data_frame['FILE_ID'][idx]
        fmri_data_path = os.path.join(self.fmri_dir, 
                                      str(subject_id), 
                                      'tfMRI_MOTOR_RL.nii.gz')
        
        # Initial shape of an fmri data: [depth (Z), height (Y), width (X), time (T)]
        fmri_img = load_img(fmri_data_path)
        fmri_data = fmri_img.get_fdata()

        start = int(self.data_frame['START_TIME'][idx])
        end = int(self.data_frame['END_TIME'][idx])

        # Slice the fMRI data to get the time period of interest
        # (from 'start' to 'end' index)
        fmri_data = fmri_data[:, :, :, start:end]
        
        # Average the fMRI data along the time axis 
        # (axis 3 corresponds to the time dimension)
        # Shape after mean: (X, Y, Z)
        averaged_data = np.mean(fmri_data, axis=3)  

        # Expected format for convolution: [batch, channels (sensitivity=1), 
        #                                   depth(Z), height(Y), width (X)]
        # Expand the first dimension with 1 because of the sensitivity => channels
        data = np.expand_dims(averaged_data, axis=0)

        # Converts the averaged data to a PyTorch tensor (float32 type)
        data_tensor = torch.from_numpy(data).to(torch.float32)
    
        return data_tensor, data_tensor

