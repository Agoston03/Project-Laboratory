from dataloader.fmri_dataset import FMRIDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class FMRIDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_csv,
                 val_csv, 
                 test_csv, 
                 fmri_dir, 
                 batch_size=8, 
                 num_workers=4):
        """
        Initializes the FMRIDataModule.

        Args:
            train_csv (str): Path to training metadata CSV.
            val_csv (str): Path to validation metadata CSV.
            test_csv (str): Path to test metadata CSV.
            fmri_dir (str): Directory containing fMRI data as subdirectories.
            batch_size (int, optional): Batch size. Default is 8.
            num_workers (int, optional): Number of subprocesses for
                                         data loading. Default is 4.
        """
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.fmri_dir = fmri_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = FMRIDataset(self.train_csv, self.fmri_dir)
            self.val_dataset = FMRIDataset(self.val_csv, self.fmri_dir)

        if stage == "test" or stage is None:
            self.test_dataset = FMRIDataset(self.test_csv, self.fmri_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
