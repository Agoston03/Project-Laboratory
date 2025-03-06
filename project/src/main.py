

if __name__ == '__main__':
    import torch
    from dataloader.fmri_data_module import FMRIDataModule
    from model.unet import LitUNet3D, UNet3D
    from pytorch_lightning import Trainer

    base = r'C:\Users\agost\bme\6\Project-Laboratory\project\fmri_data\data'
    fmri_dir = fr'{base}\fmri'
    train_path = fr'{base}\new_format_config\train.csv'
    val_path = fr'{base}\new_format_config\val.csv'
    test_path = fr'{base}\new_format_config\test.csv'

    # Input and Label shape: [batch_size, channels, depth, height, width] => [2, 1, 91, 109, 91]
    datamodule = FMRIDataModule(train_path, 
                                val_path,
                                test_path, 
                                fmri_dir,
                                batch_size=2, 
                                num_workers=4)

    datamodule.setup()

    model = UNet3D(channels=1, 
                   init_dim=1, 
                   dim=64,
                   out_dim=1,
                   dim_mults=[1,2,4,8],
                   init_kernel_size=3)
    lit_model = LitUNet3D(model)

    use_gpu = torch.cuda.is_available()

    trainer = Trainer(
        max_epochs=10,
        accelerator="gpu" if use_gpu else "cpu",  # Use "gpu" if available, otherwise "cpu"
        devices=1 if use_gpu else "auto"  # Specify number of GPUs, "auto" lets PyTorch decide
    )

    trainer.fit(lit_model, datamodule)
