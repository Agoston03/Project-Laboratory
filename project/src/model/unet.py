import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L

class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 dim_out):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv3d(dim, 
                      dim_out, 
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim_out, 
                      dim_out, 
                      kernel_size=3, 
                      padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)

class DownSample(nn.Module):
    def __init__(self, 
                 dim, 
                 dim_out):
        super().__init__()
        self.conv = Block(dim, 
                          dim_out)
        self.pool = nn.AvgPool3d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        pooled = self.pool(x)
        return x, pooled

class UpSample(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.up = nn.ConvTranspose3d(dim, 
                                     dim_out, 
                                     kernel_size=2, 
                                     stride=2) 
        self.conv = Block(dim_out * 2, dim_out) 

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = F.interpolate(input=x1,
                           size=x2.shape[2:],
                           mode='trilinear',
                           align_corners=False)

        print(x1.shape)
        print(x2.shape)

        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)
    
class UNet3D(nn.Module):
    def __init__(self, 
                 channels,
                 init_dim,
                 dim,
                 out_dim,
                 dim_mults,
                 init_kernel_size):
        super().__init__()

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
            
        num_resolutions = len(in_out)

        init_padding = init_kernel_size // 2

        self.init_conv = nn.Conv3d(in_channels=channels,
                                   out_channels=init_dim,
                                   kernel_size=init_kernel_size,
                                   padding=init_padding)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= (num_resolutions - 1)
            if not is_last:
                self.downs.append(DownSample(dim_in, 
                                            dim_out))

        mid_dim1, mid_dim2 = dims[-2], dims[-1]
        self.bottleneck = Block(mid_dim1, mid_dim2)

        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx >= (num_resolutions - 1)
            if not is_last:
                self.ups.append(UpSample(dim_out,
                                        dim_in))

        self.out = nn.Conv3d(in_channels=dim,
                             out_channels=out_dim,
                             kernel_size=1)

    def forward(self, x):
        x = self.init_conv(x)
        skip_connections = []

        for down in self.downs:
            x, pooled = down(x)
            skip_connections.append(x)
            x = pooled

        x = self.bottleneck(x)
        for up, skip in zip(self.ups, 
                            reversed(skip_connections)):
            x = up(x, skip)

        x = self.out(x)
        print(x.shape)
        return x

class LitUNet3D(L.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.lr = learning_rate

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = F.mse_loss(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = F.mse_loss(outputs, labels)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = F.mse_loss(outputs, labels)
        self.log('test_loss', loss, prog_bar=True)
        return loss