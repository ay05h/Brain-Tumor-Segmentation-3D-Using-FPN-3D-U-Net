import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x_pool = self.pool(x)
        return x_pool, x  


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv3d(2*out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class FPNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPNLayer, self).__init__()
        self.conv = nn.Conv3d(out_channels, out_channels, kernel_size=3,padding=1)
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = x + skip_connection
        x = self.relu(self.conv(x))
        return x

class UNet3D_FPN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNet3D_FPN, self).__init__()
        
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        
        self.bottleneck = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
        
        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)
        
        self.fpn1 = FPNLayer(1024, 512)
        self.fpn2 = FPNLayer(512, 256)
        self.fpn3 = FPNLayer(256, 128)
        self.fpn4 = FPNLayer(128, 64)
        
        self.final_conv = nn.Conv3d(64, output_channels, kernel_size=1)

    def forward(self, x):
        enc1, skip1 = self.encoder1(x)
        enc2, skip2 = self.encoder2(enc1)
        enc3, skip3 = self.encoder3(enc2)
        enc4, skip4 = self.encoder4(enc3)

        bottleneck_output = self.bottleneck(enc4)
        

        fpn1_output = self.fpn1(bottleneck_output, skip4)
        decoder1_output = self.decoder1(bottleneck_output,fpn1_output)
        
        
        fpn2_output = self.fpn2(fpn1_output, skip3)
        decoder2_output = self.decoder2(decoder1_output,fpn2_output)
        
        
        fpn3_output = self.fpn3(fpn2_output, skip2)
        decoder3_output = self.decoder3(decoder2_output,fpn3_output)
        
        
 
        fpn4_output = self.fpn4(fpn3_output, skip1)
        decoder4_output = self.decoder4(decoder3_output,fpn4_output)
    
        output = self.final_conv(decoder4_output)
        
        return output

model = UNet3D_FPN(input_channels=4, output_channels=4)
dummy_input = torch.randn(1, 4, 128, 128, 128)  
output = model(dummy_input)
print(output.shape)
# Dimension Visualisation of model output 
# torch.Size([1, 4, 128, 128, 128])
