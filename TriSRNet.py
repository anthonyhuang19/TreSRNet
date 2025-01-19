import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class DualChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(DualChannelAttention, self).__init__()
        
        # Global Context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1_global = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu_global = nn.ReLU(inplace=True)
        self.fc2_global = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        
        # Local Context
        self.local_pool = nn.AdaptiveAvgPool2d(3)
        self.fc1_local = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu_local = nn.ReLU(inplace=True)
        self.fc2_local = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global context attention
        global_context = self.global_pool(x)
        global_attention = self.fc2_global(self.relu_global(self.fc1_global(global_context)))
        
        # Local context attention
        local_context = self.local_pool(x)
        local_attention = self.fc2_local(self.relu_local(self.fc1_local(local_context)))
        
        # Combine both global and local attention
        combined_attention = global_attention + local_attention
        combined_attention = F.interpolate(combined_attention, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        return x * self.sigmoid(combined_attention)

class MultiScaleSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleSpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=7, padding=3, bias=False)
        self.conv_out = nn.Conv2d(3 * in_channels // 2, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat1 = F.relu(self.conv1(x))
        feat2 = F.relu(self.conv2(x))
        feat3 = F.relu(self.conv3(x))
        combined_feats = torch.cat([feat1, feat2, feat3], dim=1)
        att_map = self.conv_out(combined_feats)
        return x * self.sigmoid(att_map)

class ECAAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(ECAAttention, self).__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.conv(avg_out)
        return x * self.sigmoid(avg_out)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = self.avg_pool(x)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        return x * self.sigmoid(se)

class SemanticFusionAttention(nn.Module):
    def __init__(self, in_channels):
        super(SemanticFusionAttention, self).__init__()
        self.global_attention = ECAAttention(in_channels)  # Replace NonLocalBlock with ECA
        self.local_attention = MultiScaleSpatialAttention(in_channels)
        self.se_attention = SEBlock(in_channels)  # Add squeeze and excitation

    def forward(self, x):
        global_features = self.global_attention(x)  # Apply global attention using ECA
        local_features = self.local_attention(x)  # Apply local attention
        se_features = self.se_attention(x)  # Apply squeeze-and-excitation
        return global_features + local_features + se_features  # Combine all

class TripleAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(TripleAttention, self).__init__()
        self.channel_att = DualChannelAttention(in_channels=in_channels, reduction=reduction)
        self.spatial_att = MultiScaleSpatialAttention(in_channels=in_channels)
        self.self_att =  SemanticFusionAttention(in_channels)

    def forward(self, x):
        residual = x
        X_channel = self.channel_att(x)
        X_spatial = self.spatial_att(x)
        X_fusion = self.self_att(x)
        return (X_channel + X_spatial + X_fusion) / 3 + residual


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, n_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attention = TripleAttention(in_channels=input_dim, reduction=16)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, input_dim)
        )
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.layer_norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        return x

class ChannelWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelWiseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=in_channels, bias=False)

    def forward(self, x):
        return self.conv(x)
    
class SuperResolutionTransformer(nn.Module):
    def __init__(self, input_channels, output_channels, n_heads, ff_dim, n_blocks):
        super(SuperResolutionTransformer, self).__init__()
        self.input_conv = nn.Sequential(
            nn.ConvTranspose2d(512,128, kernel_size=8, stride=16, padding=2),
            nn.Upsample(size=(128,128), mode='bilinear', align_corners=False)
        )
        self.layer_norm_input = nn.LayerNorm(128)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(128, n_heads, ff_dim) for _ in range(n_blocks)])
        self.layer_norm_output = nn.LayerNorm(128)
        self.ccm = ChannelWiseConv(128, 128)
        self.reverse_conv = nn.Sequential(nn.Conv2d(128,1024, kernel_size=4, stride=16, padding=1))

    def forward(self, x):
        x = self.input_conv(x)
        x = self.layer_norm_input(x)
        prev_input = x
        
        for block in self.transformer_blocks:
            x = block(x)
            x = self.layer_norm_output(x + prev_input)
            prev_input = x
       
        x = self.ccm(x)
       
        x = self.reverse_conv(x)
       
        return x 

class DenseBlock(nn.Module):
    def __init__(self,in_channels,channels=32,residual_beta=0.2):
        super(DenseBlock,self).__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()

        for i in range(5):
            self.blocks.append(ConvBlock(in_channels=in_channels + channels *i,out_channels=channels if i <= 3 else in_channels,kernel_size=3,stride=1,padding=1,use_act= True if i <= 3 else False,))
    
    def forward(self,x):
        new_input = x
        for block in self.blocks:
            out = block(new_input)
            new_input = torch.cat([new_input,out], dim=1)
        return self.residual_beta * out + x
class RRDB(nn.Module):
    def __init__(self, in_channels,residual_beta =0.2):
        super(RRDB,self).__init__()
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(*[DenseBlock(in_channels=in_channels) for i in range(3)])
    def forward(self, x):
        return self.rrdb(x) * self.residual_beta + x

class TriSRNet(nn.Module):
    def __init__(self, num_classes):
        super(TriSRNet, self).__init__()
        self.num_classes = num_classes
        
        # Contracting path (downsampling)
        self.contract_1 = self.conv_block(in_channels=3, out_channels=64)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contract_2 = self.conv_block(in_channels=64, out_channels=128)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contract_3 = self.conv_block(in_channels=128, out_channels=256)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contract_4 = self.conv_block(in_channels=256, out_channels=512)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.model = SuperResolutionTransformer(input_channels=3, output_channels=3, n_heads=8, ff_dim=256, n_blocks=8)
        self.bottleneck = self.conv_block(in_channels=512, out_channels=512)
        self.rrdb0 = nn.Sequential(*[RRDB(512) for _ in range(6)])
        self.rrdb1 = nn.Sequential(*[RRDB(256) for _ in range(6)])
        self.rrdb2 = nn.Sequential(*[RRDB(128) for _ in range(6)])
        self.rrdb3 = nn.Sequential(*[RRDB(64) for _ in range(6)])
        
        # Expansive path (upsampling)
        self.upconv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.expand_1 = self.conv_block(in_channels=1024, out_channels=512)
        
        self.upconv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.expand_2 = self.conv_block(in_channels=512, out_channels=256)
        
        self.upconv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.expand_3 = self.conv_block(in_channels=256, out_channels=128)
        
        self.upconv_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.expand_4 = self.conv_block(in_channels=128, out_channels=64)
        
        # Final output layer
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels)
        )
        return block
    
    def forward(self, x):
        # Contracting path
        contract_1_out = self.contract_1(x)  # [1, 64, 64, 64]
        pool_1_out = self.pool_1(contract_1_out)  # [1, 64, 32, 32]
        
        contract_2_out = self.contract_2(pool_1_out)  # [1, 128, 32, 32]
        pool_2_out = self.pool_2(contract_2_out)  # [1, 128, 16, 16]
        
        contract_3_out = self.contract_3(pool_2_out)  # [1, 256, 16, 16]
        pool_3_out = self.pool_3(contract_3_out)  # [1, 256, 8, 8]
        
        contract_4_out = self.contract_4(pool_3_out)  # [1, 512, 8, 8]
        pool_4_out = self.pool_4(contract_4_out)  # [1, 512, 4, 4]
        
        # Bottleneck
        bottleneck_out = self.bottleneck(pool_4_out)  # [1, 1024, 4, 4]
        res = self.model(bottleneck_out)
       
        # Expansive path
        upconv_1_out = self.upconv_1(res)
        if upconv_1_out.shape[2] != contract_4_out.shape[2] or upconv_1_out.shape[3] != contract_4_out.shape[3]:
            upconv_1_out = F.interpolate(upconv_1_out, size=(contract_4_out.shape[2], contract_4_out.shape[3]), mode='bilinear', align_corners=False)
        contract_4_out = self.rrdb0(contract_4_out)
        expand_1_out = self.expand_1(torch.cat((upconv_1_out, contract_4_out), dim=1))
        
        upconv_2_out = self.upconv_2(expand_1_out)  # [1, 256, 16, 16]
        contract_3_out = self.rrdb1(contract_3_out)
        expand_2_out = self.expand_2(torch.cat((upconv_2_out, contract_3_out), dim=1))  # [1, 512, 16, 16]
        
        upconv_3_out = self.upconv_3(expand_2_out)  # [1, 128, 32, 32]
        contract_2_out = self.rrdb2(contract_2_out)
        expand_3_out = self.expand_3(torch.cat((upconv_3_out, contract_2_out), dim=1))  # [1, 256, 32, 32]
        
        upconv_4_out = self.upconv_4(expand_3_out)  # [1, 64, 64, 64]
        contract_1_out = self.rrdb3(contract_1_out)
        expand_4_out = self.expand_4(torch.cat((upconv_4_out, contract_1_out), dim=1))  # [1, 128, 64, 64]
        
        # Final output layer
        final_out = self.final_conv(expand_4_out)  # [1, num_classes, 64, 64]
        
        # Upsample to 256x256
        final_out = nn.functional.interpolate(final_out, size=(512, 512), mode='bilinear', align_corners=False)  # [1, num_classes, 256, 256]
        
        return final_out

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,use_act, kernel_size =3,stride =1,padding =1):
        super(ConvBlock,self).__init__()
        self.convolution = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.activation = nn.LeakyReLU(0.2,inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.activation(self.convolution(x))

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=(64,64,128,128,256,256,512,512)):
        super(Discriminator,self).__init__()

        self.in_channels = in_channels
        self.features =list(features)

        blocks = []
        for index, feature in enumerate(self.features):
            blocks.append(ConvBlock(in_channels=in_channels,out_channels=feature,kernel_size=3,stride=1 + index %2,padding=1,use_act=True))
            in_channels = feature
        
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(in_features=512*6*6, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=1)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        features = self.blocks(x)
        out = self.classifier(features)
        out = self.sigmoid(out)
        return out 


# # Example usage:
# start_time = time.time()
# model = TriSRNet(num_classes=3)  

# input_tensor = torch.randn(1, 3,128,128)  
# output_tensor = model(input_tensor)
# print(output_tensor.shape)  # Should print torch.Size([1, 3, 256, 256]) for 3-class output
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Time taken: {elapsed_time:.4f} seconds")