from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import torch
from skimage.color import rgb2gray
import torch.nn.functional as F
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
path="./cine_seg.npz" # input the path of cine_seg.npz in your environment
data=np.load(path,allow_pickle=True)
def imgshow(im, cmap=None, rgb_axis=None, dpi=100, figsize=(6.4, 4.8)):
    if isinstance(im, torch.Tensor):
        im = im.to('cpu').detach().cpu().numpy()
    if rgb_axis is not None:
        im = np.moveaxis(im, rgb_axis, -1)
        im = rgb2gray(im)

    plt.figure(dpi=dpi, figsize=figsize)
    norm_obj = Normalize(vmin=im.min(), vmax=im.max())
    plt.imshow(im, norm=norm_obj, cmap=cmap)
    plt.colorbar()
    plt.show()
    plt.close('all')
#把list转换成array
#把list转换成array
data_new=[]
for x in data.files:
    data_new.append(data[x])
data_new=np.array(data_new)
#计算data_new中的label有多少全是0并且删除，data_new的shape为(1798,2,256,256),label是data_new[:,1]
count=0
print(data_new.shape)
for i in range(data_new.shape[0]):
    if np.sum(data_new[i,1])==0:
        count+=1
print(count)
data_new=data_new[data_new[:,1].sum(axis=(1,2))!=0]
#计算data_new中的label的类别数量小于等于3个的并且删除，data_new的shape为(1798,2,256,256),label是data_new[:,1]
count=0
print(data_new.shape)
for i in range(data_new.shape[0]):
    if len(np.unique(data_new[i,1]))<=3:
        count+=1
print(count)
data_new=data_new[np.array([len(np.unique(x))>3 for x in data_new[:,1]])]
print(data_new.shape)
total_samples = len(data_new)
num_train = int(total_samples * 4 / 7)
num_val = int(total_samples * 1 / 7)
num_test = total_samples - num_train - num_val
print(total_samples,num_train, num_val, num_test)
train_input = data_new[:750][:,0]
val_input = data_new[750:1000][:,0]
test_input = data_new[1000:][:,0]
train_output = data_new[:750][:,1]
val_output = data_new[750:1000][:,1]
test_output = data_new[1000:][:,1]
#保存为png图片
#test将第一个图片保存为png
# import cv2
# i=0
# for x in data.files:
#     if i<num_train:
#         #保存到train文件夹
#         cv2.imwrite('dataset/ACDC-2D-threelabel/train/Img/'+x+'.png',data[x][0])
#         cv2.imwrite('dataset/ACDC-2D-threelabel/train/GT/'+x+'.png',data[x][1])
#     if i>=num_train and i<num_train+num_val:
#         #保存到val文件夹
#         cv2.imwrite('dataset/ACDC-2D-threelabel/val/Img/'+x+'.png',data[x][0])
#         cv2.imwrite('dataset/ACDC-2D-threelabel/val/GT/'+x+'.png',data[x][1])
#     if i>=num_train+num_val:
#         #保存到test文件夹
#         cv2.imwrite('dataset/ACDC-2D-threelabel/test/Img/'+x+'.png',data[x][0])
#         cv2.imwrite('dataset/ACDC-2D-threelabel/test/GT/'+x+'.png',data[x][1])
#     i+=1
#将0,85,170,255转换为0,1,2,3
train_output[train_output == 85] = 1
train_output[train_output == 170] = 2
train_output[train_output == 255] = 3
val_output[val_output == 85] = 1
val_output[val_output == 170] = 2
val_output[val_output == 255] = 3
test_output[test_output == 85] = 1
test_output[test_output == 170] = 2
test_output[test_output == 255] = 3
#加入数据增强，现在有1000张图片，每张图片旋转90度和翻转，增加到3000张图片,现在的shape是（1000,128,128），增强后的shape是（3000,128,128）
print(train_input.shape, train_output.shape)
#旋转90度
def rotate90(img):
    return np.rot90(img, k=1, axes=(0, 1))
#翻转
def flip(img):
    return np.flip(img, axis=1)
#应用到训练集，验证集和测试集不需要增强
train_input_aug = np.concatenate([train_input, np.array([rotate90(x) for x in train_input])])
train_output_aug = np.concatenate([train_output, train_output])
train_input_aug = np.concatenate([train_input_aug, np.array([flip(x) for x in train_input])])
train_output_aug = np.concatenate([train_output_aug, train_output])
print(train_input_aug.shape, train_output_aug.shape)
#把numpy数组转变成torch类型，构建loader
train_data = torch.from_numpy(train_input).float()
train_label = torch.from_numpy(train_output).float()
val_data = torch.from_numpy(val_input).float()
val_label = torch.from_numpy(val_output).float()
test_data = torch.from_numpy(test_input) .float()
test_label = torch.from_numpy(test_output) .float()
BZ=10
train = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_label), batch_size=BZ, shuffle=True)
val = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_data, val_label), batch_size=BZ, shuffle=True)
test = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data, test_label), batch_size=BZ, shuffle=True)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        #use nn.Sequential(),BatchNorm2d,ReLU. Use padding to keep the size of image.
        ######################## WRITE YOUR ANSWER BELOW ########################
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        #########################################################################

    def forward(self, x):
        """
        x: [B, C_in, H, W]
        out: [B, C_out, H, W]
        """
        out = self.layers(x)
        return out

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        #use nn.Sequential,nn.MaxPool2d and DoubleConv defined by you.
        ######################## WRITE YOUR ANSWER BELOW ########################
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        #########################################################################

    def forward(self, x):        
        ######################## WRITE YOUR ANSWER BELOW ########################
        return self.maxpool_conv(x)
        #########################################################################
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()        
        
        #use nn.ConvTranspose2d for upsampling.
        ######################## WRITE YOUR ANSWER BELOW ########################
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        #########################################################################
        self.conv = DoubleConv(in_channels, out_channels)
        

    def forward(self, x1, x2):        
        
        
        x1 = self.up(x1)
        H1, W1 = x1.shape[2:]
        H2, W2 = x2.shape[2:]
        
        # print('before padding:', x1.shape)
        
        #use F.pad to change the shape of x1.
        ######################## WRITE YOUR ANSWER BELOW ########################
        x1 = F.pad(x1, [
            (W2-W1) // 2, # left
            (W2-W1) // 2, # right
            (H2-H1) // 2, # top
            (H2-H1) // 2  # bottom
            ])
        #########################################################################
        # print('after padding: ',x1.shape)

        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        
        return out
class MyUNet(nn.Module):
    def __init__(self, n_channels, n_classes, C_base=64):
        super(MyUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        #Use class defined by you. Be careful about params here.
        ######################## WRITE YOUR ANSWER BELOW ########################
        self.in_conv = DoubleConv(n_channels, C_base)

        self.down1 = Down(C_base, 2 * C_base)
        self.down2 = Down(2 * C_base, 4 * C_base)
        self.down3 = Down(4 * C_base, 8 * C_base)
        self.down4 = Down(8 * C_base, 16 * C_base)
        self.up1 = Up(16 * C_base, 8 * C_base)
        self.up2 = Up(8 * C_base, 4 * C_base)
        self.up3 = Up(4 * C_base, 2 * C_base)
        self.up4 = Up(2 * C_base, C_base)
        #########################################################################
        self.out_projection = nn.Conv2d(C_base, n_classes, kernel_size=1)
        

    def forward(self, x):
        """
        :param x: [B, n_channels, H, W]
        :return [B, n_classes, H, W]
        """ 
        ######################## WRITE YOUR ANSWER BELOW ########################
        x1 = self.in_conv(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        #########################################################################
        pred = self.out_projection(x)        
        
        return pred


from segment_anything.modeling import ImageEncoderViT, TwoWayTransformer
from functools import partial
import torch
class MySamFeatSeg(nn.Module):
    def __init__(
        self,
        image_encoder,
        seg_decoder,
        img_size = 1024,
    ):
        super().__init__()
        self.img_size = img_size
        self.image_encoder = image_encoder
        self.mask_decoder = seg_decoder

    def forward(self,
                x):
        original_size = x.shape[-1]
        x = F.interpolate(
            x,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        image_embedding = self.image_encoder(x) #[B, 256, 64, 64]
        out = self.mask_decoder(image_embedding)

        if out.shape[-1] != original_size:
            out = F.interpolate(
                out,
                (original_size, original_size),
                mode="bilinear",
                align_corners=False,
            )
        return out
    def freeze_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False
    def get_embedding(self, x):
        original_size = x.shape[-1]
        x = F.interpolate(
            x,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        image_embedding = self.image_encoder(x)
        out = nn.functional.adaptive_avg_pool2d(image_embedding, 1).squeeze()
        return out

class MySegDecoderCNN(nn.Module):
    def __init__(self,
                 num_classes=4,
                 embed_dim=256,
                 num_depth=1,
                 top_channel=64,
                 ):
        super().__init__()

        self.input_block = nn.Sequential(
            nn.Conv2d(embed_dim, top_channel*2**num_depth, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(top_channel * 2 ** num_depth, top_channel * 2 ** num_depth, kernel_size=1),
            nn.ReLU(inplace=True),

        )
        self.blocks = nn.ModuleList()
        for i in range(num_depth):
            if num_depth > 2 > i:
                block = nn.Sequential(
                    nn.Conv2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i - 1), 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(top_channel * 2 ** (num_depth - i),  top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(top_channel * 2 ** (num_depth - i),  top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i - 1), 2,
                                       stride=2)
                )

            self.blocks.append(block)

        self.final = nn.Sequential(
            nn.Conv2d(top_channel, top_channel, 3, padding=1),
            nn.Conv2d(top_channel, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.input_block(x)
        for blk in self.blocks:
            x = blk(x)
        return self.final(x)
# encoder_depth = 
encoder_depth=32
encoder_embed_dim=1280
encoder_num_heads=16
prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
encoder_global_attn_indexes=[7, 15, 23, 31]
SegModel = MySamFeatSeg(
    image_encoder=ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    ),
    #seg_decoder=SegDecoderLinear(num_classes=num_classes),
    # seg_decoder = MyUNet(n_channels=1, n_classes=4)
    seg_decoder=MySegDecoderCNN(num_classes=4, num_depth=2)
)
with open("sam_vit_h_4b8939.pth", "rb") as f:
     state_dict = torch.load(f)

loaded_keys = []
for k in state_dict.keys():
     if k in SegModel.state_dict().keys():
         loaded_keys.append(k)
SegModel.load_state_dict(state_dict, strict=False)
# Model, Loss, and Optimizer
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# Model, Loss, and Optimizer
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import os 
torch.cuda.set_device(2)

import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SegModel.to(device)
model.freeze_encoder()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop (simplified)

# torch.save(model.state_dict(), 'model_final_Unet_depth2.pth')
model.load_state_dict(torch.load('model_final_Unet_depth2.pth'))
model.eval()
input=val_data[101].to(device)
label=val_label[101].to(device)
output=model(input.reshape(1,1,256,256).repeat(1, 3, 1, 1))
_, predicted_indices = torch.max(output, 1)  # 使用torch.max获取类别索引，第一个维度是类别维

# 将类别索引映射到灰度值
# 创建一个映射数组
mapping = torch.tensor([0, 85, 170, 255], dtype=torch.uint8).to(predicted_indices.device)
# 使用映射数组将类别索引转换为相应的灰度值
final_image = mapping[predicted_indices]
#展示val81的原始图片
print(val_label[101].shape)
print(np.unique(val_label[101]))
#输出每个类别的像素数量
for i in range(4):
    a=val_label[101]==i
    print(sum(sum(a)))

def dice_coefficient(pred, true, label):
    index={85:1,170:2,255:3}
    # 获取指定标签的二值图像
    pred_binary = (pred == label)
    true_binary = (true == index[label])
    #显示二值图像
    
    # 计算交集
    intersection = np.sum(pred_binary & true_binary)
    # 计算每个图像的目标区域像素数量
    pred_sum = np.sum(pred_binary)
    true_sum = np.sum(true_binary)
    
    # 计算Dice系数
    if pred_sum + true_sum == 0:
        return 1.0  # 如果预测和真实图像中都没有这个标签，认为Dice系数为1
    dice = (2. * intersection) / (pred_sum + true_sum)
    
    return dice
pred=final_image.reshape(256,256).cpu().numpy()
true=label.cpu().numpy()
dice_rv = dice_coefficient(pred, true, 85)
dice_myo = dice_coefficient(pred, true, 170)
dice_lv = dice_coefficient(pred, true, 255)

print(f'RV Dice系数: {dice_rv}')
print(f'MYO Dice系数: {dice_myo}')
print(f'LV Dice系数: {dice_lv}')
#计算验证集上的Dice系数
def dice_coefficient(pred, true, label):
    index={85:1,170:2,255:3}
    # 获取指定标签的二值图像
    pred_binary = (pred == label)
    true_binary = (true == index[label])
    imgs = np.zeros((256,256))
    imgs[pred_binary] = 255
    cv2.imwrite("nsml.png",imgs)
    true = np.zeros((256,256))
    true[true_binary] = 255
    cv2.imwrite("true.png",true)
    # 计算交集
    intersection = np.sum(pred_binary & true_binary)
    # 计算每个图像的目标区域像素数量
    pred_sum = np.sum(pred_binary)
    true_sum = np.sum(true_binary)
    
    # 计算Dice系数
    if pred_sum + true_sum == 0:
        return 1.0  # 如果预测和真实图像中都没有这个标签，认为Dice系数为1
    dice = (2. * intersection) / (pred_sum + true_sum)
    
    return dice
test_data=val_data.to(device)
test_label=val_label.to(device)
model = model.to(device)
total_rv = 0
total_myo = 0
total_lv = 0
cnt_rv=cnt_myo=cnt_lv=0
min_rv=min_myo=min_lv=1
index_min_rv=index_min_myo=index_min_lv=0
for i in range(len(val_data)):
    input=val_data[i].to(device)
    label=val_label[i].to(device)
    output=model(input.reshape(1,1,256,256).repeat(1, 3, 1, 1))
    _, predicted_indices = torch.max(output, 1)  # 使用torch.max获取类别索引，第一个维度是类别维

    # 将类别索引映射到灰度值
    # 创建一个映射数组
    mapping = torch.tensor([0, 85, 170, 255], dtype=torch.uint8).to(predicted_indices.device)
    # 使用映射数组将类别索引转换为相应的灰度值
    final_image = mapping[predicted_indices]
    pred=final_image.reshape(256,256).cpu().numpy()
    true=label.cpu().numpy()
    # print(pred)
    dice_rv = dice_coefficient(pred, true, 85)
    dice_myo = dice_coefficient(pred, true, 170)
    dice_lv = dice_coefficient(pred, true, 255)
    if dice_rv<min_rv:
        min_rv=dice_rv
        index_min_rv=i
    if dice_myo<min_myo:
        min_myo=dice_myo
        index_min_myo=i
    if dice_lv<min_lv:
        min_lv=dice_lv
        index_min_lv=i
    if dice_rv!=0:
        total_rv += dice_rv
        cnt_rv+=1
    if dice_myo!=0:
        total_myo += dice_myo
        cnt_myo+=1
    if dice_lv!=0:
        total_lv += dice_lv
        cnt_lv+=1
total_rv /= cnt_rv
total_myo /= cnt_myo
total_lv /=  cnt_lv
print(f'RV Dice系数: {total_rv}')
print(f'MYO Dice系数: {total_myo}')
print(f'LV Dice系数: {total_lv}')
print(min_rv,index_min_rv)
print(min_myo,index_min_myo)
print(min_lv,index_min_lv)