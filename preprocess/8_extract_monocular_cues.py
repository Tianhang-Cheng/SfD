# adapted from https://github.com/EPFL-VILAB/omnidata
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
import argparse
import os.path
import glob

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio.v2 as imageio
from datasets.data_info import  omnidata_path, pretrained_models

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')
    parser.add_argument('--task', dest='task', help="normal or depth", default='normal', type=str)
    parser.add_argument(
        '--train_res', type=int, default=800, help='training resolution for nerf')
    parser.add_argument(
        '--instance_num', type=int, default=7, required=True, help='number of instances in the image')
    parser.add_argument(
        '--instance_dir', type=str, default=None, help='instance directory'
    )
    args = parser.parse_args()
    root_dir = pretrained_models
    sys.path.append(omnidata_path)
    print(sys.path)
    # from modules.unet import UNet
    from modules.midas.dpt_depth import DPTDepthModel
    from data.transforms import get_transform

    n = same_obj_num = args.instance_num
    train_res = args.train_res
    instance_dir = args.instance_dir

    # read path
    raw_dir = os.path.join(instance_dir, 'raw')
    temp_dir = os.path.join(raw_dir, 'temp')
    sfm_outputs_dir = os.path.join(temp_dir, 'sfm_outputs')
    real_world = True
    
    # define path
    output_dir = instance_dir
    train_dir = os.path.join(instance_dir, 'train')

    output_path = train_dir
    img_path = os.path.join(output_path, '000_rgb.png')

    trans_topil = transforms.ToPILImage()
    # os.system(f"mkdir -p {args.output_path}")
    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # image_size = imageio.imread(img_path).shape[0]
    image_size = 800 # feed to neural network, not important

    # get target task and model
    if args.task == 'normal': 
        ## Version 1 model
        # pretrained_weights_path = root_dir + 'omnidata_unet_normal_v1.pth'
        # model = UNet(in_channels=3, out_channels=3)
        # checkpoint = torch.load(pretrained_weights_path, map_location=map_location)

        # if 'state_dict' in checkpoint:
        #     state_dict = {}
        #     for k, v in checkpoint['state_dict'].items():
        #         state_dict[k.replace('model.', '')] = v
        # else:
        #     state_dict = checkpoint
        
        
        pretrained_weights_path = root_dir + 'omnidata_dpt_normal_v2.ckpt'
        model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                            transforms.CenterCrop(image_size),
                                            get_transform('rgb', image_size=None)])

    elif args.task == 'depth':
        pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
        # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
        model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                            transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=0.5, std=0.5)])

    else:
        print("task should be one of the following: normal, depth")
        sys.exit()

    trans_rgb = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                    transforms.CenterCrop(image_size),
                                    ])


    def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
        if mask_valid is not None:
            img[~mask_valid] = torch.nan
        sorted_img = torch.sort(torch.flatten(img))[0]
        # Remove nan, nan at the end of sort
        num_nan = sorted_img.isnan().sum()
        if num_nan > 0:
            sorted_img = sorted_img[:-num_nan]
        # Remove outliers
        trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
        trunc_mean = trunc_img.mean()
        trunc_var = trunc_img.var()
        eps = 1e-6
        # Replace nan by mean
        img = torch.nan_to_num(img, nan=trunc_mean)
        # Standardize
        img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
        return img

    def save_outputs(img_path, output_file_name): 
        with torch.no_grad():
            

            print(f'Reading input {img_path} ...')
            img = Image.open(img_path)

            img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)

            rgb_path = os.path.join(output_path, f'{output_file_name}_rgb.png')
            # trans_rgb(img).save(rgb_path)

            if img_tensor.shape[1] == 1:
                img_tensor = img_tensor.repeat_interleave(3,1)

            output = model(img_tensor).clamp(min=0, max=1)

            if args.task == 'depth':
                save_path = os.path.join(output_path, f'000_{args.task}.png')
                #output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
                output = output.clamp(0,1)
                
                np.save(save_path.replace('.png', '.npy'), output.detach().cpu().numpy()[0])
                
                #output = 1 - output
                #output = standardize_depth_map(output)
                plt.imsave(save_path, output.detach().cpu().squeeze(),cmap='viridis')
                
            else:
                save_path = os.path.join(train_dir, '000_normal_pretrain.png')
                #import pdb; pdb.set_trace()
                # np.save(save_path.replace('.png', '.npy'), output.detach().cpu().numpy()[0])  
                # output_norm = torch.norm(output, dim=1, keepdim=True)
                # output = output / output_norm
                mask_path = os.path.join(train_dir, '000_mask.png')
                if os.path.exists(mask_path):
                    mask = imageio.imread(mask_path).astype(bool)
                    output_size = mask.shape[0]
                else:
                    mask = None
                    output_size = 800

                x = output.clone()   
                x[:, 0] =   x[:, 0]
                x[:, 1] = 1-x[:, 1]
                x[:, 2] = 1-x[:, 2] # TODO: check if this is correct 

                # plt.imshow(x[0].permute(1,2,0).detach().cpu().numpy())
                x = F.interpolate(x, size=(output_size, output_size), mode='bilinear', align_corners=False)
                x = ((x).detach()[0].permute(1,2,0).cpu().numpy())
                if mask is not None:
                    x[~mask] = 1.0
                img = Image.fromarray((x * 255).astype(np.uint8))
                imageio.imwrite(save_path, (x*255).astype(np.uint8))
    
            print(f'Writing output {save_path} ...')


    img_path = Path(img_path)
    if img_path.is_file():
        save_outputs(img_path, os.path.splitext(os.path.basename(img_path))[0])
    elif img_path.is_dir():
        for f in glob.glob(img_path+'/*'):
            save_outputs(f, os.path.splitext(os.path.basename(f))[0])
    else:
        print("invalid file path!")
        sys.exit()