from tqdm import tqdm
import argparse
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import scipy.io as sio
from models import DenoisingModel
from dataloader import get_test_data_SR
import utils
from skimage import img_as_ubyte


parser = argparse.ArgumentParser(description='Super-resolve images of RealSR dataset')
parser.add_argument('--input_dir', default='./data/bsd68',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/super_resolution/model_SR_x',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--scale', default='4', type=str, help='Scale factor for super-resolution')
parser.add_argument('--save_images', action='store_true', help='Save super-resolved images in the result directory')


args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

output_dir = args.result_dir+args.scale

utils.mkdir(output_dir)

test_dataset = get_test_data_SR(args.input_dir+args.scale+'/LR/')

test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=8, drop_last=False)

model_denoise = DenoisingModel()

weights = args.weights+args.scale+'.pth'
utils.load_checkpoint(model_denoise, weights)
print("===>Testing using weights: ", weights)

model_denoise.cuda()

model_restoration=nn.DataParallel(model_denoise)

model_denoise.eval()

with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        input_img = data_test[0].cuda()
        filenames = data_test[1]
        rgb_restored = model_denoise(input_img)
        rgb_restored = torch.clamp(rgb_restored,0,1)
     
        input_img = input_img.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        if args.save_images:
            for batch in range(len(input_img)):
                sr_img = img_as_ubyte(rgb_restored[batch])
                utils.save_img(os.path.join(output_dir, filenames[batch][:-4]+'.png'), sr_img)