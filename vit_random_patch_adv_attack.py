import torch
import torch.nn as nn
from torchvision.models.inception import InceptionAux
from torchvision.models import vit_b_16, vit_b_32, ViT_B_16_Weights, ViT_B_32_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
from masked_fgsm import FGSM as FGSM_atk
from masked_cw import CW as CW_atk
from masked_pgdl2 import PGDL2 as PGDL2_atk
import sys
from utils import Logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--index_model', type = int, default = 0)
parser.add_argument('--index_dataset', type = int, default = 0)
parser.add_argument('--path_ori_data', type = str, default = None)
parser.add_argument('--path_vit_model', type = str, default = None)
parser.add_argument('--log_dst_path', type=str, default=None)
parser.add_argument('--numb_patches', type = float, default=-1)
parser.add_argument('--batch_size', type = int, default=32)
args = parser.parse_args()

_model_names = ['VIT_B_16','VIT_B_32']
_dataset_names = ['CIFAR10','CIFAR100','MNIST','IMGNET']
_adver_methods = ['PGD','CW' ,'FGSM']
_patch_sizes = {'VIT_B_16': 16, 'VIT_B_32': 32}

model_name = _model_names[args.index_model]
dataset_name = _dataset_names[args.index_dataset]

def initModel(name, path = None):
    _out_features = {"CIFAR10": 10, "CIFAR100": 100, "MNIST": 10}
    if(name == 'VIT_B_16'):
        model = vit_b_16(weights = ViT_B_16_Weights.DEFAULT)
        preprocess = ViT_B_16_Weights.DEFAULT.transforms
        if(dataset_name != 'IMGNET'): 
            model.heads.head = torch.nn.Linear(in_features = model.heads.head.in_features, out_features = _out_features[dataset_name])
    if(name == 'VIT_B_32'):
        model = vit_b_32(weights = ViT_B_32_Weights.DEFAULT)
        preprocess = ViT_B_32_Weights.DEFAULT.transforms
        if(dataset_name != 'IMGNET'): 
            model.heads.head = torch.nn.Linear(in_features = model.heads.head.in_features, out_features = _out_features[dataset_name])            
    if(path != None and path!="None"):
        model.load_state_dict(torch.load(path, weights_only = True))
    return model.to(device), preprocess

_mean = (0.485, 0.456, 0.406)
_std = (0.229, 0.224, 0.225)
_width_vit_model = 224
_height_vit_model = 224

transforms_vit_model = {
    "CIFAR10": {
        'train':
        transforms.Compose([
            transforms.Resize((_width_vit_model, _height_vit_model)),
            #transforms.RandomAffine(0,scale=(0.8, 1.2)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(), # ngang
            transforms.ToTensor(),
            transforms.Normalize(_mean , _std)
        ]),
        'validate':
        transforms.Compose([
            transforms.Resize((_width_vit_model, _height_vit_model)),
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std)
        ])
    },
    "CIFAR100":{
        'train':
        transforms.Compose([
            transforms.Resize((_width_vit_model, _height_vit_model)),
            #transforms.RandomAffine(0,scale=(0.8, 1.2)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(), # ngang
            transforms.ToTensor(),
            transforms.Normalize(_mean , _std)
        ]),
        'validate':
        transforms.Compose([
            transforms.Resize((_width_vit_model, _height_vit_model)),
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std)
        ])
    },
    "MNIST":{
        'train':
        transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize((_width_vit_model, _height_vit_model)),
            #transforms.RandomAffine(0,scale=(0.8, 1.2)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(), # ngang
            transforms.ToTensor(),
            transforms.Normalize(_mean , _std)
        ]),
        'validate':
        transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize((_width_vit_model, _height_vit_model)),
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std)
        ]),
    },
    "IMGNET":{
        'train':
        transforms.Compose([
            transforms.Resize((_width_vit_model, _height_vit_model)),
            #transforms.RandomAffine(0,scale=(0.8, 1.2)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(), # ngang
            transforms.ToTensor(),
            transforms.Normalize(_mean , _std)
        ]),
        'validate':
        transforms.Compose([
            transforms.Resize((_width_vit_model, _height_vit_model)),
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std)
        ]),
    }
}

def initData(name, preprocess, path):
    if(name == 'CIFAR10'):
        test_dataset = datasets.CIFAR10(
            root = path,
            train = False,
            download = True,
            transform = preprocess["validate"]
        )
    elif(name == 'CIFAR100'):
        test_dataset = datasets.CIFAR100(
            root = path,
            train = False,
            transform = preprocess["validate"],
            download = True
        )
    elif(name == 'MNIST'):
        test_dataset = datasets.MNIST(
            root = path,
            train = False,
            transform = preprocess["validate"],
            download = True 
        )
    elif(name == 'IMGNET'):
        test_dataset = datasets.ImageNet(root = path, split = "val", transform = preprocess["validate"])    
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True)
    return test_dataloader


def FGSM(model, imgs, labels, epsilon, mask):
    atk = FGSM_atk(model = model, eps = epsilon, mask = mask)
    adv_imgs = atk(imgs, labels)
    return adv_imgs

def PGD(model, imgs, labels, epsilon, nepoch, lr, mask):
    atk = PGDL2_atk(model = model, eps = epsilon, alpha = lr, steps = nepoch, mask = mask)
    adv_imgs = atk(imgs, labels)
    return adv_imgs

def CW(model, imgs, labels, alpha, nepoch, mask):
    atk = CW_atk(model, steps = nepoch, lr = alpha, mask=mask)
    adv_imgs = atk(imgs, labels)
    return adv_imgs

pgd_nepoch = 50
pgd_lr = 0.2

cw_lr = 0.1
cw_nepoch = 100

adv_eps = 0.1

def adver_attk(model, imgs, labels, epsilon, method, mask = None):
    if(method == 'FGSM'):
        return FGSM(model, imgs, labels, epsilon, mask)
    if(method == 'PGD'):
        return PGD(model, imgs, labels, epsilon, pgd_nepoch, pgd_lr, mask)
    if(method == 'CW'):
        return CW(model, imgs, labels, cw_lr, cw_nepoch, mask)
    
def patch_mask_maker(n, c, w, h, p, num_patch):
    perm = {"w": torch.randperm(w//p, device=device), "h": torch.randperm(h//p, device=device)}
    axis = {"w": torch.zeros(w,device=device), "h": torch.zeros(h,device=device)}
    for dak in ["w", "h"]:
        for i in range(0, num_patch):
            axis[dak][perm[dak][i]*p:(perm[dak][i]+1)*p]=1
    iter_mask = (axis["w"][:,None]*axis["h"][None,:]).unsqueeze(0)
    iter_mask = torch.concat([iter_mask, iter_mask, iter_mask], dim=0).unsqueeze(0)
    mask = torch.empty((0, c, w, h), device = device)
    for iter in range(0, n):
        mask = torch.concat([mask, iter_mask], dim=0)
    return mask

def random_patch_adver_attks(model, imgs, labels, epsilon, method, patch_size, number_of_patches):
    N, C, W, H = imgs.shape
    patch_mask = patch_mask_maker(N, C, W, H, patch_size, int(number_of_patches * W//patch_size))
    adv_imgs = adver_attk(model, imgs, labels, epsilon, method, mask=patch_mask)
    #adv_imgs = imgs
    return adv_imgs


def get_accuracy(model, loader, adver_method, epsilon):
    model.eval()
    running_corrects = {"ori": {"top 1": 0, "top 5": 0}, "adver": {"top 1": 0, "top 5": 0}}
    accuracy = {"ori": {"top 1": 0, "top 5": 0}, "adver": {"top 1": 0, "top 5": 0}}
    num_imgs = 0
    patch_size = _patch_sizes[model_name]
    number_of_patches = args.numb_patches
    print(adver_method)
    print("patch size: {}".format(patch_size))
    print("number of patches: {}".format(number_of_patches))
    for batch_idx, (imgs, labels) in enumerate(loader):
        if((batch_idx+1)%20==0 or batch_idx==0):
            print("batch #{}".format(batch_idx+1))
            for dakmim in ["ori", "adver"]:
                for dakmim2 in ["top 1", "top 5"]:
                    print("{} - {}: {}/{}".format(dakmim, dakmim2, running_corrects[dakmim][dakmim2], num_imgs))   
        num_imgs += imgs.shape[0]
        imgs = imgs.to(device)
        labels = labels.to(device)
        adver_imgs = random_patch_adver_attks(model, imgs, labels, epsilon, adver_method, patch_size, number_of_patches)
        outs = {"ori": model(imgs), "adver": model(adver_imgs)}
        for dakmim in ["ori", "adver"]:
            for logits_idx, logits in enumerate(outs[dakmim]):
                preds = torch.argsort(F.softmax(logits, dim=-1), dim = -1, descending=True)
                running_corrects[dakmim]["top 1"] += (labels[logits_idx] == preds[0])
                running_corrects[dakmim]["top 5"] += (labels[logits_idx] in preds[:5])
    for dakmim in ["ori", "adver"]:
        for dakmim2 in ["top 1", "top 5"]:
            print("{} - {}: {}/{}".format(dakmim, dakmim2, running_corrects[dakmim][dakmim2], num_imgs))
            accuracy[dakmim][dakmim2] = running_corrects[dakmim][dakmim2] / num_imgs
    return accuracy

if __name__ == '__main__':
    try:
        vit_model, _ = initModel(model_name, args.path_vit_model)
        dataloader = initData(dataset_name, transforms_vit_model[dataset_name], args.path_ori_data)
        sys.stdout = Logger(args.log_dst_path)
        print("Adversarial attacks with number of patches *= {}".format(args.numb_patches))
        print("On model {} - dataset{}".format(model_name, dataset_name))
        for adver_method in _adver_methods:
            accuracy = get_accuracy(vit_model, dataloader, adver_method, adv_eps)
            print("{} ori - top 1: {} - top 5: {}".format(adver_method, accuracy["ori"]["top 1"], accuracy["ori"]["top 5"]))
            print("{} adver - top 1: {} - top 5: {}".format(adver_method, accuracy["adver"]["top 1"], accuracy["adver"]["top 5"]))

    except KeyboardInterrupt:
        print("Press Ctrl-C to terminate while statement")