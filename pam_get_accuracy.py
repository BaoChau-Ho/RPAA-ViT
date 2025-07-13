import torch
import torch.nn as nn
from torchvision.models.inception import InceptionAux
from torchvision.models import vit_b_16, resnet50, resnet101, mobilenet_v2, alexnet, densenet121, inception_v3, ViT_B_16_Weights, ResNet50_Weights, ResNet101_Weights, MobileNet_V2_Weights, AlexNet_Weights, DenseNet121_Weights, Inception_V3_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import torchattacks
import sys
from utils import Logger
from attention import PatchAttentionModel3200
from attention_nw import PatchAttentionModel
from pynvml import *
from os import system

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--model_vit_path', type = str, default = None)
parser.add_argument('--model_pam_path', type = str, default = None)
parser.add_argument('--sigma_boi')
parser.add_argument('--ori_data_path', type = str, default = None)
parser.add_argument('--log_dst_path', type = str, default = None)
parser.add_argument('--index_dataset', type = int, default = 0)
parser.add_argument('--index_method', type = int, default = 0)
parser.add_argument('--batch_size', type = int, default = 32)
args = parser.parse_args()

_model_names = ['RESNET50', 'RESNET101', 'MOBILENET_V2', 'ALEXNET', 'DENSENET121', 'INCEPTION_V3']
_dataset_names = ['CIFAR10','CIFAR100','MNIST','IMGNET']
_method_names = ['PGD','CW','FGSM']

adver_method = _method_names[args.index_method]
dataset_name = _dataset_names[args.index_dataset]

print("Dataset name: {}".format(dataset_name))
print("Adversarial attack: {}".format(adver_method))

_mean = (0.485, 0.456, 0.406)
_std = (0.229, 0.224, 0.225)
_width_vit_model = 224
_height_vit_model = 224

def initModel(name, path = None):
    _out_features = {"CIFAR10": 10, "CIFAR100": 100, "MNIST": 10}
    if(name == 'VIT_B_16'):
        model = vit_b_16(weights = ViT_B_16_Weights.DEFAULT)
        if(dataset_name != 'IMGNET'): 
            model.heads.head = torch.nn.Linear(in_features = model.heads.head.in_features, out_features = _out_features[dataset_name])
    if(name=='PAM'):
        if(args.sigma_boi==3200):
            model = PatchAttentionModel3200(
                img_size = _width_vit_model, 
                patch_size = 16,
                hidden_dim = 956,
                n_heads = 8,
                attn_threshold=0.0
            )
        else:
            model = PatchAttentionModel(
                img_size = _width_vit_model, 
                patch_size = 16, 
                hidden_dim = 956,
                n_heads = 8,
                attn_threshold=0.0,
                sigma = args.sigma_boi
            )
    if(path != None and path!="None"):
        model.load_state_dict(torch.load(path, weights_only = True))
    return model.to(device)

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
            transform = preprocess
        )
    elif(name == 'CIFAR100'):
        test_dataset = datasets.CIFAR100(
            root = path,
            train = False,
            transform = preprocess,
            download = True
        )
    elif(name == 'MNIST'):
        test_dataset = datasets.MNIST(
            root = path,
            train = False,
            transform = preprocess,
            download = True 
        )
    elif(name == 'IMGNET'):
        test_dataset = datasets.ImageNet(root = path, split = "val", transform = preprocess)
    dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True)
    return dataloader

def denormalize(img, mean, std):
    if(isinstance(mean, list)): mean = torch.tensor(mean)
    if(isinstance(std, list)): std = torch.tensor(std)
    nimg = img * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
    return nimg

def normalize(img, mean, std):
    return transforms.Normalize(mean = mean, std = std)(img)

def FGSM(model, imgs, labels, epsilon):
    atk = torchattacks.FGSM(model = model, eps = epsilon)
    adv_imgs = atk(imgs, labels)
    delta = adv_imgs - imgs
    return delta

def PGD(model, imgs, labels, epsilon, nepoch, lr):
    atk = torchattacks.PGDL2(model = model, eps = epsilon, alpha = lr, steps = nepoch)
    adv_imgs = atk(imgs, labels)
    delta = adv_imgs - imgs
    return delta

def CW(model, imgs, labels, alpha, nepoch):
    atk = torchattacks.CW(model, steps = nepoch, lr = alpha)
    adv_imgs = atk(imgs, labels)
    delta = adv_imgs - imgs
    return delta

pgd_nepoch = 10
pgd_lr = 0.01

cw_lr = 0.01
cw_nepoch = 10

adv_eps = 0.031

def adver_attk(model, imgs, labels, epsilon, method):
    if(method == 'PGD'):
        return PGD(model, imgs, labels, epsilon, pgd_nepoch, pgd_lr)
    elif(method == 'CW'):
        return CW(model, imgs, labels, cw_lr, cw_nepoch)
    elif(method == 'FGSM'):
        return FGSM(model, imgs, labels, epsilon)
    
def get_accuracy(model_vit, model_pam,  loader, adver_method, epsilon):
    running_corrects = {"ori": {"top 1": 0, "top 5": 0}, "adver": {"top 1": 0, "top 5": 0}}
    accuracy = {"ori": {"top 1": 0, "top 5": 0}, "adver": {"top 1": 0, "top 5": 0}}
    num_imgs = 0
    print(adver_method)
    for batch_idx, (imgs, labels) in enumerate(loader):
        if(batch_idx%100==99 or batch_idx==0):
            print("batch #{}".format(batch_idx+1))
            for dakmim in ["ori", "adver"]:
                for dakmim2 in ["top 1", "top 5"]:
                    print("{} - {}: {}/{}".format(dakmim, dakmim2, running_corrects[dakmim][dakmim2], num_imgs))   
        num_imgs += imgs.shape[0]
        imgs = imgs.to(device)
        labels = labels.to(device)
        delta = adver_attk(model_vit, imgs, labels, epsilon, adver_method)
        adv_imgs = model_pam(imgs,delta)
        outs = {"ori": model_vit(imgs), "adver": model_vit(adv_imgs)}
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
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)        
    try:
        gpu_busy = True
        while(gpu_busy):
            info = nvmlDeviceGetMemoryInfo(h)
            print(f'used     : {int(info.used/(1.05*1000000))}',
            end="\r", flush=True)
            gpu_busy = int(info.used/(1.05*1000000)) > 7000
        system("cls")                
        sys.stdout = Logger(args.log_dst_path)
        model_vit = initModel('VIT_B_16', args.model_vit_path)
        model_pam = initModel('PAM',args.model_pam_path)
        print(args.model_pam_path)
        dataloader = initData(dataset_name, transforms_vit_model[dataset_name]["validate"],args.ori_data_path)
        get_accuracy(model_vit, model_pam, dataloader, adver_method, adv_eps)

    except KeyboardInterrupt:
        print("Press Ctrl-C to terminate while statement")