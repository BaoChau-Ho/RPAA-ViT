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
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--index_model', type = int, default = 0)
parser.add_argument('--index_dataset', type = int, default = 0)
parser.add_argument('--path_ori_data', type = str, default = None)
parser.add_argument('--path_vit_model', type = str, default = None)
parser.add_argument('--log_dst_path', type=str, default=None)
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
        if(dataset_name != 'IMGNET'): 
            model.heads.head = torch.nn.Linear(in_features = model.heads.head.in_features, out_features = _out_features[dataset_name])
    if(name == 'VIT_B_32'):
        model = vit_b_32(weights = ViT_B_32_Weights.DEFAULT)
        if(dataset_name != 'IMGNET'): 
            model.heads.head = torch.nn.Linear(in_features = model.heads.head.in_features, out_features = _out_features[dataset_name])            
    if(path != None and path!="None"):
        model.load_state_dict(torch.load(path, weights_only = True))
    return model.to(device)

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
    
def attn_mask_maker(attn_model, imgs):
    _ = attn_model(imgs)
    cls_weight = attn_model.blocks[-1].attn.cls_attn_map.mean(dim=1).view(-1, 14, 14).detach()
    mask = cls_weight
    print(mask.shape)
    return mask

def attn_adver_attks(model, attn_model, imgs, labels, epsilon, method):
    patch_mask = attn_mask_maker(attn_model, imgs)
    adv_imgs = adver_attk(model, imgs, labels, epsilon, method, mask=patch_mask)
    #adv_imgs = imgs
    return adv_imgs


def get_accuracy(model, attn_model, loader, adver_method, epsilon):
    model.eval()
    running_corrects = {"ori": {"top 1": 0, "top 5": 0}, "adver": {"top 1": 0, "top 5": 0}}
    accuracy = {"ori": {"top 1": 0, "top 5": 0}, "adver": {"top 1": 0, "top 5": 0}}
    num_imgs = 0
    patch_size = _patch_sizes[model_name]
    print(adver_method)
    print("patch size: {}".format(patch_size))
    for batch_idx, (imgs, labels) in enumerate(loader):
        if((batch_idx+1)%20==0 or batch_idx==0):
            print("batch #{}".format(batch_idx+1))
            for dakmim in ["ori", "adver"]:
                for dakmim2 in ["top 1", "top 5"]:
                    print("{} - {}: {}/{}".format(dakmim, dakmim2, running_corrects[dakmim][dakmim2], num_imgs))   
        num_imgs += imgs.shape[0]
        imgs = imgs.to(device)
        labels = labels.to(device)
        adver_imgs =   attn_adver_attks(model, imgs, labels, epsilon, adver_method)
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

def my_forward_wrapper(attn_obj):
    def my_forward(x):
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        attn_obj.cls_attn_map = attn[:, :, 0, 2:]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return my_forward

if __name__ == '__main__':
    try:
        vit_model = initModel(model_name, args.path_vit_model)
        attn_model = timm.create_model('vit_base_patch16_clip_224.openai_ft_in1k', pretrained=True)
        attn_model.eval()
        attn_model.blocks[-1].attn.forward = my_forward_wrapper(attn_model.blocks[-1].attn)
        dataloader = initData(dataset_name, transforms_vit_model[dataset_name], args.path_ori_data)
        sys.stdout = Logger(args.log_dst_path)
        print("Adversarial attacks with Attention Mask")
        print("On model {} - dataset{}".format(model_name, dataset_name))
        for adver_method in _adver_methods:
            accuracy = get_accuracy(vit_model, attn_model, dataloader, adver_method, adv_eps)
            print("{} ori - top 1: {} - top 5: {}".format(adver_method, accuracy["ori"]["top 1"], accuracy["ori"]["top 5"]))
            print("{} adver - top 1: {} - top 5: {}".format(adver_method, accuracy["adver"]["top 1"], accuracy["adver"]["top 5"]))

    except KeyboardInterrupt:
        print("Press Ctrl-C to terminate while statement")