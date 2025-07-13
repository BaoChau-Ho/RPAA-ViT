import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_b_32, ViT_B_32_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import os
from pytorchtools import EarlyStopping
from utils import Logger
from pynvml import *
import copy
from attention_nw2 import PatchAttentionModel
import torchattacks
from matplotlib import pyplot as plt
from os import system
import ml_collections

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--index_model', type = int, default = 0)
parser.add_argument('--index_dataset', type = int, default = 0)
parser.add_argument('--path_ori_data', type = str)
parser.add_argument('--path_vit_model', type = str)
parser.add_argument('--checkpoint', type=int, default = 0)
parser.add_argument('--dst_model_dir',type=str)
parser.add_argument('--adv_method', type = int, default = 0)
parser.add_argument('--log_dst_path', type=str)
parser.add_argument('--batch_size', type = int, default=16)
args = parser.parse_args()

_model_names = ['VIT_B_16','VIT_B_32']
_dataset_names = ['CIFAR10','CIFAR100','MNIST','IMGNET']
_adver_methods = ['PGD', 'CW', 'FGSM']
_patch_sizes = {'VIT_B_16': 16, 'VIT_B_32': 32}

model_name = _model_names[args.index_model]
dataset_name = _dataset_names[args.index_dataset]

_mean = (0.485, 0.456, 0.406)
_std = (0.229, 0.224, 0.225)
_width_vit_model = 224
_height_vit_model = 224

def get_pam_config():
    config = ml_collections.ConfigDict()
    config.num_heads = 8
    config.hidden_dim = 512
    config.attn_dropout_rate = 0.0
    config.img_size = 224
    config.patch_size = 16
    config.embed_dropout_rate = 0.1
    config.mlp_dim = 1024
    config.mlp_dropout_rate = 0.1
    config.number_layers = 6
    config.pam_dropout_rate = 0.1
    return config

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
    if(name=='PAM'):
        model = PatchAttentionModel(get_pam_config())
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
        train_dataset = datasets.CIFAR10(
            root = path,
            train = True,
            download = True,
            transform = preprocess["train"]
        )        
        test_dataset = datasets.CIFAR10(
            root = path,
            train = False,
            download = True,
            transform = preprocess["validate"]
        )
    elif(name == 'CIFAR100'):
        train_dataset = datasets.CIFAR100(
            root = path,
            train = True,
            transform = preprocess["train"],
            download = True
        )        
        test_dataset = datasets.CIFAR100(
            root = path,
            train = False,
            transform = preprocess["validate"],
            download = True
        )
    elif(name == 'MNIST'):
        train_dataset = datasets.MNIST(
            root = path,
            train = True,
            transform = preprocess["train"],
            download = True 
        )
        test_dataset = datasets.MNIST(
            root = path,
            train = False,
            transform = preprocess["validate"],
            download = True 
        )
    if(name=='IMGNET'):
        dataset = datasets.ImageNet(
            root = path,
            split = 'val',
            transform = preprocess["validate"],
        )
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.7, 0.3])
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    return train_dataloader, test_dataloader


def FGSM(model, imgs, labels, epsilon):
    atk = torchattacks.FGSM(model = model, eps = epsilon)
    atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    adv_imgs = atk(imgs, labels)
    delta = adv_imgs - imgs
    return delta

def PGD(model, imgs, labels, epsilon, nepoch, lr):
    atk = torchattacks.PGDL2(model = model, eps = epsilon, alpha = lr, steps = nepoch)
    atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    adv_imgs = atk(imgs, labels)
    delta = adv_imgs - imgs
    return delta

def CW(model, imgs, labels, alpha, nepoch):
    atk = torchattacks.CW(model, steps = nepoch, lr = alpha)
    atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    adv_imgs = atk(imgs, labels)
    delta = adv_imgs - imgs
    return delta

pgd_nepoch = 50
pgd_lr = 0.1

cw_lr = 0.1
cw_nepoch = 100

adv_eps = 0.031

def adver_attk(model, imgs, labels, epsilon, method):
    if(method == 'PGD'):
        return PGD(model, imgs, labels, epsilon, pgd_nepoch, pgd_lr)
    elif(method == 'CW'):
        return CW(model, imgs, labels, cw_lr, cw_nepoch)
    elif(method == 'FGSM'):
        return FGSM(model, imgs, labels, epsilon)

train_nepoch = 100
train_lr = 1e-6
train_gamma = 0.1
train_regu_lambda = 0.001
vit_patch_size = _patch_sizes[model_name]

def l2_regu(model):
    params = torch.concat([x.view(-1) for x in model.parameters()])
    return torch.norm(params, 2)

def train(vit_model, pam_model, dataloader, ckpt_path, adv_method):
    optimizer = torch.optim.Adam(pam_model.parameters(), lr = train_lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = train_gamma)
    early_stopping = EarlyStopping(patience=10, verbose=True, path = ckpt_path)
    best_acc = -1e9
    best_model_wts = copy.deepcopy(pam_model.state_dict())
    criterion = nn.CrossEntropyLoss()
    pam_model = pam_model.to(device)
    for epoch in range(1,train_nepoch+1):
        print("#Epoch: {}/{}".format(epoch, train_nepoch))
        for phase in ["train", "validate"]:
            if(phase == 'train'): pam_model.train()
            elif(phase == 'validate'): pam_model.eval()            
            running_corrects = 0
            running_loss = 0
            num_imgs = 0            
            acc_list = []
            loss_list = []
            for batch_idx, (inps, labels) in enumerate(dataloader[phase]):
                num_imgs+=inps.shape[0]
                inps = inps.to(device)
                labels = labels.to(device)

                delta = adver_attk(vit_model, inps, labels, epsilon = adv_eps, method = adv_method)
                adv_imgs = pam_model(inps,delta)

                outs = vit_model(adv_imgs)
                preds = torch.argmax(F.softmax(outs, dim = -1), dim = -1)
                loss = -criterion(outs, labels)
                corrects = torch.sum(preds == labels)
                if((batch_idx+1)%50==0 or batch_idx==0):
                    print("epoch: {} - batch_idx: {}".format(epoch, batch_idx+1))                
                    print("acc: {} - loss: {}".format(corrects/inps.shape[0], -loss.item()))
                
                acc_list.append((corrects/inps.shape[0]).cpu().numpy())
                loss_list.append(-loss.item())
                running_corrects += corrects
                running_loss += -loss.item()
                if(phase == "train"):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            plt.plot(acc_list)
            #plt.savefig("/media/administrator/Data1/BC/vit_adver/log_result_pam_train/CIFAR10_{}/epoch_{}_phase_{}_acc.png".format(adv_method, epoch, phase))
            plt.clf()
            #plt.savefig("/media/administrator/Data1/BC/vit_adver/log_result_pam_train/CIFAR10_{}/epoch_{}_phase_{}_loss.png".format(adv_method, epoch, phase))
            plt.clf()
            phase_loss = running_loss / num_imgs
            phase_acc = running_corrects / num_imgs
            print('phase:{} - loss: {:.4f} - acc: {:.4f}'.format(phase,phase_loss, phase_acc))
            if(phase == "validate"):
                epoch_loss = phase_loss
                epoch_acc = phase_acc
        lr_scheduler.step()
        if(epoch_acc > best_acc):
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(pam_model.state_dict())
            torch.save(best_model_wts, ckpt_path)

        early_stopping(epoch_loss, pam_model)
        if(early_stopping.early_stop):
            print("Early Stopped")
            break

    pam_model.load_state_dict(best_model_wts)
    return pam_model, best_acc
    


if __name__=='__main__':
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
        print("Training PAM Model")
        print("VIT Model: {}".format(model_name))
        print("Dataset: {}".format(dataset_name))
        print("Adv method: {}".format(_adver_methods[args.adv_method]))
        best_model_path = os.path.join(args.dst_model_dir, "weights_pam2_{}_{}_best.h5".format(_adver_methods[args.adv_method], dataset_name))
        ckpt_model_path = os.path.join(args.dst_model_dir, "weights_pam2_{}_{}_checkpoint.h5".format(_adver_methods[args.adv_method],dataset_name))
        print("Best model path: {}".format(best_model_path))
        print("Checkpoint model path: {}".format(ckpt_model_path))
        vit_model = initModel(model_name, args.path_vit_model)
        train_dataloader, test_dataloader = initData(dataset_name, transforms_vit_model[dataset_name], args.path_ori_data)
        pam_model = initModel("PAM")
        if(args.checkpoint):
            print("Running from checkpoint")
            pam_model.load_state_dict(torch.load(ckpt_model_path,weights_only = True))
        dataloader = {
            "train": train_dataloader,
            "validate": test_dataloader
        }
        best_pam_model, best_acc = train(
            vit_model = vit_model, 
            pam_model = pam_model,
            dataloader=dataloader,
            ckpt_path = ckpt_model_path,
            adv_method = _adver_methods[args.adv_method]
        )
        torch.save(best_pam_model.state_dict(), best_model_path)
        
    except KeyboardInterrupt:
        print("Press Ctrl-C to terminate while statement")