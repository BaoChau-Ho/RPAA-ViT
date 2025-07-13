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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--model_1_path', type = str, default = None)
parser.add_argument('--model_vit_path', type = str, default = None)
parser.add_argument('--ori_data_path', type = str, default = None)
parser.add_argument('--log_dst_path', type = str, default = None)
parser.add_argument('--index_model_1', type = int, default = 0)
parser.add_argument('--index_dataset', type = int, default = 0)
parser.add_argument('--batch_size', type = int, default = 32)
args = parser.parse_args()

_model_names = ['RESNET50', 'RESNET101', 'MOBILENET_V2', 'ALEXNET', 'DENSENET121', 'INCEPTION_V3']
_dataset_names = ['CIFAR10','CIFAR100','MNIST','IMGNET']
#_method_names = ['PGD','CW']
_method_names=['FGSM']

first_model_name = _model_names[args.index_model_1]
dataset_name = _dataset_names[args.index_dataset]

print("First model name: {}".format(first_model_name))
print("Dataset name: {}".format(dataset_name))

def initModel(name, path = None):
    _out_features = {"CIFAR10": 10, "CIFAR100": 100, "MNIST": 10}
    if(name == 'VIT_B_16'):
        model = vit_b_16(weights = ViT_B_16_Weights.DEFAULT)
        preprocess = ViT_B_16_Weights.DEFAULT.transforms
        if(dataset_name != 'IMGNET'): 
            model.heads.head = torch.nn.Linear(in_features = model.heads.head.in_features, out_features = _out_features[dataset_name])
    if('RESNET50' in name):
        model = resnet50(weights = ResNet50_Weights.DEFAULT)
        preprocess = ResNet50_Weights.DEFAULT.transforms
        if(dataset_name != 'IMGNET'): 
            model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=_out_features[dataset_name])
    if('RESNET101' in name):
        model = resnet101(weights = ResNet101_Weights.DEFAULT)
        preprocess = ResNet101_Weights.DEFAULT.transforms
        if(dataset_name!='IMGNET'):
            model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=_out_features[dataset_name])
    if('MOBILENET' in name):
        model = mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)
        preprocess = MobileNet_V2_Weights.DEFAULT.transforms
        if(dataset_name!='IMGNET'):
            model.classifier[1] = nn.Linear(in_features = model.classifier[1].in_features, out_features=_out_features[dataset_name])
    if('ALEXNET' in name):
        model = alexnet(weights = AlexNet_Weights.DEFAULT)
        preprocess = AlexNet_Weights.DEFAULT.transforms
        if(dataset_name!='IMGNET'):
            in_features = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(in_features = in_features, out_features = _out_features[dataset_name])
    if('DENSENET' in name):
        model = densenet121(weights = DenseNet121_Weights.DEFAULT)
        preprocess = DenseNet121_Weights.DEFAULT.transforms
        if(dataset_name!='IMGNET'):
            in_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_features=in_features, out_features=_out_features[dataset_name])
    if('INCEPTION' in name):
        model = inception_v3(weights = Inception_V3_Weights.DEFAULT)
        preprocess = Inception_V3_Weights.DEFAULT.transforms
        if(dataset_name!='IMGNET'):
            in_features = model.fc.in_features
            in_features2 = 768
            model.fc = nn.Linear(in_features=in_features, out_features=_out_features[dataset_name])
            model.AuxLogits = InceptionAux(in_channels = in_features2, num_classes = _out_features[dataset_name])  
    if(path != None and path!="None"):
        model.load_state_dict(torch.load(path, weights_only = True))
    return model.to(device), preprocess

_mean = (0.485, 0.456, 0.406)
_std = (0.229, 0.224, 0.225)
_width_vit_model = 224
_height_vit_model = 224
_width_first_model = 224
_height_first_model = 224
if('inception' in first_model_name):
    _width_first_model = 299
    _height_first_model = 299

transforms_data_first_model = {
    "CIFAR10": {
        'train':
        transforms.Compose([
            transforms.Resize((_width_first_model, _height_first_model)),
            #transforms.RandomAffine(0,scale=(0.8, 1.2)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(), # ngang
            transforms.ToTensor(),
            transforms.Normalize(_mean , _std)
        ]),
        'validate':
        transforms.Compose([
            transforms.Resize((_width_first_model, _height_first_model)),
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std)
        ])
    },
    "CIFAR100":{
        'train':
        transforms.Compose([
            transforms.Resize((_width_first_model, _height_first_model)),
            #transforms.RandomAffine(0,scale=(0.8, 1.2)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(), # ngang
            transforms.ToTensor(),
            transforms.Normalize(_mean , _std)
        ]),
        'validate':
        transforms.Compose([
            transforms.Resize((_width_first_model, _height_first_model)),
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std)
        ])
    },
    "MNIST":{
        'train':
        transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize((_width_first_model, _height_first_model)),
            #transforms.RandomAffine(0,scale=(0.8, 1.2)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(), # ngang
            transforms.ToTensor(),
            transforms.Normalize(_mean , _std)
        ]),
        'validate':
        transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize((_width_first_model, _height_first_model)),
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std)
        ]),
    },
    "IMGNET":{
        'train':
        transforms.Compose([
            transforms.Resize((_width_first_model, _height_first_model)),
            #transforms.RandomAffine(0,scale=(0.8, 1.2)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(), # ngang
            transforms.ToTensor(),
            transforms.Normalize(_mean , _std)
        ]),
        'validate':
        transforms.Compose([
            transforms.Resize((_width_first_model, _height_first_model)),
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std)
        ]),
    }
}

transforms_data_first_to_vit = transforms.Compose([
    transforms.Resize((_width_vit_model, _height_vit_model))
])

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
    return adv_imgs

def PGD(model, imgs, labels, epsilon, nepoch, lr):
    atk = torchattacks.PGDL2(model = model, eps = epsilon, alpha = lr, steps = nepoch)
    adv_imgs = atk(imgs, labels)
    return adv_imgs

def CW(model, imgs, labels, alpha, nepoch):
    atk = torchattacks.CW(model, steps = nepoch, lr = alpha)
    adv_imgs = atk(imgs, labels)
    return adv_imgs

def BIM(model, imgs, labels, epsilon, alpha):
    atk = torchattacks.BIM(model, eps = epsilon, alpha = alpha, steps = 0)
    adv_imgs = atk(imgs, labels)
    return adv_imgs

pgd_nepoch = 100
pgd_lr = 0.001

bim_alpha = 2/225
if("inception" in first_model_name): bim_alpha = 2/299

cw_lr = 0.001
cw_nepoch = 100

adv_eps = 0.031

def adver_attk(model, imgs, labels, epsilon, method):
    if(method == 'FGSM'):
        return FGSM(model, imgs, labels, epsilon)
    if(method == 'PGD'):
        return PGD(model, imgs, labels, epsilon, pgd_nepoch, pgd_lr)
    if(method == 'BIM'):
        return BIM(model, imgs, labels, epsilon, bim_alpha)
    if(method == 'CW'):
        return CW(model, imgs, labels, cw_lr, cw_nepoch)
    
def get_accuracy(model, model_adver, preprocess, loader, adver_method, epsilon):
    model.eval()
    model_adver.eval()
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
        adver_imgs = adver_attk(model_adver, imgs, labels, epsilon, adver_method)
        adver_imgs = preprocess(adver_imgs)
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
        modelVIT, preprocessVIT = initModel('VIT_B_16', args.model_vit_path)
        model_adver, preprocess_adver = initModel(first_model_name, args.model_1_path)
        dataloader = initData(dataset_name, transforms_data_first_model[dataset_name]['validate'], args.ori_data_path)
        sys.stdout = Logger(args.log_dst_path)
        for method_name in _method_names:
            accuracy = get_accuracy(modelVIT, model_adver, transforms_data_first_to_vit, dataloader, method_name, adv_eps)
            print("{} ori - top 1: {} - top 5: {}".format(method_name, accuracy["ori"]["top 1"], accuracy["ori"]["top 5"]))
            print("{} adver - top 1: {} - top 5: {}".format(method_name, accuracy["adver"]["top 1"], accuracy["adver"]["top 5"]))

    except KeyboardInterrupt:
        print("Press Ctrl-C to terminate while statement")