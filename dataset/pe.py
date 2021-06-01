import logging
import math
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import pickle
from .randaugment import RandAugmentMC
from torch.utils.data import Dataset, DataLoader
#import albumentations
import pydicom
import cv2
import gdcm

logger = logging.getLogger(__name__)
cifar10_mean = (0.456, 0.456, 0.456) #(0.4914, 0.4822, 0.4465)
cifar10_std = (0.224, 0.224, 0.224)#(0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
image_size = 576

def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X

class PE_SSL(Dataset):
    def __init__(self, image_dict, bbox_dict, image_list, target_size, targets, transform=None):
        self.image_dict=image_dict
        self.bbox_dict=bbox_dict
        self.image_list=image_list###[:1000]
        self.target_size=target_size
        self.transform=transform
        #self.images=images
        self.targets=targets
    def __len__(self):
        return self.image_list.shape[0]
    def __getitem__(self,index):
        path='../../RSNA-STR-Pulmonary-Embolism-Detection/input/train/'
        study_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[0]
        series_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[1]
        data1 = pydicom.dcmread(path+study_id+'/'+series_id+'/'+self.image_dict[self.image_list[index]]['image_minus1']+'.dcm')
        data2 = pydicom.dcmread(path+study_id+'/'+series_id+'/'+self.image_list[index]+'.dcm')
        data3 = pydicom.dcmread(path+study_id+'/'+series_id+'/'+self.image_dict[self.image_list[index]]['image_plus1']+'.dcm')
        x1 = data1.pixel_array
        x2 = data2.pixel_array
        x3 = data3.pixel_array
        x1 = x1*data1.RescaleSlope+data1.RescaleIntercept
        x2 = x2*data2.RescaleSlope+data2.RescaleIntercept
        x3 = x3*data3.RescaleSlope+data3.RescaleIntercept
        x1 = np.expand_dims(window(x1, WL=100, WW=700), axis=2)
        x2 = np.expand_dims(window(x2, WL=100, WW=700), axis=2)
        x3 = np.expand_dims(window(x3, WL=100, WW=700), axis=2)
        x = np.concatenate([x1, x2, x3], axis=2)
        bbox = self.bbox_dict[self.image_dict[self.image_list[index]]['series_id']]
        x = x[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        x = cv2.resize(x, (self.target_size,self.target_size))
        img = Image.fromarray(x)
        if self.transform is not None:
            img = self.transform(img)
        # if self.transform is None:
        #     x = transforms.ToTensor()(x)
        #     x = transforms.Normalize(mean=[0.456, 0.456, 0.456], std=[0.224, 0.224, 0.224])(x)
        # else:
        #     x = self.transform(image=x)['image']
        #     x = x.transpose(2, 0, 1)
        y = self.image_dict[self.image_list[index]]['pe_present_on_image']
        y_true=self.targets[index]
        if y_true!=y:
            print('yyyyyy', y, y_true)
        return img, y


def get_data(args):
# prepare input
    
    path='../../RSNA-STR-Pulmonary-Embolism-Detection/trainval/process_input/split2/'
    with open(path +'image_list_train.pickle', 'rb') as f:
        image_list_train = pickle.load(f)#[:1000]
    with open(path+'image_dict.pickle', 'rb') as f:
        image_dict = pickle.load(f) 
    
    path2='../../RSNA-STR-Pulmonary-Embolism-Detection/trainval/lung_localization/split2/'
    with open(path2+'bbox_dict_train.pickle', 'rb') as f:
        bbox_dict_train = pickle.load(f) 
    print(len(image_list_train), len(image_dict), len(bbox_dict_train))
    print('AAAAAAAAAAAAA')
    with open(path+'image_list_valid.pickle', 'rb') as f:
        image_list_valid = pickle.load(f)#[:1000] 
    #with open('../process_input/split2/image_dict.pickle', 'rb') as f:
    #    image_dict = pickle.load(f) 
    with open(path2+'bbox_dict_valid.pickle', 'rb') as f:
        bbox_dict_valid = pickle.load(f)
    print(len(image_list_valid), len(image_dict), len(bbox_dict_valid))
    with open(path +'series_list_train.pickle', 'rb') as f:
        series_list_train = pickle.load(f)
    with open(path + 'series_list_valid.pickle', 'rb') as f:
        series_list_valid = pickle.load(f) 
    with open(path+'series_dict.pickle', 'rb') as f:
        series_dict = pickle.load(f)
    ###########
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])
    #base_dataset= PEDataset(image_dict=image_dict, bbox_dict=bbox_dict_train, image_list=image_list_train, target_size=image_size)#, transform=train_transform)
    ############
    targets, train_images, targets_ser=get_labels(series_list_train, series_dict, image_dict)
    print('ttt', len(targets), len(train_images))
 
    train_labeled_img, train_unlabeled_idxs, targets_labeled = x_u_split(args, targets, targets_ser,series_list_train, series_dict, image_dict)
    
    targets=np.array(targets)
    train_labeled_dataset = PE_SSL(image_dict=image_dict, bbox_dict=bbox_dict_train, image_list=train_labeled_img, target_size=image_size, targets=targets_labeled,transform=transform_labeled)
    train_unlabeled_dataset = PE_SSL(image_dict=image_dict, bbox_dict=bbox_dict_train, image_list=train_images[train_unlabeled_idxs], target_size=image_size, targets=targets[train_unlabeled_idxs],transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))
     #### cifar mean
    val_targets, val_images,_=get_labels(series_list_valid[0:300], series_dict, image_dict)
    test_dataset=PE_SSL(image_dict=image_dict, bbox_dict=bbox_dict_valid, image_list=val_images, target_size=image_size, targets=val_targets,transform=transform_val)
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_labels(series_list, series_dict, image_dict):
    gt_list=[]
    gt_ser=[]
    image_lst=[] 
    for n in tqdm(range(len(series_list))):
        gt_ser.append(series_dict[series_list[n]]['negative_exam_for_pe'])
        image_list_ser = series_dict[series_list[n]]['sorted_image_list']
        image_lst+=image_list_ser
        for m in range(len(image_list_ser)):
            gt_list.append(image_dict[image_list_ser[m]]['pe_present_on_image'])
    return gt_list, np.array(image_lst), gt_ser

def x_u_split(args, labels, labels_ser, series_list, series_dict, image_dict):
    args.num_classes=2 ###########
    label_per_class = args.num_labeled // args.num_classes
    #labels = np.array(labels)
    labels_ser = np.array(labels_ser)
    labeled_idx = []
    labeled_img_list=[]
    gt_labeled=[]
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels_ser == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled
    

   
    np.random.shuffle(labeled_idx)
    for j in labeled_idx:
            images=series_dict[series_list[j]]['sorted_image_list']
            labeled_img_list+=images
            for m in range(len(images)):
                gt_labeled.append(image_dict[images[m]]['pe_present_on_image'])

    #
    print('# of labels',len(labeled_img_list))
    labeled_img_list= np.array(labeled_img_list)
    gt_labeled=np.array(gt_labeled)
    print('sss', gt_labeled.shape, labeled_img_list.shape)
    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / (args.num_labeled*200))
        labeled_img_list= np.hstack([labeled_img_list for _ in range(num_expand_x)])
        gt_labeled=np.hstack([gt_labeled for _ in range(num_expand_x)])
    print('sss22', gt_labeled.shape, labeled_img_list.shape)
    return labeled_img_list, unlabeled_idx, gt_labeled


    
class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

        
class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


