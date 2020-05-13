import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from PIL import Image, ImageOps, ImageEnhance
import torchvision
import os
import tqdm
import numpy as np
import pandas as pd
import pickle

NUM_PTS = 971
INPUT_SIZE = (224,288)
SUBMISSION_HEADER = "file_name,Point_M0_X,Point_M0_Y,Point_M1_X,Point_M1_Y,Point_M2_X,Point_M2_Y,Point_M3_X,Point_M3_Y,Point_M4_X,Point_M4_Y,Point_M5_X,Point_M5_Y,Point_M6_X,Point_M6_Y,Point_M7_X,Point_M7_Y,Point_M8_X,Point_M8_Y,Point_M9_X,Point_M9_Y,Point_M10_X,Point_M10_Y,Point_M11_X,Point_M11_Y,Point_M12_X,Point_M12_Y,Point_M13_X,Point_M13_Y,Point_M14_X,Point_M14_Y,Point_M15_X,Point_M15_Y,Point_M16_X,Point_M16_Y,Point_M17_X,Point_M17_Y,Point_M18_X,Point_M18_Y,Point_M19_X,Point_M19_Y,Point_M20_X,Point_M20_Y,Point_M21_X,Point_M21_Y,Point_M22_X,Point_M22_Y,Point_M23_X,Point_M23_Y,Point_M24_X,Point_M24_Y,Point_M25_X,Point_M25_Y,Point_M26_X,Point_M26_Y,Point_M27_X,Point_M27_Y,Point_M28_X,Point_M28_Y,Point_M29_X,Point_M29_Y\n"


class ThousandLandmarksDataset(data.Dataset):
    def __init__(self, root, transforms, split="train", train_size=0.8):
        super(ThousandLandmarksDataset, self).__init__()
        assert(split in ("train", "val", "test"))
        root = os.path.join(root, "test" if split == "test"  else "train")
        self.root = root
        landmark_file_name = os.path.join(root, 'landmarks.csv') if split != "test" \
            else os.path.join(root, "test_points.csv")
        images_root = os.path.join(root, "images")
        self.images_root = images_root

        self.image_names = []
        self.landmarks = []

        print(f"Preparing {split} dataset...")
        loaded = False
        try:
            assert(split in ["train", "val"])
            with open(os.path.join(root, "landmarks.pickle"), 'rb') as fp:
                self.image_names, self.landmarks = pickle.load(fp)
            loaded = True
        except:  
            pass

        if not loaded:
            df = pd.read_csv(landmark_file_name, delimiter='\t')
            self.image_names = np.array(df.iloc[:,0])
            if split in ["train", "val"]:
                self.landmarks = np.array(df.iloc[:,1:]).astype(np.int16)
                self.landmarks = self.landmarks.reshape(-1, self.landmarks.shape[1]//2, 2)
                try:
                    with open(os.path.join(root, "landmarks.pickle"), 'wb') as fp:
                        pickle.dump((self.image_names, self.landmarks), fp)
                except:
                    pass
            else:
                self.landmarks = None
            

            if split == "train":
                self.image_names = self.image_names[:int(len(self.image_names)*0.8)]
                self.landmarks = self.landmarks[:int(len(self.landmarks)*0.8)]
            if split == "val":
                self.image_names = self.image_names[int(len(self.image_names)*0.8):]
                self.landmarks = self.landmarks[int(len(self.landmarks)*0.8):]
        print("Done!")

        self.transforms = transforms

    def __getitem__(self, idx):
        sample = {}
        if self.landmarks is not None:
            landmarks = self.landmarks[idx]
            sample["landmarks"] = landmarks

        #image = cv2.imread(self.image_names[idx])
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_name = os.path.join(self.images_root, self.image_names[idx])
        image = Image.open(image_name)
        sample["image"] = image

        if self.transforms is not None:
            sample = self.transforms(sample)

        sample['image'] = torchvision.transforms.functional.to_tensor(sample['image'])

        return sample

    def __len__(self):
        return len(self.image_names)


def recover_landmarks(sample, prediction):
    landmarks = prediction
    landmarks = landmarks - sample['paste_position'][:,None,:]
    landmarks = landmarks * sample['orig_size'][:,None,:] / sample['new_size'][:,None,:]
    return landmarks
 
def validate(model, dataloader, batch_size=192):
    model.eval()
    val_loss = []
    for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc="validation..."):
        images = batch["image"].cuda()
        landmarks = batch["landmarks"].cuda()
        batch['orig_size'] = batch['orig_size'].cuda()
        batch['paste_position'] = batch['paste_position'].cuda()
        batch['new_size'] = batch['new_size'].cuda()

        with torch.no_grad():
            pred_landmarks = model(images)
            pred_landmarks[...,0] = pred_landmarks[...,0]*INPUT_SIZE[0]
            pred_landmarks[...,1] = pred_landmarks[...,1]*INPUT_SIZE[1]

        pred_landmarks = pred_landmarks
        pred_landmarks = recover_landmarks(batch, pred_landmarks)

        loss = F.mse_loss(pred_landmarks, landmarks, reduction='mean')
        val_loss.append(loss.item())

    return np.mean(val_loss)

##################################################
#                Augumentations                  #
##################################################

class PasteToSize():
    def __init__(self, target_size, train=True):
        self.target_size = target_size
        self.train = train
    def __call__(self, sample):
        image = sample['image']
        if 'landmarks' in sample:
            landmarks = sample['landmarks']
        else:
            landmarks = None

        if self.train:
            min_bbox = np.concatenate([landmarks.min(axis=0), landmarks.max(axis=0)])
            
            min_bbox_scale = max((min_bbox[2] - min_bbox[0])/self.target_size[0], (min_bbox[3] - min_bbox[1])/self.target_size[1])
            new_scale = 0.6*np.random.random() + 0.3
            new_size = (int(image.size[0]*new_scale/min_bbox_scale), int(image.size[1]*new_scale/min_bbox_scale)) 

            image = image.resize(new_size)
            landmarks = landmarks*new_scale/min_bbox_scale
            min_bbox = min_bbox*new_scale/min_bbox_scale

            llc_range = (
                                                int(0.05*self.target_size[0] - min_bbox[0]),
                                                int(0.05*self.target_size[1] - min_bbox[1]),
                                                int(0.95*self.target_size[0] - min_bbox[2]),
                                                int(0.95*self.target_size[1] - min_bbox[3]),
                                               )

            new_image = Image.new('RGB', self.target_size)
            paste_position = (np.random.randint(llc_range[0], llc_range[2] + 1), np.random.randint(llc_range[1],llc_range[3] + 1))
            new_image.paste(image, paste_position)
            landmarks[:,0] = landmarks[:,0] + paste_position[0]
            landmarks[:,1] = landmarks[:,1] + paste_position[1]
            landmarks = torch.FloatTensor(landmarks)
            sample = {'image': new_image, 'landmarks': landmarks}
        else:
            size = image.size
            scale = max(image.size[0]/self.target_size[0], image.size[1]/self.target_size[1]) 
            new_scale = 1.
            new_size = (int(image.size[0]*new_scale/scale), int(image.size[1]*new_scale/scale))
            image = image.resize(new_size)
            new_image = Image.new('RGB', self.target_size)
            paste_position = (int((self.target_size[0] - image.size[0])/2), int((self.target_size[1] - image.size[1])/2))
            new_image.paste(image, paste_position)

            size = torch.LongTensor(size)
            new_size = torch.LongTensor(new_size)
            paste_position = torch.LongTensor(paste_position)
    
            sample['image'] = new_image
            sample['orig_size'] = size
            sample['new_size'] = new_size
            sample['paste_position'] = paste_position

        return sample
             


def Identity(sample, *args, **kwargs):
    return sample

def AutoContrast(sample, *args, **kwargs):
    sample['image'] = ImageOps.autocontrast(sample['image'])
    return sample

def Equalize(sample, *args, **kwargs):
    sample['image'] = ImageOps.equalize(sample['image'])
    return sample

def Rotate(sample, mag, *args, **kwargs):
    image = sample['image']
    landmarks = sample['landmarks']

    angle = mag*(2*np.random.random() - 1)*np.pi/4
    cs = np.cos(angle)
    sn = np.sin(angle)
    b_vec = (image.size[0]*(1 - cs)/2 + image.size[1]*sn/2, image.size[1]*(1 - cs)/2 - image.size[0]*sn/2)
    image = image.transform(image.size, Image.AFFINE, (cs, -sn, b_vec[0], sn, cs, b_vec[1]), Image.BILINEAR)

    new_landmarks = np.zeros_like(landmarks)
    new_landmarks[..., 0] = (landmarks[..., 0] - b_vec[0])*cs + (landmarks[..., 1] - b_vec[1])*sn
    new_landmarks[..., 1] = (landmarks[..., 1] - b_vec[1])*cs - (landmarks[..., 0] - b_vec[0])*sn
    landmarks = new_landmarks
    return {'image': image, 'landmarks': landmarks}

def Solarize(sample, *args, **kwargs):
    sample['image'] = ImageOps.solarize(sample['image'], 0)
    return sample

def Color(sample, mag, *args, **kwargs):
    sample['image'] = ImageEnhance.Color(sample['image']).enhance(0.1 + 1.8*mag)
    return sample

def Posterize(sample, mag, *args, **kwargs):
    sample['image'] = ImageOps.posterize(sample['image'], 8 - np.random.binomial(4, mag))
    return sample

def Contrast(sample, mag, *args, **kwargs):
    if np.random.random() > 0.5:
        mag = -mag
    sample['image'] = ImageEnhance.Contrast(sample['image']).enhance(1 + mag)
    return sample

def Brightness(sample, mag, *args, **kwargs):
    if np.random.random() > 0.5:
        mag = -mag
    sample['image'] = ImageEnhance.Brightness(sample['image']).enhance(1 + mag)
    return sample
    
def Sharpness(sample, mag, *args, **kwargs):
    if np.random.random() > 0.5:
        mag = -mag
    sample['image'] = ImageEnhance.Sharpness(sample['image']).enhance(1 + mag)
    return sample

def ShearX(sample, mag, *args, **kwargs):
    image = sample['image']
    landmarks = sample['landmarks']

    par = (np.random.random() - 1)*mag
    b = -par*image.size[1]/2
    image = image.transform(image.size, Image.AFFINE, (1, par, b, 0, 1, 0), Image.BILINEAR)

    new_landmarks = landmarks.copy()
    new_landmarks[...,0] = landmarks[...,0] - b - par*(landmarks[...,1])
    landmarks = new_landmarks
    return {'image': image, 'landmarks': landmarks}

def ShearY(sample, mag, *args, **kwargs):
    image = sample['image']
    landmarks = sample['landmarks']

    par = (np.random.random() - 1)*mag
    b = -par*image.size[0]/2
    image = image.transform(image.size, Image.AFFINE, (1, 0, 0, par, 1, b), Image.BILINEAR)
    new_landmarks = landmarks.copy()
    new_landmarks[...,1] = landmarks[...,1] - b - par*(landmarks[...,0])
    landmarks = new_landmarks
    return {'image': image, 'landmarks': landmarks}

def DropBlock(sample, mag, *args, **kwargs):
    image = sample['image']
    landmarks = sample['landmarks']

    min_bbox = np.concatenate([landmarks.min(axis=0), landmarks.max(axis=0)])
    width = (min_bbox[2] - min_bbox[0])*(0.5 + 0.5*np.random.random())*mag
    width = int(width)
    height = (min_bbox[3] - min_bbox[1])*(0.5 + 0.5*np.random.random())*mag
    height = int(height)
    paste_position = min_bbox[:2] + (min_bbox[2:] - np.array([width, height]))*np.random.random()
    paste_position = paste_position.astype(np.int)
    paste_position[1] = image.size[1] - paste_position[1]

    image.paste(Image.new('RGB', (width, height)), tuple(paste_position))
    sample['image'] = image
    return sample


aug_list = [Identity, AutoContrast, Equalize, Rotate, Color, Posterize,
            Contrast, Brightness, Sharpness, ShearX, ShearY, DropBlock]


class RandAug():
    def __init__(self, n_augs, magnitude):
        self.n_augs = n_augs
        self.magnitude = magnitude

    def __call__(self, sample):
        ops = np.random.choice(aug_list, self.n_augs)
        for op in ops:
            sample = op(sample, self.magnitude)
        return sample

class Transforms():
    def __init__(self, target_size, n_augs, magnitude):
        self.randaug = RandAug(n_augs, magnitude)
        self.paste_to_size = PasteToSize(target_size)
        
    def __call__(self, sample):
        sample = self.randaug(sample)
        sample = self.paste_to_size(sample)
        return sample

