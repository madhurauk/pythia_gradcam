import sys
sys.path.append('pythia')
sys.path.append('purva-vqa-maskrcnn-benchmark')
print("in pythia_dataset: pythia and purva-vqa-maskrcnn-benchmark appended to sys path")

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from vqaTools.vqa import VQA

class VQA_Dataset():
    def __init__(self, evalIds, annFile, quesFile, imgDir, dataSubType, target_image_size, channel_mean, channel_std, splits, subset, checkpoint):
        # initialize VQA api for QA annotations
        self.vqa = VQA(annFile, quesFile)
        self.annIds = self.vqa.getQuesIds()
        #self.annIds = self._get_evaluated_ids(evalIds)
        self.splits = splits
        self.subset = subset
        self.checkpoint = checkpoint
        self.subset_size = int(len(self.annIds) / self.splits)
        self.imgDir = imgDir
        self.dataSubType = dataSubType
        self.target_image_size = target_image_size
        self.channel_mean = channel_mean
        self.channel_std = channel_std

    def __len__(self):
        if self.subset < self.splits - 1:
            return self.subset_size - self.checkpoint
        else:
            left_over = len(self.annIds) - (self.subset_size * self.splits)
            return self.subset_size + left_over - self.checkpoint

    def _image_transform(self, img):
        im = np.array(img).astype(np.float32)
        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(800) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > 1333:
            im_scale = float(1333) / float(im_size_max)
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)
        return img, im_scale

    def __getitem__(self, index):
        print("in pythia_dataset gtitem: index: ",index)
        #print("self.annIds: ", self.annIds)
        index = self.subset * self.subset_size + index + self.checkpoint
        annId = int(self.annIds[index])
        #annId = index
        ann = self.vqa.loadQA(annId)[0]
        imgId = ann['image_id']
        imgFilename = 'COCO_' + self.dataSubType + '_' + str(imgId).zfill(12) + '.jpg'
        question = self.vqa.getQuestion(ann)
        image_path = self.imgDir + imgFilename
        img = Image.open(image_path)
        raw_image = cv2.imread(image_path)
        resnet_img = img.convert("RGB")
        data_transforms = transforms.Compose([
            transforms.Resize(self.target_image_size),
            transforms.ToTensor(),
            transforms.Normalize(self.channel_mean, self.channel_std),
        ])
        resnet_img = data_transforms(resnet_img)
        if len(np.shape(img)) == 2:
            img = img.convert("RGB")
        detectron_img, detectron_scale = self._image_transform(img)
        return {"annId": annId, "question": question, "resnet_img": resnet_img, "detectron_img": detectron_img, "detectron_scale": detectron_scale, "raw_image": raw_image}

    def _get_evaluated_ids(self, path):
        # save np.load
        np_load_old = np.load
        # modify the default parameters of np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        # call load_data with allow_pickle implicitly set to true
        ids = np.load(path)
        np.load = np_load_old
        #ids = np.squeeze(ids[:, :1].astype(np.int32))
        return ids
