# -*- coding: utf-8 -*-
"""
https://drive.google.com/drive/folders/1VqhHYcYJCO338f3X-vHaBy-LKXxFEZf-?usp=sharing


Feito por Julio César Domingues dos Santos

"""

import json
import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as T
import os
import xml.etree.ElementTree as ET
from PIL import Image,ImageDraw
from IPython.display import display
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

os.chdir('/content/drive/MyDrive/datasets/object_detection')
from mapeval import voc_eval

img = Image.open('imagens/IMG_20210623_114259.jpg')

img

# dados rotulados de https://www.makesense.ai/
h_files = {}
with open('labels.csv') as csv_file:
    for line in csv_file:
        v = line.split(',')
        class_att = v[0]
        x1,y1,x2,y2 = [int(x) for x in v[1:5]]
        img_file = v[5]
        if img_file not in h_files.keys():
            h_files[img_file] = {'boxes':[],'labels':[]}
        h_files[img_file]['boxes'].append((x1,y1,x2,y2)) 
        h_files[img_file]['labels'].append(class_att)

for img_file in h_files.keys():
    print(img_file)
    print(h_files[img_file]['boxes'])
    print(h_files[img_file]['labels'])

class MyDataset():
    def __init__(self,transforms = None):
        self.data = []
        self.transforms = transforms
        self.target_names = ['blanck']
        self.htarget_names = {'blanck':0,'calculadora':1,'carteira':2,'controle':3,'grafite':4,'lapiseira':5,'oculos':6,'moeda':7}
        self.read_csv()

    def get_label_id(self,name):
        if name not in self.htarget_names:
            self.htarget_names[name] = len(self.target_names)
            self.target_names.append(name)
        return self.htarget_names[name]
    
    def read_csv(self):
        h_files = {}
        with open('labels.csv') as csv_file:
            for line in csv_file:
                v = line.split(',')
                class_att = v[0]
                x1,y1,width,height = [int(x) for x in v[1:5]]
                x2 = x1 + width
                y2 = y1 + height
                img_file = v[5]
                if img_file not in h_files.keys():
                    h_files[img_file] = {'boxes':[],'labels':[]}
                h_files[img_file]['boxes'].append((x1,y1,x2,y2)) 
                h_files[img_file]['labels'].append(self.get_label_id(class_att))

        for img_file in h_files.keys():
            h = {}
            h['file_img'] = 'imagens/'+img_file
            h['labels'] = h_files[img_file]['labels']
            h['boxes']  = h_files[img_file]['boxes']
            self.data.append(h)


    def read_json(self):
        with open('labels.json') as json_file:
            data = json.load(json_file)
            for h in data:
                if 'objects' in h['Label']:
                    file_url = h['Labeled Data']
                    file_name = 'imagens'+os.sep+h['External ID']
                    objects = h['Label']['objects']
                    boxes = []
                    labels = []
                    for obj in objects:
                        labels.append(self.get_label_id(obj['value']))
                        x1 = int(obj['bbox']['left'])
                        y1 = int(obj['bbox']['top'])
                        x2 = x1 + int(obj['bbox']['width'])
                        y2 = y1 + int(obj['bbox']['height'])
                        bbox  = [x1,y1,x2,y2]
                        boxes.append(bbox)
                    h = {}
                    h['file_img'] = file_name
                    h['labels'] = labels
                    h['boxes']  = boxes
                    self.data.append(h)
    def __getitem__(self,i):
        img   = Image.open(self.data[i]['file_img']).convert("RGB")
        boxes = torch.tensor(self.data[i]['boxes'])
        if self.transforms != None:
            img,boxes = self.transforms(img,boxes)
        r = dict()
        r['boxes']   = boxes
        r['labels']  = torch.tensor(self.data[i]['labels'])
        return img,r
    def __len__(self):
        return len(self.data)

def resize(img,boxes,size):
    w, h = img.size
    ow, oh = size
    sw = float(ow) / w
    sh = float(oh) / h
    img = img.resize((ow,oh), Image.BILINEAR)
    boxes = boxes * torch.tensor([sw,sh,sw,sh])
    return img, boxes

size = (300,300)
def transform_data(img,boxes):
    img,boxes = resize(img,boxes,size)
    img = T.Compose([
          T.ToTensor(), 
          T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))          
    ])(img)
    return img,boxes

data = MyDataset(transforms=transform_data)

n = len(data)
n_treino = int(0.7*n)
n_teste  = int(0.15*n)
n_val    = int(0.15*n)+1

n,n_treino,n_teste,n_val

ds_treino,ds_teste,ds_val = torch.utils.data.random_split(data,(n_treino,n_teste,n_val))

def collate_fn(batch):
    return tuple(zip(*batch))
dl_treino = torch.utils.data.DataLoader(ds_treino,batch_size = 8,collate_fn=collate_fn)
dl_teste  = torch.utils.data.DataLoader(ds_teste,batch_size = 12,collate_fn=collate_fn)
dl_val    = torch.utils.data.DataLoader(ds_val,batch_size = 12,collate_fn=collate_fn)

imgs,targets = next(iter(dl_treino))

len(imgs)

nview = Image.open(data.data[0]['file_img'])

data.data[0]['boxes']

def draw_boxes(img,boxes,labels):
    imdraw = ImageDraw.Draw(img)
    for (box,label) in zip(boxes,labels):
        box = list(box)
        imdraw.rectangle(box,outline='red')
        text = "%d"%(label)
        imdraw.text((box[0],box[1]),text,fill='red')
    display(img)

draw_boxes(nview,data.data[0]['boxes'],data.data[0]['labels'])

nview = T.ToPILImage()(imgs[0]*torch.Tensor([0.229,0.224,0.225]).view(3,1,1)+torch.Tensor([0.485,0.456,0.406]).view(3,1,1))

draw_boxes(nview,targets[0]['boxes'],targets[0]['labels'])

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Fine tunning
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,3)

opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

device

model.to(device)

def train(epoch):
    model.train()
    bloss=[]
    for images,targets in dl_treino:
        images = list(image.to(device) for image in images)
        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
        
        loss_dict = model(images,targets)
        losses = sum(loss for loss in loss_dict.values())
        
        opt.zero_grad()
        losses.backward()
        opt.step()
        #print(loss_dict)
        #wlog = {}
        for loss in loss_dict.keys():
            print("%.10s %4.3f"%(loss,loss_dict[loss].item()))
        print("Total Loss %4.3f\n"%(losses))
        bloss.append(losses.item())
    
    print("\nEPOCH %d LR %5.5f\n"%(epoch,opt.param_groups[0]['lr']))

def evaluate(epoch):
    model.eval()
    pred_boxes = []
    pred_labels = []
    pred_scores = []
    gt_boxes = []
    gt_labels = []
    lmap = []
    lap  = []
    with torch.no_grad():
        for images,targets in dl_teste:
            images = list(image.to(device) for image in images)
            pred   = model(images)
            for i in range(len(targets)):
                gt_boxes.append(targets[i]['boxes'])
                gt_labels.append(targets[i]['labels'])
                pred_boxes.append(pred[i]['boxes'].cpu())
                pred_labels.append(pred[i]['labels'].cpu())
                pred_scores.append(pred[i]['scores'].cpu())
                r = voc_eval(pred_boxes, pred_labels, pred_scores,
                gt_boxes, gt_labels)
                print(r)
                lmap.append(r['map'])
                #lap.append(r['ap'])
    print(np.mean(lmap))
    #print(np.mean(lap,axis=0))
    return np.mean(lmap)

best_map = 0.0

for epoch in range(100):
    train(epoch)
    map = evaluate(epoch)
    lr_scheduler.step(1.0-map)
    if map > best_map:
        best_map = map
        torch.save(model,'best_map_labelbox.pth')
        print('saving model')

nview = T.ToPILImage()(imgs[-1]*torch.Tensor([0.229,0.224,0.225]).view(3,1,1)+torch.Tensor([0.485,0.456,0.406]).view(3,1,1))

model.eval()

dl_val       = torch.utils.data.DataLoader(ds_val,batch_size=30,collate_fn=collate_fn)
imgs,targets = next(iter(dl_val))

dic   = {}
for i in imgs:
  pred = model(i.view([1,3,300,300]).to(device))
  cont  = 0
  for j in pred[0]['scores']:
    cont += j
  cont = cont/len(pred[0]['scores'])

  dic.update({float(cont):i})

  nview = T.ToPILImage()(i*torch.Tensor([0.229,0.224,0.225]).view(3,1,1)+torch.Tensor([0.485,0.456,0.406]).view(3,1,1))
  model.eval()

items = dic.items()
sorted_items = sorted(items)

best  = []
worse = []
tam = len(sorted_items)
for i in range(0,10):
  worse.append(sorted_items[i][1])
for i in range((tam-10),tam-1):
  best.append(sorted_items[i][1])

print("---------10 PIORES--------------")
for i in worse:
  pred = model(i.view([1,3,300,300]).to(device))
  nview = T.ToPILImage()(i*torch.Tensor([0.229,0.224,0.225]).view(3,1,1)+torch.Tensor([0.485,0.456,0.406]).view(3,1,1))
  model.eval()
  draw_boxes(nview,pred[0]['boxes'],pred[0]['labels'])

print("---------10 MELHORES--------------")
for i in best:
  pred = model(i.view([1,3,300,300]).to(device))
  nview = T.ToPILImage()(i*torch.Tensor([0.229,0.224,0.225]).view(3,1,1)+torch.Tensor([0.485,0.456,0.406]).view(3,1,1))
  model.eval()
  draw_boxes(nview,pred[0]['boxes'],pred[0]['labels'])