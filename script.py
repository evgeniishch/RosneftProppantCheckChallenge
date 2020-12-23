import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from torch.utils.data import DataLoader, Dataset

import os
from os import walk
import random

from tqdm import tqdm
import numpy as np
import importlib

import RPCC_metric_utils_for_participants
importlib.reload(RPCC_metric_utils_for_participants)
sive_diam_pan = RPCC_metric_utils_for_participants.sive_diam_pan

import img_preproc
importlib.reload(img_preproc)

def set_seed(seed = 241199):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed()


class RosneftDataset(Dataset):
    def __init__(self, path, transforms):
        self.path = path
        self.files = os.listdir(path)
        self.transforms = transforms
        
    def __getitem__(self, item):
        path = os.path.join(self.path, self.files[item])
        img = cv2.imread(path)
        img = img_preproc.planarize(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)['image']
        img = torch.from_numpy(img)
        return img.permute(2, 0, 1)
    
    def __len__(self):
        return len(self.files)
    
    
ds = RosneftDataset(
    "./data/test/", 
    A.Compose([
        A.Normalize(),
        A.Resize(600, 1000),
    ]),
)
loader = DataLoader(ds, 32, shuffle=False)


f_names = []
for (dirpath, dirnames, filenames) in walk('./data/test/'):
    f_names.extend(filenames)
    break

clf = img_preproc.build_classifier()

def calc_part(names):
    part_outs = []
    for i in tqdm(names):
        path = os.path.join('./data/test/', i)
        img = cv2.imread(path)
        count = img_preproc.get_opencv_count(img, clf)
        part_outs.append(count)
    return part_outs


from multiprocessing import Pool
pool = Pool()

result1 = pool.apply_async(calc_part, [f_names[:round(len(f_names)*0.33)]])
result2 = pool.apply_async(calc_part, [f_names[round(len(f_names)*0.33):2*round(len(f_names)*0.33)]]) 
result3 = pool.apply_async(calc_part, [f_names[2*round(len(f_names)*0.33):]])

answer1 = result1.get()
answer2 = result2.get()
answer3 = result3.get()

outputs_cnts = answer1 + answer2 + answer3

outputs_cnts = [item if item is not None else -10 for item in outputs_cnts]
outputs_cnts = [item if item > 200 and item < 5000 else None for item in outputs_cnts]
ocv_mean = np.mean(outputs_cnts[outputs_cnts != None])
outputs_cnts = [item if item is not None else ocv_mean for item in outputs_cnts]


### DISTR
class DistrNet(nn.Module):
    def __init__(self, resnet):
        super(DistrNet, self).__init__()
        self.resnet = resnet
        self.fc1 = nn.Linear(1000, 1000)
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 20)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.6)
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.bn1(x)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = torch.load('./models/model_dists_v3.pth')
model.cuda().eval()

outputs_dists = []
with torch.no_grad():
    for batch in tqdm(loader):
        batch = batch.cuda()
        outputs_dists.extend(model(batch).softmax(dim=1).cpu().detach().numpy().squeeze().tolist())
        
        
def get_submit(cnt_preds, dist_preds, indices):
    submit = []
    for idx, cnt, dist in zip(indices, cnt_preds, dist_preds):
        cnt = int(cnt)
        sizes = np.random.choice(sive_diam_pan, size=cnt, p=dist / np.sum(dist))
        submit.extend([{
            "ImageId": idx,
            "prop_size": sizes[i]
        } for i in range(cnt)])
    return pd.DataFrame.from_records(submit)

train = pd.read_csv("./data/labels/train.csv")
train_cnt = train[~train.prop_count.isnull()]

max_cnt = train_cnt.prop_count.max()
min_cnt = train_cnt.prop_count.min()

norm = lambda cnt: (cnt - min_cnt) / (max_cnt - min_cnt)
inorm = lambda cnt: cnt * (max_cnt - min_cnt) + min_cnt

assert inorm(norm(1500)) == 1500

submission = get_submit(
    outputs_cnts,
    outputs_dists,
    map(lambda x: x.rstrip('.jpg'), loader.dataset.files),
)
submission.ImageId = submission.ImageId.astype(int)
submission.to_csv("answers.csv", index=False)

print("OK")
