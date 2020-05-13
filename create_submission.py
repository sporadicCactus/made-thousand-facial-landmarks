import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data

import numpy as np
import pandas as pd

from models import Resnet50_PANet
from utils import ThousandLandmarksDataset, PasteToSize, NUM_PTS, INPUT_SIZE, SUBMISSION_HEADER

from argparse import ArgumentParser
import os
import tqdm

def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", required=True)
    parser.add_argument("--weights", "-r", required=True)
    parser.add_argument("--data", "-d", required=True)
    parser.add_argument("--batch-size", "-b", default=64, type=int)
    return parser.parse_args()

def recover_landmarks(sample, prediction):
    landmarks = prediction
    landmarks = landmarks - sample['paste_position'][:,None,:]
    landmarks = landmarks * sample['orig_size'][:,None,:] / sample['new_size'][:,None,:]
    return landmarks

args = parse_arguments()

model = Resnet50_PANet().cuda()

model.load_state_dict(torch.load(args.weights))
model = model.eval().half()


test_dataset = ThousandLandmarksDataset(args.data, PasteToSize(INPUT_SIZE, train=False), split="test")
test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=False, drop_last=False)


test_dir = os.path.join(args.data, "test")

output_file = os.path.join(f".",f"{args.name}.csv")
wf = open(output_file, 'w')
wf.write(SUBMISSION_HEADER)

mapping_path = os.path.join(test_dir, 'test_points.csv')
mapping = pd.read_csv(mapping_path, delimiter='\t')

test_predictions = []
for batch in tqdm.tqdm(test_dataloader, total=len(test_dataloader), desc="predicting..."):
    inputs = batch["image"].half().cuda()
    with torch.no_grad():
        predictions = model(inputs)
    predictions[...,0] *= INPUT_SIZE[0]
    predictions[...,1] *= INPUT_SIZE[1]

    landmarks = recover_landmarks(batch, predictions.cpu().float())
    test_predictions.append(landmarks)
test_predictions = torch.cat(test_predictions, dim=0)

for i, row in mapping.iterrows():
    file_name = row[0]
    point_index_list = np.array(eval(row[1]))
    points_for_image = test_predictions[i]
    needed_points = points_for_image[point_index_list].numpy().astype(np.int)
    wf.write(file_name + ',' + ','.join(map(str, needed_points.reshape(2 * len(point_index_list)))) + '\n')
