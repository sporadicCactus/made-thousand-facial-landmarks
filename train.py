import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
import numpy as np
import os
from argparse import ArgumentParser
import tqdm

from utils import ThousandLandmarksDataset, Transforms, INPUT_SIZE, PasteToSize, validate
from models import Resnet50_PANet

try:
    import apex
    apex_available = True
except:
    apex_available = False


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", required=True)
    parser.add_argument("--resume", "-r", default=None)
    parser.add_argument("--data", "-d", required=True)
    parser.add_argument("--batch-size", "-b", default=64, type=int)
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float)
    parser.add_argument("--anneals", "-an", default=1, type=int)
    return parser.parse_args()


args = parse_arguments()

train_dataset = ThousandLandmarksDataset(args.data, Transforms(INPUT_SIZE, 3, .2), split="train")
train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True,
                                   shuffle=True, drop_last=True)
val_dataset = ThousandLandmarksDataset(args.data, PasteToSize(INPUT_SIZE, train=False), split="val")
val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True,
                                 shuffle=False, drop_last=False)


model = Resnet50_PANet().cuda()

if args.resume is not None:
    model.load_state_dict(torch.load(args.resume))


opt = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum=0.9, weight_decay=1e-5)

if apex_available:
    model, opt = apex.amp.initialize(model, opt, opt_level='O2')

total_its = args.epochs*len(train_dataloader)
anneal_threshold = total_its/(2**args.anneals - 1)
anneal_counter = 0
anneals_done = 0
for n_ep in range(args.epochs):
    for it_num, batch in enumerate(train_dataloader):
        anneal_counter += 1
        if anneal_counter > anneal_threshold:
            anneal_threshold = anneal_threshold*2
            anneal_counter = 0 

            val_loss = validate(model, val_dataloader, args.batch_size)
            print(f"val_loss: {val_loss:.3f}")
            torch.save(model.state_dict(), f'./{args.name}_val_{val_loss:.3f}.pth')
            anneals_done += 1

        if anneals_done >= args.anneals:
            break
        
        model = model.train()
        lr = args.learning_rate * np.cos(0.5*np.pi*anneal_counter/anneal_threshold)
        opt.param_groups[0]['lr'] = lr
        inputs = batch['image'].cuda()
        targets = batch['landmarks'].cuda()
        
        outputs = model(inputs)
        outputs[...,0] = outputs[...,0]*INPUT_SIZE[0]
        outputs[...,1] = outputs[...,1]*INPUT_SIZE[1]

        loss = F.smooth_l1_loss(outputs*0.1, targets*0.1)/0.1
        print(f"Epoch: {n_ep} || it: {it_num}/{len(train_dataloader)} || lr: {lr:.8f} || train_loss: {loss.item():.5f}")

        opt.zero_grad()
        if apex_available:
            with apex.amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        opt.step()

val_loss = validate(model, val_dataloader)
print(f"val_loss: {val_loss:.3f}")
torch.save(model.state_dict(), f'./{args.name}_val_{val_loss:.3f}.pth')
    
