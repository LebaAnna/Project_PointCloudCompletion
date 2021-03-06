import argparse

import torch
import numpy
import torch.optim as optim

from dataset.data_loader import ShapeNet
from model import Folding
from loss import ChamferDistance
from utils import save_point_cloud

parser = argparse.ArgumentParser()
parser.add_argument('--partial_root', type=str, default='./dataset/train/partial')
parser.add_argument('--gt_root', type=str, default='./dataset/train/gt')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--num_input', type=int, default=2048)
parser.add_argument('--num_coarse', type=int, default=1024)
parser.add_argument('--num_dense', type=int, default=16384)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--loss_d1', type=str, default='cd')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cd_loss = ChamferDistance()
loss_d1 = cd_loss 


train_dataset = ShapeNet(partial_path=args.partial_root, gt_path=args.gt_root, split='train')
val_dataset = ShapeNet(partial_path=args.partial_root, gt_path=args.gt_root, split='val')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

network = Folding()
if args.model is not None:
    print('Loaded trained model from {}.'.format(args.model))
    network.load_state_dict(torch.load(args.model))
else:
    print('Begin training new model.')
network.to(DEVICE)
optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

max_iter = int(len(train_dataset) / args.batch_size + 0.5)
minimum_loss = 1e4
best_epoch = 0

for epoch in range(1, args.epochs + 1):
    # training
    network.train()
    total_loss, iter_count = 0, 0
    for i, data in enumerate(train_dataloader, 1):
        partial_input, coarse_gt, dense_gt = data
        partial_input = partial_input.to(DEVICE)
        coarse_gt = coarse_gt.to(DEVICE)
        dense_gt = dense_gt.to(DEVICE)
        partial_input = partial_input.permute(0, 2, 1)

        optimizer.zero_grad()

        v, y_coarse, y_detail = network(partial_input)

        y_coarse = y_coarse.permute(0, 2, 1)
        y_detail = y_detail.permute(0, 2, 1)

        loss = loss_d1(coarse_gt, y_coarse) + args.alpha * loss_d1(dense_gt, y_detail)
        loss.backward()
        optimizer.step()
        
        iter_count += 1
        total_loss += loss.item()
              
    print("Training epoch {}/{}: avg loss = {}".format(epoch, args.epochs, total_loss / iter_count))

    # evaluation
    network.eval()
    with torch.no_grad():
        total_loss, iter_count = 0, 0
        for i, data in enumerate(val_dataloader, 1):
            partial_input, coarse_gt, dense_gt = data
            
            partial_input = partial_input.to(DEVICE)
            coarse_gt = coarse_gt.to(DEVICE)
            dense_gt = dense_gt.to(DEVICE)
            partial_input = partial_input.permute(0, 2, 1)
            
            v, y_coarse, y_detail = network(partial_input)

            y_coarse = y_coarse.permute(0, 2, 1)
            y_detail = y_detail.permute(0, 2, 1)
                                    
            loss = loss_d1(coarse_gt, y_coarse) + args.alpha * loss_d1(dense_gt, y_detail)
            total_loss += loss.item()
            iter_count += 1

        mean_loss = total_loss / iter_count
        print("Validation epoch {}/{}, loss is {}".format(epoch, args.epochs, mean_loss))

        # Save model
        if mean_loss < minimum_loss:
            partial_input, coarse_gt, dense_gt = val_dataset[random.randint(0, len(val_dataset))]
            partial_input = partial_input.to(DEVICE)
            coarse_gt = coarse_gt.to(DEVICE)
            dense_gt = dense_gt.to(DEVICE)
            partial_input = partial_input.permute(0, 2, 1)
            
            v, y_coarse, y_detail = network(partial_input)
            
            y_coarse = y_coarse.permute(0, 2, 1)
            y_detail = y_detail.permute(0, 2, 1)
            save_point_cloud(partial_input.numpy(), 'visual/partial_input.pcd')
            save_point_cloud(dense_gt.numpy(), 'visual/gt.pcd')
            save_point_cloud(y_detail.numpy(), 'visual/output.pcd')
            best_epoch = epoch
            minimum_loss = mean_loss
            torch.save(network.state_dict(), + 'model/trained_model_fold.pth')
            

   

