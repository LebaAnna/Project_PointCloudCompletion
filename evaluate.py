import argparse
import torch

from dataset.data_loader import ShapeNet
from model import Folding
from loss import ChamferDistance


parser = argparse.ArgumentParser()
parser.add_argument('--partial_root', type=str, default='./dataset/test/partial')
parser.add_argument('--gt_root', type=str, default='./dataset/test/gt')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cd_loss = ChamferDistance()

test_dataset = ShapeNet(partial_path=args.partial_root, gt_path=args.gt_root, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

network = Folding()
network.load_state_dict(torch.load('/model/trained_model_fold.pth'))
network.to(DEVICE)

# Mean cd loss
network.eval()
with torch.no_grad():
    total_loss, iter_count = 0, 0
    for i, data in enumerate(test_dataloader, 1):
        partial_input, coarse_gt, dense_gt = data
        
        partial_input = partial_input.to(DEVICE)
        coarse_gt = coarse_gt.to(DEVICE)
        dense_gt = dense_gt.to(DEVICE)
        partial_input = partial_input.permute(0, 2, 1)
        
        v, y_coarse, y_detail = network(partial_input)

        y_detail = y_detail.permute(0, 2, 1)

        loss = cd_loss(dense_gt, y_detail)
        total_loss += loss.item()
        iter_count += 1
    partial_input, coarse_gt, dense_gt = test_dataset[random.randint(0, len(test_dataset))]
    partial_input = partial_input.to(DEVICE)
    coarse_gt = coarse_gt.to(DEVICE)
    dense_gt = dense_gt.to(DEVICE)
    partial_input = partial_input.permute(0, 2, 1)
           
    v, y_coarse, y_detail = network(partial_input)
          
    y_coarse = y_coarse.permute(0, 2, 1)
    y_detail = y_detail.permute(0, 2, 1)
    save_point_cloud(partial_input.numpy(), 'visual/partial_input.pcd')
    save_point_cloud(dense_gt.numpy())
    save_point_cloud(y_detail.numpy())

    mean_loss = total_loss / iter_count
    print("\nTesting loss is {}".format(mean_loss))
