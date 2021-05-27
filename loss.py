import torch
import torch.nn as nn

from distance.chamfer_distance import ChamferDistanceFunction



class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()
    
    def forward(self, pcs1, pcs2):
        dist1, dist2 =  ChamferDistanceFunction.apply(pcs1, pcs2)  # (B, N), (B, M)
        dist1 = torch.mean(torch.sqrt(dist1))
        dist2 = torch.mean(torch.sqrt(dist2))
        return (dist1 + dist2) / 2

if __name__ == '__main__':
    from utils import setup_seed
    setup_seed(20)

    pcs1 = torch.rand(10, 1024, 3)
    pcs2 = torch.rand(10, 1024, 3)

    cd_loss = ChamferDistance()
    print(cd_loss(pcs1, pcs2))

    
