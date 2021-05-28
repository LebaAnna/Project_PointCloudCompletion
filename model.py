import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # first shared mlp
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

        # second shared mlp
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)
    
    def forward(self, x):
        n = x.size()[2]

        # first shared mlp
        x = F.relu(self.bn1(self.conv1(x)))           
        f = self.bn2(self.conv2(x))                   
        
        # point-wise maxpool
        g = torch.max(f, dim=2, keepdim=True)[0]      
        
        # expand and concat
        x = torch.cat([g.repeat(1, 1, n), f], dim=1)  

        # second shared mlp
        x = F.relu(self.bn3(self.conv3(x)))           
        x = self.bn4(self.conv4(x))                   
        
        # point-wise maxpool
        v = torch.max(x, dim=-1)[0]                   
        
        return v


class Decoder(nn.Module):
    def __init__(self, num_coarse=1024, num_dense=16384):
        super(Decoder, self).__init__()

        self.num_coarse = num_coarse
        
        # fully connected layers
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 3 * num_coarse)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)

        # shared mlp
        self.conv1 = nn.Conv1d(3+2+1024, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(512)
        
    def forward(self, x):
        b = x.size()[0]
        # global features
        v = x  

        # fully connected layers to generate the coarse output
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = F.relu(self.bn3(self.conv1(x)))
        x = F.relu(self.bn4(self.conv2(x)))
        x = self.conv3(x)               
        
        return x
    
class Enc_fold(nn.Module):
    def __init__(self):
        super(Enc_fold, self).__init__()
        self.feat = PointNetfeat(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans = self.feat(x)  # x = batch,1024
        x = F.relu(self.bn1(self.fc1(x)))  # x = batch,512
        x = self.fc2(x)  # x = batch,512

        return x, trans
    
class Dec_fold(nn.Module):
    def __init__(self):
        super(Dec_fold, self).__init__()
        self.fold1 = DecFold1()
        self.fold2 = DecFold2()

    def forward(self, x):  
        batch_size = x.size(0)
        x = torch.unsqueeze(x, 1) 
        x = x.repeat(1, 45 ** 2, 1) 
        code = x
        code = x.transpose(2, 1)  

        meshgrid = [[-0.3, 0.3, 45], [-0.3, 0.3, 45]]
        grid = GridSamplingLayer(batch_size, meshgrid)  
        grid = torch.from_numpy(grid)

        if x.is_cuda:
            grid = grid.cuda()

        x = torch.cat((x, grid), 2)  
        x = x.transpose(2, 1)  

        x = self.fold1(x) 
        p1 = x  

        x = torch.cat((code, x), 1)  

        x = self.fold2(x)  

        return x
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # x = batch,1024,n(n=2048)
        x = torch.max(x, 2, keepdim=True)[0]  # x = batch,1024,1
        x = x.view(-1, 1024)  # x = batch,1024
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans

class DecFold1(nn.Module):
    def __init__(self):
        super(FoldingNetDecFold1, self).__init__()
        self.conv1 = nn.Conv1d(514, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)

        self.relu = nn.ReLU()

    def forward(self, x):  # input x = batch,514,45^2
        x = self.relu(self.conv1(x))  # x = batch,512,45^2
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x


class DecFold2(nn.Module):
    def __init__(self):
        super(FoldingNetDecFold2, self).__init__()
        self.conv1 = nn.Conv1d(515, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)
        self.relu = nn.ReLU()

    def forward(self, x):  # input x = batch,515,45^2
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = Enc_fold()
        self.decoder = Dec_fold()
    
    def forward(self, x):
        v = self.encoder(x)
        y_coarse, y_detail = self.decoder(v)
        return v, y_coarse, y_detail


if __name__ == "__main__":
    pcs = torch.rand(16, 3, 2048)
    encoder = Encoder()
    v = encoder(pcs)
    print(v.size())

    decoder = Decoder()
    decoder(v)
    y_c, y_d = decoder(v)
    print(y_c.size(), y_d.size())

    ae = AutoEncoder()
    v, y_coarse, y_detail = ae(pcs)
    print(v.size(), y_coarse.size(), y_detail.size())
