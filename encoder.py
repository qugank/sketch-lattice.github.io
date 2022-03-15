import torch
import torch.nn as nn
import torchvision
import time
import numpy as np
import random
from generation_hyper_params import hp
from utils.data_process import get_node_coordinates_graph
from utils.sketch_processing import draw_three


class FeatureExtractionBasic(nn.Module):
    def __init__(self):
        super(FeatureExtractionBasic, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 2, 2, 0)  # 64
        self.conv2 = nn.Conv2d(8, 32, 2, 2, 0)  # 32
        self.conv3 = nn.Conv2d(32, 64, 2, 2, 0)  # 16
        self.conv4 = nn.Conv2d(64, 128, 2, 2, 0)  # 8
        self.conv5 = nn.Conv2d(128, 256, 2, 2, 0)  # 4
        self.conv6 = nn.Conv2d(256, 512, 2, 2, 0)  # 2
        self.maxpooling1 = nn.MaxPool2d(2)  # 1
        pass

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))
        x = nn.ReLU()(self.conv5(x))
        x = nn.ReLU()(self.conv6(x))
        x: torch.Tensor = self.maxpooling1(x)
        x = x.view(-1, 512)
        return x


class FeatureExtraction(nn.Module):
    def __init__(self, graph_num=0, graph_size=0, train=True):
        super().__init__()
        self.graph_num = graph_num
        self.graph_size = graph_size
        assert self.graph_num
        assert self.graph_size
        self.featureGenerator = FeatureExtractionBasic()
        self.bn1 = nn.BatchNorm1d(512)

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: (batch_size, graph_num, 3, graph_size, graph_size)
        :return:
        """
        if inputs.shape[0] != 1:
            tmp_batch = 1
            tmp_result = []
            inputs = inputs.view(tmp_batch, -1, 1, self.graph_size, self.graph_size)
            for i in range(tmp_batch):
                tmp_result.append(self.featureGenerator(inputs[i]))
            result = torch.cat(tmp_result).view(-1, self.graph_num, 512)
        else:
            result = self.featureGenerator(inputs.view(-1, 1, self.graph_size, self.graph_size)
                                           ).view(-1, self.graph_num, 512)
        result = self.bn1(result.view(-1, 512)).view(-1, self.graph_num, 512)
        return result


class CoordinateEmbedding(nn.Module):
    def __init__(self, words_number: int, out_dim: int):
        super().__init__()
        self.words_number = words_number
        self.out_dim = out_dim
        self.emb = nn.Embedding(words_number, out_dim // 2, padding_idx=0)

    def forward(self, x: torch.Tensor):
        x = x.long()
        x = self.emb(x)
        x = x.view(-1, hp.graph_number, self.out_dim).contiguous()
        return x


class CoordinateEmbeddingXYSep(nn.Module):
    def __init__(self, words_number: int, out_dim: int):
        super().__init__()
        self.words_number = words_number
        self.out_dim = out_dim
        self.embX = nn.Embedding(words_number, out_dim // 2, padding_idx=0)
        self.embY = nn.Embedding(words_number, out_dim // 2, padding_idx=0)

    def forward(self, c: torch.Tensor):
        c = c.long()
        x, y = torch.split(c, 1, dim=2)
        x_emb = self.embX(x.squeeze(2))
        y_emb = self.embY(y.squeeze(2))
        c = torch.cat([x_emb, y_emb], dim=2)
        c = c.view(-1, hp.graph_number, self.out_dim).contiguous()
        return c


if __name__ == '__main__':
    print("""Coordinate Embedding""")
    CE = CoordinateEmbedding(hp.words_number, hp.embedding_dim)
    CE = CoordinateEmbeddingXYSep(hp.words_number, hp.embedding_dim)
    i = torch.randn((64, hp.graph_number, 2))
    i -= i.min()
    i /= i.max()
    i *= 255
    i = i.long()
    print(i.max(), i.min())
    r = CE(i)
    print("i shape", i.shape)
    print("r shape", r.shape)
    exit(0)


class GCNPropagation2(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.fc1 = nn.Linear(dim_in, dim_in, bias=False)
        self.fc2 = nn.Linear(dim_in, dim_in // 2, bias=False)
        self.fc3 = nn.Linear(dim_in // 2, dim_out, bias=False)

        self.fc = nn.Linear(dim_in, dim_out, bias=False)

        self.relu = nn.ReLU()
        self.I = torch.eye(hp.graph_number)
        if torch.cuda.is_available():
            self.I = self.I.cuda()

    def normalize(self, A: np.ndarray, symmetric=True):
        # A = A+I
        A += self.I
        d = A.sum(axis=2)
        if symmetric:
            # D = D^-1/2
            D = torch.diag_embed(torch.pow(d, -0.5))
            return torch.matmul(D, torch.matmul(A, D))
        else:
            # D=D^-1
            D = torch.diag_embed(torch.pow(d, -1))
            return torch.matmul(D, A)

    def forward(self, X: np.ndarray, A: np.ndarray):
        """
        :param X: (batch, graph_num, in_feature_num)
        :param A: (batch, graph_num, graph_num)
        :return:
        """
        A = A + self.I
        return self.fc(torch.matmul(A, X))


class GCNPropagation(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.resBlockPool = [[
                                 nn.Linear(dim_in, dim_in // 2 * 3, bias=False),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(dim_in // 2 * 3, dim_in, bias=False),
                                 nn.ReLU(),
                             ] * 2]

        self.resBlockSequence = [nn.Sequential(*b).cuda() for b in self.resBlockPool]

        self.out_linear = nn.Linear(dim_in, dim_out, bias=False)

        self.norm = nn.BatchNorm1d(hp.graph_number)
        self.relu = nn.ReLU()

        self.I = torch.eye(hp.graph_number)
        if torch.cuda.is_available():
            self.I = self.I.cuda()

    def normalize(self, A, symmetric=True):
        """
        not work!
        :param A:
        :param symmetric:
        :return:
        """
        # A = A+I
        A += self.I
        d = A.sum(axis=2)
        if symmetric:
            # D = D^-1/2
            D = torch.diag_embed(torch.pow(d, -0.5))
            return torch.matmul(D, torch.matmul(A, D))
        else:
            # D=D^-1
            D = torch.diag_embed(torch.pow(d, -1))
            return torch.matmul(D, A)

    def forward(self, X, A):
        """
        :param X: (batch, graph_num, in_feature_num)
        :param A: (batch, graph_num, graph_num)
        :return:
        """
        A = A + self.I

        for block in self.resBlockSequence:
            X = torch.matmul(A, X)
            last = X
            X = block(X)
            X = X + last

        return X


class EncoderGCN(nn.Module):
    def __init__(self, ):
        super(EncoderGCN, self).__init__()
        # model
        self.emb = CoordinateEmbeddingXYSep(hp.words_number, hp.embedding_dim)

        self.gcn = GCNPropagation(hp.embedding_dim, hp.gcn_out_dim)

        self.fc_h2z = nn.Linear(hp.gcn_out_dim, hp.Nz)
        # z, mu, sigma
        self.fc_mu = nn.Linear(hp.Nz, hp.Nz)
        self.fc_sigma = nn.Linear(hp.Nz, hp.Nz)

        self.norm1 = nn.BatchNorm1d(hp.gcn_out_dim)

    def forward(self, C, A):
        """
        return z, mu, sigma
        :param input_imgs: (batch_size, graph_num, 3, graph_size, graph_size)
        :param adj_matrix: (batch_size, graph_num, graph_num)
        """
        C = self.emb(C)
        X = self.gcn(C, A)
        X = torch.sum(X, dim=1)  # (B, S, dims)
        X = self.norm1(X)
        X = torch.tanh(X)

        # generate mu sigma
        mu = self.fc_mu(X)
        sigma = self.fc_sigma(X)
        sigma_e = torch.exp(sigma / 2.)

        # normal sample
        z_size = mu.size()
        if mu.get_device() != -1:  # not in cpu
            n = torch.normal(torch.zeros(z_size), torch.ones(z_size)).cuda(mu.get_device())
        else:  # in cpu
            n = torch.normal(torch.zeros(z_size), torch.ones(z_size))
        # sample z
        z = mu + sigma_e * n
        return z, mu, sigma, X


class EncoderPatchGCN(nn.Module):
    def __init__(self, ):
        super(EncoderPatchGCN, self).__init__()
        # model
        self.emb = FeatureExtraction(hp.graph_number, 128)
        self.gcn = GCNPropagation(512, hp.gcn_out_dim)

        self.fc_h2z = nn.Linear(hp.gcn_out_dim, hp.Nz)
        # z, mu, sigma
        self.fc_mu = nn.Linear(512, hp.Nz)
        self.fc_sigma = nn.Linear(512, hp.Nz)

        self.norm1 = nn.BatchNorm1d(512)

    def forward(self, C, A):
        """
        return z, mu, sigma
        :param input_imgs: (batch_size, graph_num, 3, graph_size, graph_size)
        :param adj_matrix: (batch_size, graph_num, graph_num)
        """
        C = self.emb(C)
        X = self.gcn(C, A)
        X = torch.sum(X, dim=1)  # (B, S, dims)
        X = self.norm1(X)
        X = torch.tanh(X)

        # generate mu sigma
        mu = self.fc_mu(X)
        sigma = self.fc_sigma(X)
        sigma_e = torch.exp(sigma / 2.)

        # normal sample
        z_size = mu.size()
        if mu.get_device() != -1:  # not in cpu
            n = torch.normal(torch.zeros(z_size), torch.ones(z_size)).cuda(mu.get_device())
        else:  # in cpu
            n = torch.normal(torch.zeros(z_size), torch.ones(z_size))
        # sample z
        z = mu + sigma_e * n
        return z, mu, sigma, X


class SPAttention(nn.Module):
    """
    attention on S dim.
    """

    def __init__(self):
        pass

    def forward(self, x):  # (B, S, f_dims)
        pass
