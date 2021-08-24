import os
import shutil

from generation_hyper_params import hp
import numpy as np
import matplotlib.pyplot as plt
import PIL
import random
from sklearn.metrics.pairwise import euclidean_distances as cdist

import torch
import torch.nn as nn
from torch import optim
from torch.nn import init
from encoder import EncoderGCN
from decoder import DecoderRNN
from utils.sketch_processing import draw_three
# from utils.data_process import draw_three
from utils.data_process import get_node_coordinates_graph

import time


################################# load and prepare data
class SketchesDataset:
    def __init__(self, path: str, category: list, mode="train"):
        self.sketches = None
        self.sketches_normed = None
        self.max_sketches_len = 0
        self.path = path
        self.category = category
        self.mode = mode

        tmp_sketches = []
        self.category_index = []
        now_count = 0
        for c in self.category:
            dataset = np.load(os.path.join(self.path, c), encoding='latin1', allow_pickle=True)
            tmp_sketches.append(dataset[self.mode])
            self.category_index.append([now_count, now_count + len(dataset[self.mode])])
            now_count += len(dataset[self.mode])
            print(f"dataset: {c} added.")
        print("different categories steps", self.category_index)
        tmp_sketches = np.concatenate(tmp_sketches)

        tmp_nodes = []
        for c in self.category:
            dataset = np.load(os.path.join(self.path, f"{c}_nodes_train.npz"), encoding='latin1', allow_pickle=True)
            tmp_nodes.append(dataset[self.mode])
            print(f"dataset: {c}_nodes_train.npz added.")
        self.tmp_nodes = np.concatenate(tmp_nodes)

        tmp_adjs = []
        for c in self.category:
            dataset = np.load(os.path.join(self.path, f"{c}_adjs_train.npz"), encoding='latin1', allow_pickle=True)
            tmp_adjs.append(dataset[self.mode])
            print(f"dataset: {c}_adjs_train.npz added.")
        self.tmp_adjs = np.concatenate(tmp_adjs)

        print(f"length of trainSet: {len(tmp_sketches)}")
        self.legal_sketch_list = list(range(len(tmp_sketches)))
        # self.legal_sketch_list = []
        self.legal_sketch_list = self.purify(tmp_sketches)  # data clean.  # remove toolong and too stort sketches.
        print("legal data number: ", len(self.legal_sketch_list))
        self.sketches = tmp_sketches.copy()
        self.sketches_normed = self.normalize(tmp_sketches)
        self.Nmax = self.max_size([tmp_sketches[i] for i in self.legal_sketch_list])  # max size of a sketch.

        self.mask_list = list(range(hp.graph_number))

    def max_size(self, sketches):
        """返回所有sketch中 转折最多的一个sketch"""
        sizes = [len(sketch) for sketch in sketches]
        return max(sizes)

    def purify(self, sketches):
        legal_data = []
        for index, sketch in enumerate(sketches):
            if hp.max_seq_length >= sketch.shape[0] > hp.min_seq_length:  # remove small and too long sketches.
                legal_data.append(index)
        return legal_data

    def calculate_normalizing_scale_factor(self, sketches):
        data = []
        for sketch in sketches:
            for stroke in sketch:
                data.append(stroke)
        return np.std(np.array(data))

    def normalize(self, sketches):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        data = []
        scale_factor = self.calculate_normalizing_scale_factor(sketches)
        for sketch in sketches:
            sketch = sketch.astype("float")
            sketch[:, 0:2] /= scale_factor
            data.append(sketch)
        return data

    def make_batch(self, batch_size):
        """
        :param batch_size:
        :return:
        """

        batch_idx = np.random.choice(self.legal_sketch_list, batch_size)
        batch_sketches = [self.sketches_normed[idx] for idx in batch_idx]
        batch_sketches_graphs = [self.sketches[idx] for idx in batch_idx]
        sketches = []
        lengths = []
        graphs = []  # (batch_size * graphs_num_constant, x, y)
        adjs = []
        labels = []
        index = 0

        for _sketch in batch_sketches:
            len_seq = len(_sketch[:, 0])  # sketch
            new_sketch = np.zeros((self.Nmax, 5))  # new a _sketch, all length of sketch in size is Nmax.
            new_sketch[:len_seq, :2] = _sketch[:, :2]

            # set p into one-hot.
            new_sketch[:len_seq - 1, 2] = 1 - _sketch[:-1, 2]
            new_sketch[:len_seq, 3] = _sketch[:, 2]

            # len to Nmax set as 0,0,0,0,1
            new_sketch[(len_seq - 1):, 4] = 1
            new_sketch[len_seq - 1, 2:4] = 0  # x, y, 0, 0, 1
            lengths.append(len(_sketch[:, 0]))  # lengths is _sketch length, not new_sketch length.
            sketches.append(new_sketch)
            index += 1

        for i in batch_idx:
            tmp_graph = self.tmp_nodes[i]
            if hp.mask_prob > 0.:
                for mask_index in random.sample(self.mask_list, k=int(hp.mask_prob * hp.graph_number)):
                    tmp_graph[mask_index, :] = 0

            graphs.append(tmp_graph)

            tmp_adj = self.tmp_adjs[i] / 100.
            adjs.append(tmp_adj)
            labels.append(i // 70000)

        batch = torch.from_numpy(np.stack(sketches, 1)).float()  # (Nmax, batch_size, 5)
        graphs = torch.from_numpy(np.stack(graphs, 0)).float()
        adjs = torch.from_numpy(np.stack(adjs, 0)).float()
        labels = torch.from_numpy(np.stack(labels, 0)).long()

        if hp.use_cuda:
            batch = batch.cuda()  # (Nmax, batch_size, 5)
            graphs = graphs.cuda()  # (batch_size, len, 5)
            adjs = adjs.cuda()
            labels = labels.cuda()

        return batch, lengths, graphs, adjs, labels


def make_coordinate_graph(sketch: np.ndarray, mask_prob: float):
    canvas = draw_three(sketch, img_size=hp.words_number)
    result_points, A = get_node_coordinates_graph(canvas, 8, 8,
                                                  maxPointFilled=hp.graph_number,
                                                  mask_prob=mask_prob, max_pixel_value=hp.words_number - 1)
    # import cv2
    # result_points = result_points.astype("int16")
    # for p in result_points:
    #     if p[0] == p[1] == 0:
    #         continue
    #     cv2.line(canvas, tuple(p), tuple(p), color=(0, 255, 0), thickness=5)
    # cv2.imwrite("test.jpg", canvas)
    # exit(0)
    return result_points, A


sketch_dataset = SketchesDataset(hp.data_location, hp.category, "train")
hp.Nmax = sketch_dataset.Nmax


# hp.Nmax = 148


def sample_bivariate_normal(mu_x: torch.Tensor, mu_y: torch.Tensor,
                            sigma_x: torch.Tensor, sigma_y: torch.Tensor,
                            rho_xy: torch.Tensor, greedy=False):
    mu_x = mu_x.item()
    mu_y = mu_y.item()
    sigma_x = sigma_x.item()
    sigma_y = sigma_y.item()
    rho_xy = rho_xy.item()
    # inputs must be floats
    if greedy:
        return mu_x, mu_y
    mean = [mu_x, mu_y]

    sigma_x *= np.sqrt(hp.temperature)
    sigma_y *= np.sqrt(hp.temperature)

    cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
           [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def make_image(sequence, epoch, name='_output_'):
    strokes = np.split(sequence, np.where(sequence[:, 2] > 0)[0] + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                                    canvas.tostring_rgb())
    name = f"./{hp.save_path}/" + str(epoch) + name + '.jpg'
    pil_image.save(name, "JPEG")
    plt.close("all")


################################# encoder and decoder modules

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(hp.Nz, len(hp.category))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        return self.softmax(x)


class Model:
    def __init__(self):
        if hp.use_cuda:
            self.encoder: nn.Module = EncoderGCN().cuda()
            self.decoder: nn.Module = DecoderRNN().cuda()
            # self.classifier: nn.Module = Classifier().cuda()
        else:
            self.encoder: nn.Module = EncoderGCN()
            self.decoder: nn.Module = DecoderRNN()
            # self.classifier: nn.Module = Classifier()
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), hp.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), hp.lr)
        # self.classifier_optimizer = optim.Adam(self.classifier.parameters(), hp.lr)
        self.eta_step = hp.eta_min

        # self.CLoss = nn.CrossEntropyLoss()

    def lr_decay(self, optimizer: optim, epoch: int):
        """Decay learning rate by a factor of lr_decay"""
        for param_group in optimizer.param_groups:
            if param_group['lr'] > hp.min_lr:
                param_group['lr'] *= hp.lr_decay
        if epoch % 100 == 0:
            print("lr: ", param_group['lr'])
        return optimizer

    def make_target(self, batch, lengths):
        """
        batch torch.Size([129, 100, 5])  Nmax batch_size
        """
        if hp.use_cuda:
            eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.size()[1]).cuda().unsqueeze(
                0)  # torch.Size([1, 100, 5])
        else:
            eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.size()[1]).unsqueeze(0)  # max of len(strokes)

        batch = torch.cat([batch, eos], 0)
        mask = torch.zeros(hp.Nmax + 1, batch.size()[1])
        for indice, length in enumerate(lengths):  # len(lengths) = batchsize
            mask[:length, indice] = 1
        if hp.use_cuda:
            mask = mask.cuda()
        dx = torch.stack([batch.data[:, :, 0]] * hp.M, 2)  # torch.Size([130, 100, 20])
        dy = torch.stack([batch.data[:, :, 1]] * hp.M, 2)  # torch.Size([130, 100, 20])
        p1 = batch.data[:, :, 2]  # torch.Size([130, 100])
        p2 = batch.data[:, :, 3]
        p3 = batch.data[:, :, 4]
        p = torch.stack([p1, p2, p3], 2)  # torch.Size([130, 100, 3])
        return mask, dx, dy, p

    def train(self, epoch):
        self.encoder.train()
        self.decoder.train()
        batch, lengths, graphs, adjs, labels = sketch_dataset.make_batch(hp.batch_size)
        # print(batch, lengths)

        # encode:
        # z, self.mu, self.sigma = self.encoder(batch, hp.batch_size)  # in here, Z is sampled from N(mu, sigma)
        z, self.mu, self.sigma, x = self.encoder(graphs, adjs)  # in here, Z is sampled from N(mu, sigma)
        # torch.Size([100, 128]) torch.Size([100, 128]) torch.Size([100, 128])
        # print(z.shape, self.mu.shape, self.sigma.shape)

        # create start of sequence:
        if hp.use_cuda:
            sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * hp.batch_size).cuda().unsqueeze(0)
            # torch.Size([1, 100, 5])
        else:
            sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * hp.batch_size).unsqueeze(0)
        # had sos at the begining of the batch:
        batch_init = torch.cat([sos, batch], 0)  # torch.Size([130, 100, 5])
        # expend z to be ready to concatenate with inputs:
        z_stack = torch.stack([z] * (hp.Nmax + 1))  # torch.Size([130, 100, 128])
        # inputs is concatenation of z and batch_inputs
        inputs = torch.cat([batch_init, z_stack], 2)  # torch.Size([130, 100, 133])

        # decode:
        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho_xy, self.q, _, _ = self.decoder(inputs, z)

        # prepare targets:
        mask, dx, dy, p = self.make_target(batch, lengths)

        # output_x = self.classifier(x)

        # prepare optimizers:
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # self.classifier_optimizer.zero_grad()
        # update eta for LKL:
        self.eta_step = 1 - (1 - hp.eta_min) * (hp.R ** epoch)  # self.eta_step = 1 - (1 - hp.eta_min) * hp.R
        # compute losses:
        # LKL = self.kullback_leibler_loss()
        LR = self.reconstruction_loss(mask, dx, dy, p, epoch)
        # LC_x = self.CLoss(output_x, labels)
        # loss = LR + LKL
        loss = LR
        # gradient step
        loss.backward()  # all torch.Tensor has backward.
        # gradient cliping
        nn.utils.clip_grad_norm_(self.encoder.parameters(), hp.grad_clip_encode)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), hp.grad_clip)
        # nn.utils.clip_grad_norm_(self.classifier.parameters(), hp.grad_clip)
        # optim step
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        # self.classifier_optimizer.step()
        # some print and save:
        if epoch % 2500 == 0:
            # print('epoch', epoch, 'loss', loss.item(), 'LR', LR.item(), 'LKL', LKL.item())
            print(f'gcn {hp.save_path},epoch:', epoch, 'loss', loss.item(), 'LR', LR.item(), )
            self.encoder_optimizer = self.lr_decay(self.encoder_optimizer, epoch)  # modify optimizer after one step.
            self.decoder_optimizer = self.lr_decay(self.decoder_optimizer, epoch)
            # self.classifier_optimizer = self.lr_decay(self.classifier_optimizer, epoch)
        if epoch == 0:
            return
        if epoch % 5000 == 0:
            self.conditional_generation(epoch)
        if epoch % 10000 == 0:
            self.save(epoch)

    def bivariate_normal_pdf(self, dx, dy):
        z_x = ((dx - self.mu_x) / self.sigma_x) ** 2
        z_y = ((dy - self.mu_y) / self.sigma_y) ** 2
        z_xy = (dx - self.mu_x) * (dy - self.mu_y) / (self.sigma_x * self.sigma_y)
        z = z_x + z_y - 2 * self.rho_xy * z_xy
        exp = torch.exp(-z / (2 * (1 - self.rho_xy ** 2)))
        norm = 2 * np.pi * self.sigma_x * self.sigma_y * torch.sqrt(1 - self.rho_xy ** 2)
        return exp / norm

    def reconstruction_loss(self, mask, dx, dy, p, epoch):
        pdf = self.bivariate_normal_pdf(dx, dy)  # torch.Size([130, 100, 20])
        # stroke
        LS = -torch.sum(mask * torch.log(1e-3 + torch.sum(self.pi * pdf, 2))) / float((hp.Nmax + 1) * hp.batch_size)
        # position
        LP = -torch.sum(p * torch.log(1e-3 + self.q)) / float((hp.Nmax + 1) * hp.batch_size)
        return LS + LP

    def kullback_leibler_loss(self):
        LKL = -0.5 * torch.sum(1 + self.sigma - self.mu ** 2 - torch.exp(self.sigma)) \
              / float(hp.Nz * hp.batch_size)
        if hp.use_cuda:
            KL_min = torch.Tensor([hp.KL_min]).cuda().detach()
        else:
            KL_min = torch.Tensor([hp.KL_min]).detach()
        return hp.wKL * self.eta_step * torch.max(LKL, KL_min)

    def save(self, epoch):
        # sel = np.random.rand()
        torch.save(self.encoder.state_dict(), \
                   f'./{hp.save_path}/encoderRNN_epoch_{epoch}.pth')
        torch.save(self.decoder.state_dict(), \
                   f'./{hp.save_path}/decoderRNN_epoch_{epoch}.pth')
        # torch.save(self.classifier.state_dict(), \
        #            f'./{hp.save_path}/classifier_epoch_{epoch}.pth')

    def load(self, encoder_name, decoder_name):
        saved_encoder = torch.load(encoder_name)
        saved_decoder = torch.load(decoder_name)
        self.encoder.load_state_dict(saved_encoder, strict=False)
        self.decoder.load_state_dict(saved_decoder, strict=False)

    def conditional_generation(self, epoch):
        batch, lengths, graphs, adjs, _ = sketch_dataset.make_batch(1)
        # should remove dropouts:
        self.encoder.train(False)
        self.decoder.train(False)
        # encode:
        z, mu, _, _ = self.encoder(graphs, adjs)
        z = mu  # use mu
        if hp.use_cuda:
            sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).cuda()
        else:
            sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1)
        s = sos
        seq_x = []
        seq_y = []
        seq_z = []
        hidden_cell = None
        for i in range(hp.Nmax):
            input = torch.cat([s, z.unsqueeze(0)], 2)  # start of stroke concatenate with z
            # decode:
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
            self.rho_xy, self.q, hidden, cell = \
                self.decoder(input, z, hidden_cell)
            hidden_cell = (hidden, cell)
            # sample from parameters:
            s, dx, dy, pen_down, eos = self.sample_next_state()
            # ------
            seq_x.append(dx)
            seq_y.append(dy)
            seq_z.append(pen_down)
            if eos:
                print(i)
                break
        # visualize result:
        x_sample = np.cumsum(seq_x, 0)
        y_sample = np.cumsum(seq_y, 0)
        z_sample = np.array(seq_z)
        sequence = np.stack([x_sample, y_sample, z_sample]).T
        make_image(sequence, epoch)

    def sample_next_state(self):
        """
        softmax
        """

        def adjust_temp(pi_pdf):
            pi_pdf = np.log(1e-3 + pi_pdf) / hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        pi = self.pi.data[0, 0, :].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(hp.M, p=pi)
        # get pen state:
        q = self.q.data[0, 0, :].cpu().numpy()
        q = adjust_temp(q)
        q_idx = np.random.choice(3, p=q)
        # get mixture params:
        mu_x = self.mu_x.data[0, 0, pi_idx]
        mu_y = self.mu_y.data[0, 0, pi_idx]
        sigma_x = self.sigma_x.data[0, 0, pi_idx]
        sigma_y = self.sigma_y.data[0, 0, pi_idx]
        rho_xy = self.rho_xy.data[0, 0, pi_idx]
        x, y = sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, greedy=False)  # get samples.
        next_state = torch.zeros(5)
        next_state[0] = x
        next_state[1] = y
        next_state[q_idx + 2] = 1
        if hp.use_cuda:
            return next_state.cuda().view(1, 1, -1), x, y, q_idx == 1, q_idx == 2
        else:
            return next_state.view(1, 1, -1), x, y, q_idx == 1, q_idx == 2


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
    from tqdm import tqdm

    model = Model()
    init_weights(model.encoder)
    init_weights(model.decoder)

    print(get_parameter_number(model.encoder))
    print(get_parameter_number(model.decoder))
    # exit(0)
    # init_weights(model.classifier)
    print(hp.Nmax)
    print(hp.category)
    print(hp.save_path)

    if hp.reload_index:
        print(f"reload model {hp.reload_index}")
        model.load(f"{hp.save_path}/encoderRNN_epoch_{hp.reload_index}.pth",
                   f"{hp.save_path}/decoderRNN_epoch_{hp.reload_index}.pth", )
    os.makedirs(f"./{hp.save_path}", exist_ok=True)
    shutil.copyfile(f"./generation_hyper_params.py", f"./{hp.save_path}/generation_hyper_params.py")
    shutil.copyfile(f"./encoder.py", f"./{hp.save_path}/encoder.py")
    shutil.copyfile(f"./decoder.py", f"./{hp.save_path}/decoder.py")
    for epoch in tqdm(list(range(500001))):
        epoch += hp.reload_index
        model.train(epoch)
    '''
    model.load('encoder.pth','decoder.pth')
    model.conditional_generation(0)
    '''
