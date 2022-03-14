import torch


class HParams:
    def __init__(self):
        self.reload_index = 0

        self.data_location = "./dataset_32_150"
        self.save_path = "./models_32_150"
        self.category = [
            "airplane.npz",
            # "angel.npz",
        ]

        self.enc_hidden_size = 256  # encoder LSTM h size
        self.dec_hidden_size = 512
        self.Nz = 128  # encoder output size
        self.M = 20
        self.dropout = 0.0
        self.batch_size = 32
        self.eta_min = 0.01
        self.R = 0.99995
        self.KL_min = 0.2
        self.wKL = 0.5
        self.lr = 0.001
        self.lr_decay = 0.99999
        self.min_lr = 0.00003

        self.grad_clip_encode = 20.
        self.grad_clip = 1.
        self.temperature = 0.05

        self.max_seq_length = 200
        self.min_seq_length = 0

        self.Nmax = 0
        self.embedding_dim = 128
        self.mask_prob = 0.10
        self.gcn_out_dim = 128

        # self.row_column, self.graph_number = [int(x) for x in self.data_location.replace("/", "").split("_")[-2:]]
        self.graph_number = 150
        self.row_column = 32

        self.words_number = 256
        self.picture_size = self.words_number

        self.same_category_in_batch = False

        self.use_cuda = torch.cuda.is_available()


hp = HParams()
