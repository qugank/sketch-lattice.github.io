import torch


class HParams:
    def __init__(self):
        self.reload_index = 0

        self.data_location = './fast_data_8_80_keeprio_newAdj/'
        self.save_path = "./model_save_full10_8_80_keeprio_newAdj_res/"
        # self.data_location = './fast_data_16_100_keeprio_newAdj/'
        # self.save_path = "./model_save_full10_16_100_keeprio_newAdj_res/"

        self.data_location = './fast_data_full16_8_80_keeprio_newAdj/'
        self.save_path = "./model_save_full16_8_80_keeprio_newAdj_res/"
        # self.data_location = './fast_data_full16_16_100_keeprio_newAdj/'
        # self.save_path = "./model_save_full16_16_100_keeprio_newAdj_res/"

        self.data_location = './fast_data_full16_8_80_keeprio/'
        self.save_path = "./model_save_full16_8_80_keeprio_res/"
        # self.data_location = './fast_data_full16_16_100_keeprio/'
        # self.save_path = "./model_save_full16_16_100_keeprio_res/"

        self.data_location = './fast_data_32_150_keeprio/'
        self.save_path = "./model_save_full16_32_150_keeprio_res/"
        self.save_path = "./model_save_full10_32_150_keeprio_res/"

        self.save_path = "./model_save_full16_32_150_keeprio_res_XYemb/"
        self.save_path = "./model_save_full10_32_150_keeprio_res_XYemb/"
        self.save_path = "./model_save_full10Cross_32_150_keeprio_res_XYemb/"

        self.data_location = "./fast_data_rebuttal_8_80/"
        self.data_location = "./fast_data_rebuttal_16_100/"
        self.data_location = "./fast_data_rebuttal_32_150/"
        self.data_location = "./fast_data_rebuttal_64_300/"
        self.save_path = self.data_location.replace("fast_data", "model_save_pineapple")
        self.resume_epoch = 220000

        self.data_location = "./fast_data_rebuttal_iccv_32_150/"
        self.save_path = "./model_save_rebuttal_iccv_32_150_circle_like"


        self.data_location = "./fast_data_dT_0.5_rebuttal_iccv/"
        self.save_path = "./model_save_full10Cross_32_150_keeprio_res_XYemb/"
        # self.save_path = "./model_save_32_150_dT_0.1_rebuttal_iccv"

        # self.category = ["airplane.npz", "angel.npz",
        #                  "bear.npz", "bird.npz", "butterfly.npz",
        #                  "cat.npz", "pig.npz"]
        self.category = [
            # "airplane.npz", "angel.npz", "alarm clock.npz", "apple.npz", "butterfly.npz", "belt.npz", "bus.npz",
            # "cake.npz", "cat.npz", "clock.npz", "eye.npz", "fish.npz", "pig.npz",
            # "sheep.npz", "spider.npz", "The Great Wall of China.npz", "umbrella.npz",
        ]
        self.category = [
            "airplane.npz",
            "angel.npz",
            "butterfly.npz",
            "bus.npz",
            "cake.npz",
            "The Great Wall of China.npz",
            "fish.npz",
            "spider.npz",
            "apple.npz",  # "eyeglasses.npz",
            "umbrella.npz",  # "banana.npz",
        ]

        # self.category = [
        #     "pineapple.npz",
        # ]

        # self.category = [
        #     "apple.npz",
        #     "clock.npz",
        #     "circle.npz",
        # ]  ## iccv rebuttal

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
        self.lr = 0.0003
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

        self.graph_number = 80
        self.row_column = 8

        # self.graph_number = 100
        # self.row_column = 16

        self.graph_number = 150
        self.row_column = 32

        # self.row_column, self.graph_number = [int(x) for x in self.data_location.replace("/", "").split("_")[-2:]]

        self.words_number = 256
        self.picture_size = self.words_number

        self.same_category_in_batch = False

        self.use_cuda = torch.cuda.is_available()


hp = HParams()
