"""Configuration used by the TF-C + DINOv3 InGesture training pipeline."""


class Config(object):
    def __init__(self):
        self.input_channels = 3
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128
        self.num_classes = 8
        self.dropout = 0.35
        self.features_len = 18
        self.num_epoch = 40
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4
        self.drop_last = True
        self.batch_size = 128
        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()
        self.TSlength_aligned = 206
        self.lr_f = self.lr
        self.target_batch_size = 42
        self.increased_dim = 1
        self.num_classes_target = 8
        self.features_len_f = self.features_len
        self.CNNoutput_channel = 28
        self.dinov3_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        self.dinov3_image_size = 224
        self.dinov3_freeze_backbone = True
        self.dinov3_local_files_only = False
        self.dinov3_random_init_for_smoke_test = False


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 6
