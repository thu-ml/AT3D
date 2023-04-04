class BaseLoader:
    def __init__(
        self,
        batch_size,
        modelname,
        dir="/data/chang.liu/research/3d_adv",
        device="cuda",
    ):
        self.batch_size = batch_size
        self.modelname = modelname
        self.device = device
        self.pairs = []
        self.pos = 0
        self.dir = dir
        self.need_attack = []

    def __len__(self):
        return len(self.pairs)

    def __iter__(self):
        return self

    def __next__(self):
        pass
