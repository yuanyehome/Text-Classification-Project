from prettytable import PrettyTable
import time
import torch
from torch.utils.data.dataset import Dataset


def layer_wise_parameters(model):
    table = PrettyTable()
    table.field_names = ["Layer Name", "Output Shape", "Param #"]
    table.align["Layer Name"] = "l"
    table.align["Output Shape"] = "r"
    table.align["Param #"] = "r"
    for name, parameters in model.named_parameters():
        if parameters.requires_grad:
            table.add_row([name, str(list(parameters.shape)),
                           parameters.numel()])
    return table


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total


class TextDataset(Dataset):
    def __init__(self, datas, vocab, device, lengths, is_test=False):
        self.is_test = is_test
        if not is_test:
            self.labels = torch.tensor(
                list(map(lambda x: int(x) - 1, datas.label_id))).to(device)
        self.features = torch.tensor(list(map(
            lambda sentence: list(map(
                lambda token: vocab.stoi[token], sentence)),
            datas.text
        ))).to(device)
        self.lengths = torch.tensor(lengths)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        if self.is_test:
            return (self.features[index], self.lengths[index])
        else:
            return (self.labels[index], self.features[index], self.lengths[index])
