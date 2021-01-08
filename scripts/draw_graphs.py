import numpy as np
import matplotlib.pyplot as plt
import spacy
import torch
from torchtext import data

DEVICE = torch.device("cuda:0")
spacy_en = spacy.load('en_core_web_lg')


def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


LABEL = data.Field(sequential=False, use_vocab=False)
TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)

train, val = data.TabularDataset.splits(
    path='./data', train='train.csv', validation='dev.csv', format='csv', skip_header=True,
    fields=[('label_id', LABEL), ('title', TEXT), ('description', TEXT)]
)
test = data.TabularDataset('./data/test.csv', format='csv', skip_header=True,
                           fields=[('label_id', None), ('title', TEXT), ('description', TEXT)])
TEXT.build_vocab(train, vectors='glove.840B.300d')


def draw_graph(data):
    x = sorted(np.unique(data))
    y = [np.sum(np.array(data) == item) for item in x]
    c_y = [sum(y[:i + 1]) for i in range(len(y))]
    return x, y, c_y


train_title_lengths = list(map(lambda x: len(x.title), train))
train_description_lengths = list(map(lambda x: len(x.description), train))

x, y, c_y = draw_graph(train_title_lengths)
plt.plot(x, c_y)
plt.savefig("cdf1.pdf")
plt.cla()

x, y, c_y = draw_graph(train_description_lengths)
plt.plot(x, c_y)
plt.savefig("cdf2.pdf")
plt.cla()
