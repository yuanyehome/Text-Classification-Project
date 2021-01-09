from logging import Logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import math


class RnnModel(nn.Module):
    def __init__(self, logger: Logger, num_words, num_layers: int, model_type: str,
                 input_size: int, hidden_size: int, dropout: float, num_classes: int,
                 bidirectional: bool, RNN_nonlinear_type=None, pretrained_embedding=None):
        super(RnnModel, self).__init__()

        if pretrained_embedding is not None:
            logger.info("Init embedder parameters with pretrained embedding.")
            self.embedder = nn.Embedding.from_pretrained(
                pretrained_embedding, freeze=False)
        else:
            self.embedder = nn.Embedding(num_words, input_size)

        assert model_type in ["RNN", "GRU", "LSTM"]
        if model_type == "RNN":
            assert (RNN_nonlinear_type is not None) \
                and (RNN_nonlinear_type in ["tanh", "relu"])

        self.model_type = model_type
        self.bidirectional = bidirectional
        if model_type == "RNN":
            self.rnn_model = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity=RNN_nonlinear_type,
                dropout=dropout,
                bidirectional=bidirectional
            )
        elif model_type == "LSTM":
            self.rnn_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional
            )
        elif model_type == "GRU":
            self.rnn_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional
            )
        else:
            raise ValueError("No such model type supported!")
        self.output_layer = nn.Sequential(
            nn.Linear((1 + bidirectional) * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, features: torch.Tensor, lengths: torch.Tensor):
        """
        features: [seq_len, batch_size]
        """
        features = self.embedder(features)  # [seq_len, batch_size, input_size]
        packed = pack_padded_sequence(
            features, lengths, batch_first=False, enforce_sorted=False
        )
        output, _ = self.rnn_model(packed)
        output, unpacked_lens = pad_packed_sequence(output, batch_first=False)
        assert all(unpacked_lens == lengths)
        batch_idxs = torch.arange(0, features.size(1)).long()
        len_idxs = lengths - 1
        if lengths.is_cuda:
            len_idxs.cuda()
            batch_idxs.cuda()
        output = output[len_idxs, batch_idxs, :]
        return self.output_layer(output)  # [batch_size, num_classes]


class TextCNN(nn.Module):
    def __init__(self, logger: Logger, num_words, input_size, num_classes, kernel_sizes=[2, 3, 4],
                 kernel_nums=[256, 256, 256], pooling="max", dropout=0, pretrained_embedding=None):
        super(TextCNN, self).__init__()

        self.embedder = nn.Embedding(num_words, input_size)
        if pretrained_embedding is not None:
            logger.info("Init embedder parameters with pretrained embedding.")
            self.embedder.weight.data.copy_(pretrained_embedding)

        assert len(kernel_sizes) == len(kernel_nums)
        self.conv_layers = nn.ModuleList(
            nn.Conv2d(
                in_channels=1,
                out_channels=kernel_num,
                kernel_size=(kernel_size, input_size)
            ) for (kernel_size, kernel_num) in zip(kernel_sizes, kernel_nums)
        )
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        self.output_layer = nn.Linear(sum(kernel_nums), num_classes)

    def forward(self, features: torch.Tensor):
        """
        features: [seq_len, batch_size]
        """
        features = self.embedder(features)  # [seq_len, batch_size, input_size]
        features = features.transpose(0, 1).unsqueeze(1)
        # [batch_size, 1, seq_len, input_size]
        features = [F.relu(conv(features)).squeeze(3)
                    for conv in self.conv_layers]
        # [batch_size, out_channels, seq_len]
        if self.pooling == "max":
            out_features = [F.max_pool1d(features_item, kernel_size=features_item.size(2)).squeeze(2)
                            for features_item in features]
        elif self.pooling == "avg":
            out_features = [F.avg_pool1d(features_item, kernel_size=features_item.size(2)).squeeze(2)
                            for features_item in features]
        else:
            raise ValueError("Unsupported parameters.")
        out_features = torch.cat(out_features, 1)
        # out_features: [batch_size, out_channels]
        # out_channels means sum of kernel_nums
        out_features = self.dropout(out_features)
        return self.output_layer(out_features)


class AttentionGRU(nn.Module):
    def __init__(self, logger: Logger, num_words, num_layers: int,
                 input_size: int, hidden_size: int, dropout: float,
                 num_classes: int, pretrained_embedding=None):
        super().__init__()
        if pretrained_embedding is not None:
            logger.info("Init embedder parameters with pretrained embedding.")
            self.embedder = nn.Embedding.from_pretrained(
                pretrained_embedding, freeze=False)
        else:
            self.embedder = nn.Embedding(num_words, input_size)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )

        self.output_layer = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

        self.tanh_1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        self.tanh_2 = nn.Tanh()

    def forward(self, features: torch.Tensor, lengths: torch.Tensor, vocab):
        mask = (features == vocab.stoi['<pad>'])
        features = self.embedder(features)  # [seq_len, batch_size, input_size]
        packed = pack_padded_sequence(
            features, lengths, batch_first=False, enforce_sorted=False
        )
        output, _ = self.gru(packed)
        output, unpacked_lens = pad_packed_sequence(
            output, batch_first=False, total_length=features.size(0)
        )
        assert all(unpacked_lens == lengths)

        M = self.tanh_1(output)  # [seq_len, batch_size, hidden_size * 2]
        # [batch_size, 1, seq_len]
        alpha = torch.matmul(M, self.w)
        alpha[mask] = float('-inf')
        alpha = F.softmax(alpha, dim=0).t().unsqueeze(1)
        # [batch_size, 2 * hidden_size]
        r = torch.bmm(alpha, output.transpose(0, 1)).squeeze(1)
        h = self.tanh_2(r)

        return self.output_layer(h)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, logger: Logger, num_words, num_layers: int,
                 nhead: int, dropout: float, max_length: int, d_model,
                 num_classes: int, pretrained_embedding=None):
        super().__init__()

        if pretrained_embedding is not None:
            logger.info("Init embedder parameters with pretrained embedding.")
            self.embedder = nn.Embedding.from_pretrained(
                pretrained_embedding, freeze=False)
        else:
            self.embedder = nn.Embedding(num_words, 300)

        self.fc = nn.Linear(300, d_model)

        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.output_layer = nn.Sequential(
            nn.Linear(d_model * max_length, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )

        self.d_model = d_model

    def forward(self, features: torch.Tensor, vocab):
        # [batch_size, seq_len]
        padding_mask = (features.t() == vocab.stoi['<pad>'])
        # [seq_len, batch_size, d_model]
        features = self.fc(self.embedder(features)) * math.sqrt(self.d_model)
        features = self.pos_encoder(features)
        output_features = self.encoder(
            features, src_key_padding_mask=padding_mask
        ).transpose(0, 1)
        output_features = output_features.reshape(output_features.size(0), -1)
        return self.output_layer(output_features)
