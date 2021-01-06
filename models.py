from logging import Logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class RnnModel(nn.Module):
    def __init__(self, logger: Logger, num_words, num_layers: int, model_type: str,
                 input_size: int, hidden_size: int, dropout: float, num_classes: int,
                 bidirectional: bool, RNN_nonlinear_type=None, pretrained_embedding=None):
        super(RnnModel, self).__init__()

        self.embedder = nn.Embedding(num_words, input_size)
        if pretrained_embedding is not None:
            logger.info("Init embedder parameters with pretrained embedding.")
            self.embedder.weight.data.copy_(pretrained_embedding)

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
            nn.ReLU(), nn.Linear(hidden_size, num_classes)
        )

    def forward(self, features: torch.tensor):
        """
        features: [seq_len, batch_size]
        """
        features = self.embedder(features)  # [seq_len, batch_size, input_size]
        output, _ = self.rnn_model(features)
        output = output[-1, :, :]  # [batch_size,(1+bidirectional)*hidden_size]
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

    def forward(self, features: torch.tensor):
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
