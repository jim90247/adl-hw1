from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.nn.functional import log_softmax


class SeqModel(torch.nn.Module):
    NET_TYPES = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}

    def __init__(self, embeddings: torch.tensor, padding_idx: int, hidden_size: int, num_layers: int, dropout: float,
                 bidirectional: bool, num_class: int, net_type: str) -> None:
        super(SeqModel, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False, padding_idx=padding_idx)
        # TODO: model architecture

        self.net_type: str = net_type.lower()

        if self.NET_TYPES.get(self.net_type) is None:
            raise ValueError(f"Invalid network type: {net_type}")

        self.rnn = self.NET_TYPES[net_type](embeddings.size(1),
                                            hidden_size,
                                            num_layers,
                                            batch_first=True,
                                            bidirectional=bidirectional,
                                            dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_class)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError


class SeqClassifier(SeqModel):
    def __init__(self,
                 embeddings: torch.tensor,
                 padding_idx: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 bidirectional: bool,
                 num_class: int,
                 net_type: str = 'lstm') -> None:
        super().__init__(embeddings, padding_idx, hidden_size, num_layers, dropout, bidirectional, num_class, net_type)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        embedded = self.embed(batch)

        if self.net_type == 'lstm':
            output, (hidden, cell) = self.rnn(embedded)
        else:
            # output shape (batch, seq_len, num_directions * hidden_size) as batch_first is provided
            # hidden shape (num_layers * num_directions, batch, hidden_size)
            output, hidden = self.rnn(embedded)

        last_layer_hidden = torch.cat(
            (hidden[-2, :, :], hidden[-1, :, :]), dim=1) if self.bidirectional else hidden[-1, :, :]

        return self.fc(last_layer_hidden)


class SeqLabeller(SeqModel):
    def __init__(self,
                 embeddings: torch.tensor,
                 padding_idx: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 bidirectional: bool,
                 num_class: int,
                 net_type: str = 'lstm') -> None:
        super().__init__(embeddings, padding_idx, hidden_size, num_layers, dropout, bidirectional, num_class, net_type)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        embedded = self.embed(batch)

        # output shape (batch, seq_len, num_directions * hidden_size) as batch_first is provided
        output, _ = self.rnn(embedded)

        output = output.reshape(-1, output.shape[2])

        dropped = self.dropout(output)

        return self.fc(dropped)
