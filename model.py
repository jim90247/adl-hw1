from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        padding_idx: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False, padding_idx=padding_idx)
        # TODO: model architecture
        self.rnn = nn.RNN(embeddings.size(1),
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
        embedded = self.embed(batch)

        # output shape (seq_len, batch, num_directions * hidden_size)
        # hidden shape (num_layers * num_directions, batch, hidden_size)
        output, hidden = self.rnn(embedded)

        last_layer_hidden = torch.cat(
            (hidden[-2, :, :], hidden[-1, :, :]), dim=1) if self.bidirectional else hidden[-1, :, :]

        dropped = self.dropout(last_layer_hidden)

        return self.fc(dropped)
