import json
from os import write
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import csv

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        vocab.pad_id,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt['model_state_dict'])

    # TODO: predict dataset
    criterion = nn.CrossEntropyLoss()

    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=dataset.collate_fn,
                             num_workers=4)

    predictions = []
    for batch in data_loader:
        tokens = torch.LongTensor(batch['tokens']).to(args.device)
        out = model(tokens).squeeze(1).to('cpu')
        intents = map(dataset.idx2label, out.argmax(1).tolist())
        predictions.extend({'id': id_, 'intent': intent} for id_, intent in zip(batch['id'], intents))

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=Path, help="Path to the test file.", default="./data/intent/test.json")
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument("--ckpt_path", type=Path, help="Path to model checkpoint.", default="./ckpt/intent/intent.ckpt")
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
