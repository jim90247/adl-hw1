import csv
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data.dataloader import DataLoader

from dataset import SeqLblDataset
from model import SeqLabeller
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqLblDataset(data, vocab, tag2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    ckpt = torch.load(args.ckpt_path)

    model = SeqLabeller(embeddings, vocab.pad_id, args.hidden_size, args.num_layers, args.dropout, args.bidirectional,
                        dataset.num_classes, ckpt['net_type'])
    model.eval()

    # load weights into model
    model.load_state_dict(ckpt['model_state_dict'])

    # TODO: predict dataset
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=dataset.collate_fn,
                             num_workers=4)

    predictions = []
    for batch in data_loader:
        tokens = torch.LongTensor(batch['tokens']).to(args.device)
        local_batch_size = len(batch['id'])

        out = model(tokens)
        out = out.argmax(1)
        out = out.view(local_batch_size, -1)  # batch_size x seq_len
        batch_tags = out.tolist()

        for id_, tokens_, tags in zip(batch['id'], batch['tokens'], batch_tags):
            num_valid_token = sum(t != vocab.pad_id for t in tokens_)  # keep non-padding tags only
            tags_str = " ".join(map(dataset.idx2label, tags[:num_valid_token]))
            predictions.append({'id': id_, 'tags': tags_str})

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=Path, help="Path to the test file.", default="./data/slot/test.json")
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument("--ckpt_path", type=Path, help="Path to model checkpoint.", default="./ckpt/slot/slot.ckpt")
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

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
