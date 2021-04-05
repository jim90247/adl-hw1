import argparse
import json
import os
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import SeqLblDataset
from model import SeqLabeller
from train_intent import categorical_accuracy as token_accuracy
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    # json file with "tokens", "tags" and "id"
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqLblDataset] = {
        split: SeqLblDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    data_loaders = {
        split: DataLoader(dataset,
                          batch_size=args.batch_size,
                          shuffle=(split == TRAIN),
                          collate_fn=dataset.collate_fn,
                          num_workers=4)
        for split, dataset in datasets.items()
    }

    # A 2d float tensor, where embedding[i] is the embedding of i-th token
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqLabeller(embeddings, vocab.pad_id, args.hidden_size, args.num_layers, args.dropout, args.bidirectional,
                        len(tag2idx), args.net_type)

    # TODO: init optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step, 0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=SeqLblDataset.UNK_TAG)

    model = model.to(args.device)
    criterion = criterion.to(args.device)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_stats = {'dev_acc': 0}
    for epoch in epoch_pbar:
        stats = {'train_acc': 0, 'train_loss': 0, 'dev_acc': 0, 'dev_loss': 0}

        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        for batch in data_loaders[TRAIN]:
            optimizer.zero_grad()
            tokens = torch.LongTensor(batch['tokens']).to(args.device)
            tags = torch.LongTensor(batch['tags']).to(args.device)

            out = model(tokens)
            tags = tags.view(-1) # flatten tags to become batch*max_seq_len
            loss = criterion(out, tags)
            acc = token_accuracy(out, tags, SeqLblDataset.UNK_TAG)

            stats['train_acc'] += acc.item()
            stats['train_loss'] += loss.item()

            loss.backward()
            optimizer.step()

        stats['train_acc'] /= len(data_loaders[TRAIN])
        stats['train_loss'] /= len(data_loaders[TRAIN])

        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        with torch.no_grad():
            for batch in data_loaders[DEV]:
                tokens = torch.LongTensor(batch['tokens']).to(args.device)
                tags = torch.LongTensor(batch['tags']).to(args.device)

                out = model(tokens)
                tags = tags.view(-1)
                loss = criterion(out, tags)
                acc = token_accuracy(out, tags)

                stats['dev_acc'] += acc.item()
                stats['dev_loss'] += loss.item()

            stats['dev_acc'] /= len(data_loaders[DEV])
            stats['dev_loss'] /= len(data_loaders[DEV])

        # save checkpoint if better than best one
        if stats['dev_acc'] > best_stats['dev_acc']:
            best_stats = stats.copy()
            best_stats['epoch'] = epoch + 1
            torch.save({
                'net_type': args.net_type,
                'model_state_dict': model.state_dict(),
                **best_stats
            }, os.path.join(args.ckpt_dir, "slot.ckpt"))

        scheduler.step()

        epoch_pbar.set_postfix(stats)

    pprint.pprint(best_stats)
    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--net_type", type=str, choices=SeqLabeller.NET_TYPES.keys(), default="gru")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument("--bidirectional", type=str2bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--step", type=int, default=100)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda")
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
