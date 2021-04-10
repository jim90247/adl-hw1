import argparse
from collections import defaultdict
import json
import os
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Tuple
import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from dataset import SeqLblDataset
from model import SeqLabeller
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def calculate_accuracy(preds, targets, batch_size, ignore_index: int = -1) -> Tuple[int, int, int]:
    """
    Returns the number of correct batches, correct tokens and total valid tokens.
    """
    top_preds = preds.argmax(1).view(batch_size, -1)
    targets = targets.view(batch_size, -1)

    correct_batch = 0
    correct_token = 0
    total_token = 0

    for pred, target in zip(top_preds, targets):
        m = target.ne(ignore_index)
        pred, target = torch.masked_select(pred, m), torch.masked_select(target, m)
        correct = pred.eq(target)

        correct_token += correct.sum().item()
        total_token += correct.shape[0]
        if correct.sum().item() == correct.shape[0]:
            correct_batch += 1
    return correct_batch, correct_token, total_token


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

    if args.weighted_loss:
        weights = np.zeros(len(tag2idx), dtype=np.float32)
        for sample in datasets[TRAIN]:
            for tag in sample['tags']:
                weights[tag2idx[tag]] += 1
        weights = 1 / weights
        weights /= min(weights)
        weights = np.sqrt(weights)
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weights), ignore_index=SeqLblDataset.PAD_TAG)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=SeqLblDataset.PAD_TAG)

    model = model.to(args.device)
    criterion = criterion.to(args.device)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_stats = {'dev_correct_token': 0}
    for epoch in epoch_pbar:
        stats = defaultdict(lambda: 0)  # all statistics are default to 0

        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        for batch in data_loaders[TRAIN]:
            optimizer.zero_grad()
            tokens = torch.LongTensor(batch['tokens']).to(args.device)
            tags = torch.LongTensor(batch['tags']).to(args.device)
            local_batch_size = len(batch['id'])

            out = model(tokens)
            tags = tags.view(-1)  # flatten tags to become batch*max_seq_len
            loss = criterion(out, tags)
            correct_batch, correct_token, total_token = calculate_accuracy(out, tags, local_batch_size,
                                                                           SeqLblDataset.PAD_TAG)

            stats['train_correct_token'] += correct_token
            stats['train_total_token'] += total_token
            stats['train_correct_batch'] += correct_batch
            stats['train_loss'] += loss.item() * local_batch_size

            loss.backward()
            optimizer.step()

        stats['train_acc'] = stats['train_correct_batch'] / len(datasets[TRAIN])
        stats['train_loss'] /= len(datasets[TRAIN])

        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        with torch.no_grad():
            for batch in data_loaders[DEV]:
                tokens = torch.LongTensor(batch['tokens']).to(args.device)
                tags = torch.LongTensor(batch['tags']).to(args.device)
                local_batch_size = len(batch['id'])

                out = model(tokens)
                tags = tags.view(-1)
                loss = criterion(out, tags)
                correct_batch, correct_token, total_token = calculate_accuracy(out, tags, local_batch_size,
                                                                               SeqLblDataset.PAD_TAG)

                stats['dev_correct_token'] += correct_token
                stats['dev_total_token'] += total_token
                stats['dev_correct_batch'] += correct_batch
                stats['dev_loss'] += loss.item() * local_batch_size

            stats['dev_acc'] = stats['dev_correct_batch'] / len(datasets[DEV])
            stats['dev_loss'] /= len(datasets[DEV])

        # save checkpoint if better than best one
        if stats['dev_correct_token'] > best_stats['dev_correct_token']:
            best_stats = dict(stats)
            best_stats['epoch'] = epoch + 1
            torch.save({
                'net_type': args.net_type,
                'model_state_dict': model.state_dict(),
                **best_stats
            }, os.path.join(args.ckpt_dir, "slot.ckpt"))

        scheduler.step()

        epoch_pbar.set_postfix({k: stats[k] for k in ['train_acc', 'train_loss', 'dev_acc', 'dev_loss']})

    pprint.pprint(best_stats)

    # Load the best model to generate seqeval report
    ckpt = torch.load(args.ckpt_dir / "slot.ckpt")
    model.load_state_dict(ckpt['model_state_dict'])

    predictions = []
    true_tags = []
    with torch.no_grad():
        for batch in data_loaders[DEV]:
            tokens = torch.LongTensor(batch['tokens']).to(args.device)
            tags = torch.LongTensor(batch['tags']).to(args.device)
            local_batch_size = len(batch['id'])

            out = model(tokens)
            preds = out.argmax(1)
            preds = preds.view_as(tags).tolist()  # batch_size x seq_len
            for pred, tag in zip(preds, batch['tags']):
                num_valid_token = sum(t != SeqLblDataset.PAD_TAG for t in tag)  # keep non-padding tags only
                predictions.append([datasets[DEV].idx2label(idx) for idx in pred[:num_valid_token]])
                true_tags.append([datasets[DEV].idx2label(idx) for idx in tag[:num_valid_token]])

    print(classification_report(true_tags, predictions, scheme=IOB2))
    print("Checkpoint saved to :", args.ckpt_dir / "slot.ckpt")

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
    parser.add_argument("--weighted_loss", action='store_true', default=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
