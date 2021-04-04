from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        """Create an instance of SeqClsDataset

        Args:
            data (List[Dict]): json dataset (consists of input and its label)
            vocab (Vocab): the mapping of each word (string) to an unique integer
            label_mapping (Dict[str, int]): the mapping of label (string) to label id
            max_len (int): max length of a padded sentence created by collate_fn
        """
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        return {
            "tokens":
            self.vocab.encode_batch([sample["text"].split() for sample in samples]),
            "intent":
            list(map(self.label2idx, [sample["intent"] for sample in samples if sample.get("intent") is not None])),
            "id": [sample["id"] for sample in samples]
        }

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
