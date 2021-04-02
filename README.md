# Sample Code for Homework 1 ADL NTU 109 Spring

## Environment

```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
```

## Preprocessing

```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection

```shell
python train_intent.py
```

## Links

- [Kaggle: Intent Classification](https://www.kaggle.com/c/ntu-adl-hw1-intent-cls-spring-2021)
- [Kaggle: Slot Tagging](https://www.kaggle.com/c/ntu-adl-hw1-slot-tag-spring-2021)
- [Github: chakki-works/seqeval](https://github.com/chakki-works/seqeval)
