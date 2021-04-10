# Applied Deep Learning Homework 1

## Environment

Please use **Python 3.8** and  install all packages in [requirements.txt](requirements.txt).

```bash
pip install -r requirements.txt
```

### Preprocessing

```bash
bash preprocess.sh
```

Preprocessed cache should locate at `cache/`.

## Intent Classification

Train the model:

```bash
$ python train_intent.py \
    --device cuda \
    --bidirectional true \
    --lr 0.001 \
    --dropout 0.5 \
    --step 100 \
    --net_type gru \
    --num_layer 2
Epoch: 100%|███████| 100/100 [06:25<00:00,  3.85s/it, train_acc=0.957, train_loss=0.152, dev_acc=0.893, dev_loss=0.651]
{'dev_acc': 0.9288039455811182,
 'dev_loss': 0.4916963918755452,
 'epoch': 73,
 'train_acc': 1.0,
 'train_loss': 8.895116726477173e-06}
Checkpoint saved to : ckpt/intent/intent.ckpt
```

## Slot Tagging

Train the model:

```bash
$ python train_slot.py \
    --device cuda \
    --bidirectional true \
    --lr 0.001 \
    --dropout 0.5 \
    --step 100 \
    --net_type gru \
    --num_layer 3 \
    --hidden_size 512
Epoch: 100%|█████| 100/100 [08:39<00:00,  5.20s/it, train_acc=0.994, train_loss=0.00273, dev_acc=0.772, dev_loss=0.344]
{'dev_acc': 0.792,
 'dev_correct_batch': 792,
 'dev_correct_token': 7615,
 'dev_loss': 0.18369154918193817,
 'dev_total_token': 7891,
 'epoch': 21,
 'train_acc': 0.9613473219215903,
 'train_correct_batch': 6964,
 'train_correct_token': 55970,
 'train_loss': 0.013676157660889303,
 'train_total_token': 56260}
              precision    recall  f1-score   support

        date       0.75      0.76      0.75       206
  first_name       0.88      0.88      0.88       102
   last_name       0.82      0.78      0.80        78
      people       0.74      0.75      0.74       238
        time       0.82      0.86      0.84       218

   micro avg       0.79      0.80      0.79       842
   macro avg       0.80      0.81      0.80       842
weighted avg       0.79      0.80      0.79       842

Checkpoint saved to : ckpt/slot/slot.ckpt
```
