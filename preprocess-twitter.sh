#!/bin/bash
if [ ! -f glove.twitter.27B.200d.txt ]; then
  wget http://nlp.stanford.edu/data/glove.twitter.27B.zip -O glove.twitter.27B.zip
  unzip glove.twitter.27B.zip
fi
# python preprocess_intent.py --glove_path glove.twitter.27B.200d.txt
python preprocess_slot.py --glove_path glove.twitter.27B.200d.txt --output_dir cache/slot-twitter/
