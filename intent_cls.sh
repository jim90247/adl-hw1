#!/bin/bash
set -x
python3.8 test_intent.py --test_file "${1}" --ckpt_path intent.ckpt --pred_file "${2}"
