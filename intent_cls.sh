#!/bin/bash
python3.8 test_intent.py --test_file "${1}" --ckpt_path ckpt/intent/best.pt --pred_file "${2}"
