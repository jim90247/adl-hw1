#!/bin/bash
set -x

# Make sure these links end with dl=1
INTENT_CKPT_URL="https://www.dropbox.com/s/rix3xrk5yrd5cbp/intent.ckpt?dl=1"
SLOT_CKPT_URL="https://www.dropbox.com/s/v0xtbzneq3lt743/slot.ckpt?dl=1"

wget "${INTENT_CKPT_URL}" -O intent.ckpt
wget "${SLOT_CKPT_URL}" -O slot.ckpt
