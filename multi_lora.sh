#!/bin/bash

set -x

for i in $(seq 16); do
    tune run lora_finetune_single_device\
      --config llama3/8B_lora_single_device\
      checkpointer.output_dir=/tmp/lora_finetune/$i\
      output_dir=/tmp/lora_finetune/$i
done