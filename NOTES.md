# Effect of seeds on fine-tuning with Lora

* Try out using LORA and LLama-3-8b, fine-tune against a dataset N times, and
compare the outputs of the model with a fixed seed. Do any of the fine-tuning
runs end up significantly better/worse/different than the others?

First download Llama-3:

```
tune download meta-llama/Meta-Llama-3-8B \
    --output-dir <checkpoint_dir> \
    --hf-token <ACCESS TOKEN>
```

Now we can fine-tune once to test. The default configuration uses the Alpaca
cleaned dataset from here:

https://pytorch.org/torchtune/0.1/generated/torchtune.datasets.alpaca_cleaned_dataset.html

This should be fine for our initial tests. We should also try with a coding fine-tuning 
dataset to see what happens.

```
tune run lora_finetune_single_device --config llama3/8B_lora_single_device
```




