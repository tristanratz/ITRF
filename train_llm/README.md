This directory contains the code for training the ITRF LLMs.

## Training lora / qlora
When you want to train the ITRF-13b-qlora model yourself just run the run_training.sh.
To run a different sized model you have to modify the parameters size and quantized in the train_llm.py.
The training is optimized to use most or all of the vRAM of 4x NVIDIA GeForce RTX 3090 (24GB).
You may want to modify some of the parameters (especially gradient_accumulation_steps and per device batch size, as well as optizers and so on) if you consider to train on another GPU.


Some experimental code is published in train.ipynb

## LoftQ

We also provide code for training Llama using LoftQ, with which tried to run some experiments, which was unfortunately not possible with our setup due to a bug in DeepSeed.
The code should be able to run using dataparallel or on a single graphics card.
The code is available in the train_llm_loftq.py

To get to know more about LoftQ and how to quantize your model check out: [LoftQ Training](https://github.com/huggingface/peft/tree/8665e2b5719faa4e4b91749ddec09442927b53e0/examples/loftq_finetuning)