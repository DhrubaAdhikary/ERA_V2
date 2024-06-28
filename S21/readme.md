# GPT Training with Cosine Learning Rate Decay and Mixed Precision

This repository contains a GPT model implementation with several advanced features, including:
- Cosine learning rate decay schedule
- Automatic mixed precision training
- Learning rate adjustments at each step
- Enhanced logging of training statistics
- Custom weight initialization logic

## Features

1. **Cosine Learning Rate Decay**: Implements a learning rate schedule that starts with a warmup phase followed by a cosine decay to help stabilize training.
2. **Automatic Mixed Precision**: Uses `torch.autocast` to enable mixed precision training, which improves training speed and reduces memory usage.
3. **Dynamic Learning Rate Adjustment**: Adjusts the learning rate at each training step based on the cosine decay schedule.
4. **Enhanced Logging**: Logs the loss, time taken per step, and tokens processed per second for detailed performance monitoring.
5. **Custom Weight Initialization**: Applies specific initialization logic if `NANGPT_SCALE_INIT` is present, scaling the initialization standard deviation based on the number of layers.

## Training Screenshot

![Training Screenshot](S21\Training_screenshot.png)


## Training was done on COlab T4 so :

params were adapted : 
block_size: int = 1024  # reduced max sequence length to fit into 4GB GPU
vocab_size: int = 50304  # number of tokens
n_layer: int = 6  # increased number of layers for better learning
n_head: int = 8  # increased number of heads for better learning
n_embd: int = 256  # increased embedding dimension for better learning

1. **IPYNB files** -> S21_Training_from_scratch.ipynb
2. **Python File** -> S21_main.py