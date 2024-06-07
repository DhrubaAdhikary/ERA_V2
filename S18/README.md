## Objective
Select the "en-it" dataset from opus_books.
Train a transformer model (Encoder-Decoder) using methods of your choice (e.g., PyTorch, One Cycle Policy, Automatic Mixed Precision).
Aim for a target loss of less than 1.8 within a maximum of 18 epochs.

## Model Architecture
The transformer model is built with a series of N stacked Encoder-Decoder blocks, each incorporating multi-head attention mechanisms. Our configuration includes 6 Encoder-Decoder blocks. Tokens are embedded into vectors of 512 dimensions. Each block utilizes multi-head attention with 8 heads and is followed by feed-forward networks of size 128. The model employs positional encodings to maintain sequence context and projects the decoder’s output to the target vocabulary for translation.

## Model Dimensions:

Embedding Dimension (d_model): 512
Feed-Forward Dimension (d_ff): 128
Number of Attention Heads (h): 8

## Training Optimization

To improve efficiency and performance, the following techniques are applied:

1. Parameter Sharing: Sharing weights between the source and target embeddings to reduce the number of parameters and align vector spaces, which is particularly useful for closely related languages.
2. Automatic Mixed Precision (AMP): Utilizes both FP16 and FP32 data types to speed up training while maintaining accuracy.
3. Dynamic Padding: Pads sequences in each batch to the length of the longest sequence, optimizing computational resources.
4. One Cycle Policy (OCP): A learning rate scheduling technique that allows for faster convergence and potentially better model outcomes.

## 📈 Results
Final loss after 18 epochs: 1.4