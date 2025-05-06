# Deep_Learning_Projects
Projects and HWs in [Deep Learning Course](https://deeplearning.cs.cmu.edu/F24/index.html) in CMU. Each project consists of two parts:
- Part 1 focuses on pytorch implementations using **only NumPy**.
- **Part 2** is an **open-ended Kaggle competition** involving PyTorch-based pipelines and end-to-end model training.


## 1 Introduction to Neural Networks

### MLP from Scratch
It includes forward propagation, backpropagation, activation functions, and training with gradient descent, Dropout and Adam and Adamw optimizer.

### Speech Recognition with MLP
I trained a multilayer perceptron (MLP) using PyTorch to classify frame-level phoneme states from Mel spectrogram inputs. The task involves data loading, model architecture design, hyperparameter tuning, and evaluation on a large-scale speech dataset derived from WSJ recordings.


## 2 Face Classification and Verification with CNNs

### CNN from Scratch
It includes Conv1d, Conv2d, Pooling and Flatten.

### Speech Recognition with MLP
Trained CNNs to classify face identities from the VGGFace2 dataset and designed a robust face verification pipeline using learned embeddings.
- Applied comprehensive data augmentation to improve generalization
- Built and trained CNN architectures including CNNs, ResNet, and ConvNeXt variants.
- Implemented and compared Advanced loss functions like Cross-Entropy, Triplet Loss, ArcFace.
- Implemented training pipelines with checkpointing and validation accuracy monitoring.

## 3 Speech Recognition with RNNs

### RNN and Seq2Seq from Scratch  
It includes a vanilla RNN-based and GRU based sequence-to-sequence model with CTC loss.

### Speech Recognition with RNN  
Built an encoder-decoder model using bidirectional LSTMs. The model processes Mel spectrogram inputs and produces phoneme sequences. It incorporating teacher forcing for training and handling variable-length sequences. I used greedy decoding and beam search to generate predictions, visualized attention weights for interpretability, and evaluated the system using Phoneme Error Rate (PER).

## 4 End-to-End Speech Recognition with Transformers

### Transformer from Scratch  
It includes Attention mechanism from scratch using Numpy.
I implemented a Transformer-based encoder-decoder model tailored for speech recognition. The architecture includes multi-head attention, positional encoding, padding and causal masking, and feedforward networks. I designed the encoder with self-attention layers and the decoder with masked self-attention and cross-attention, enabling sequence-to-sequence learning without recurrence.

### Speech Recognition with Transformer  
I trained a Transformer model on the Librispeech dataset to directly map audio features to character sequences. The system supports both greedy and beam search decoding. I explored different training setups including training from scratch, language model pretraining for the decoder, and conditional language modeling. By tuning the model depth, hidden size, and attention heads, I achieved competitive character error rates, demonstrating the effectiveness of Transformer-based architectures in end-to-end speech recognition.