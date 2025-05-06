# 🤖 Deep_Learning_Projects
Projects and HWs in [Deep Learning Course](https://deeplearning.cs.cmu.edu/F24/index.html) in CMU. Each project consists of two parts:
- 🔧 **Part 1**: Neural architectures implemented using **only NumPy—no DL frameworks**.
- 🏁 **Part 2**: Open-ended **Kaggle challenge** using **PyTorch-based** end-to-end pipelines.


## 🌟 Highlights

- 🧠 Implemented core deep learning modules from scratch in **NumPy**: MLPs, Optimizers， Dropout, Normalization, CNNs, RNNs, LSTMs, Attention, and Transformers.
- 🛠️ Built full **PyTorch pipelines** to solve real-world tasks in **speech recognition** and **computer vision**.
- 🧑‍💻 Designed a **face verification system** using ResNet and combined Triplet Loss, Cross-Entropy, and ArcFace to learn robust facial embeddings.
- 🗣️ Developed a **Transformer-based speech-to-text model** using CTC loss and beam search decoding, achieving competitive Character Error Rate (CER) on LibriSpeech.
- 🏆 Ranked in the **top 5%** of all class Kaggle competitions, outperforming hundreds of submissions.

## 🚀 Project Details

### 1️⃣ Introduction to Neural Networks

#### Part1: MLP from Scratch
Implemented a multilayer perceptron using NumPy, including forward/backward propagation, activation functions (ReLU, Softmax), Dropout, and optimizers like Adam and AdamW.

#### Part2: Speech Recognition with MLP  
Trained an MLP using PyTorch to classify frame-level phonemes from Mel spectrograms extracted from WSJ recordings. Built a full training pipeline with data loading, architecture tuning, and evaluation.



### 2️⃣ Face Classification and Verification with CNNs

#### Part1: CNN from Scratch  
Implemented convolutional layers (Conv1d, Conv2d), pooling, and flattening from scratch in NumPy.

#### Part2: Face Recognition with CNN  
Trained CNNs to classify and verify faces from the VGGFace2 dataset:
- Applied data augmentation and normalization.
- Trained ResNet, ConvNeXt, and custom CNN architectures.
- Used advanced loss functions like **Cross-Entropy**, **Triplet Loss**, and **ArcFace**.
- Built modular PyTorch training code with logging and checkpointing.


### 3️⃣ Speech Recognition with RNNs

#### Part1: RNN and Seq2Seq from Scratch  
Implemented RNN and GRU-based encoder-decoder models in NumPy, supporting CTC loss for alignment-free sequence prediction.

#### Part2: Speech Recognition with RNN  
Built a BiLSTM-based attention model to predict phoneme sequences from spectrograms. Used teacher forcing during training, and applied greedy and beam search decoding.


### 4️⃣ End-to-End Speech Recognition with Transformers

#### Part1: Transformer from Scratch  
Implemented a full Transformer encoder-decoder in NumPy, including multi-head self-attention, positional encoding, masking, and FFN layers.

#### Part2: Speech Recognition with Transformer  
Trained a Transformer on LibriSpeech to perform speech-to-text transcription. Explored training strategies (scratch, LM pretrain, conditional LM) and tuned model depth, hidden size, and attention heads. Used beam search decoding and achieved strong CER.
