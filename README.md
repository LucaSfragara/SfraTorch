# SfraTorch

A custom framework implementing foundational neural network components and transformer architectures from scratch, with a focus on sequence modeling, and automatic speech recognition (ASR).

## Overview

SfraTorch is a NumPy/PyTorch-based implementation of modern deep learning architectures, featuring:
- Custom autograd engine for automatic differentiation
- Comprehensive neural network layer implementations
- Transformer models (decoder-only and encoder-decoder)
- Advanced sequence generation strategies
- ASR-specific components with CTC loss support

## Project Structure

```
SfraTorch/
├── autograd_engine.py          # Automatic differentiation engine
├── functional.py                # Backward operations for autograd
├── src/
│   ├── nn/                      # Neural network layers (NumPy-based)
│   │   ├── linear.py
│   │   ├── activation.py        # ReLU, GELU, Softmax, Tanh
│   │   ├── batchnorm.py
│   │   ├── Conv1d.py
│   │   ├── Conv2d.py
│   │   ├── pool.py              # MaxPool, MeanPool
│   │   ├── multi_head_attention.py
│   │   └── scaled_dot_product_attention.py
│   └── utils/                   # Gradient buffer utilities
├── mytorch/                     # Recurrent architectures
│   ├── rnn_cell.py
│   ├── gru_cell.py
│   └── GRU.py
├── models/
│   ├── mlp.py                   # Multi-layer perceptrons
│   ├── rnn_classifier.py
│   ├── char_predictor.py
│   └── LanguageModel/           # Transformer implementations
│       ├── transformers.py      # Main transformer models
│       ├── encoder_layers.py
│       ├── decoder_layers.py
│       ├── sublayers.py         # Self/Cross-attention, FFN
│       ├── positional_encoding.py
│       ├── speech_embedding.py
│       ├── masks.py             # PadMask, CausalMask
│       └── sequence_generator.py
└── train_GRU_CTC.py            # Training script with CTC loss
```

## Core Components

### 1. Autograd Engine

Custom automatic differentiation system supporting:
- **Operation tracking**: Records computational graph dynamically
- **Gradient computation**: Backpropagation through arbitrary operations
- **Gradient buffer**: Efficient gradient storage and updates

```python
from autograd_engine import Autograd

autograd = Autograd()
# Operations are tracked automatically
# Call backward to compute gradients
autograd.backward(loss)
```

### 2. Neural Network Layers

#### Basic Layers
- **Linear**: Fully connected layer with weight and bias
- **Activation Functions**: ReLU, GELU, Softmax, Tanh
- **Normalization**: Batch Normalization (1D)

#### Convolutional Layers
- **Conv1d**: 1D convolution with stride and padding support
- **Conv2d**: 2D convolution with stride and padding support
- **Pooling**: MaxPool2d, MeanPool2d with configurable kernel and stride

#### Recurrent Layers
- **RNN Cell**: Basic recurrent cell with tanh activation
- **GRU Cell**: Gated Recurrent Unit with update and reset gates
- **GRU**: Multi-layer GRU implementation

#### Attention Mechanisms
- **Scaled Dot-Product Attention**: Core attention mechanism with optional masking
- **Multi-Head Attention**: Parallel attention heads with learned projections

### 3. Transformer Architectures

#### Decoder-Only Transformer (GPT-style)

Pre-LN transformer for autoregressive language modeling:

```python
from models.LanguageModel.transformers import DecoderOnlyTransformer

model = DecoderOnlyTransformer(
    num_layers=12,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    dropout=0.1,
    max_len=512,
    num_classes=50000,
    weight_tying=True,
    layer_drop_rate=0.1
)
```

**Features:**
- Causal masking for autoregressive generation
- Pre-Layer Normalization architecture
- Optional weight tying between embedding and output projection
- Layer dropout for regularization
- Greedy and beam search decoding

#### Encoder-Decoder Transformer (ASR)

Transformer for sequence-to-sequence tasks with speech input:

```python
from models.LanguageModel.transformers import EncoderDecoderTransformer

model = EncoderDecoderTransformer(
    input_dim=80,                    # Mel-spectrogram features
    time_reduction=4,
    reduction_method='lstm',
    num_encoder_layers=6,
    num_encoder_heads=8,
    d_ff_encoder=2048,
    num_decoder_layers=6,
    num_decoder_heads=8,
    d_ff_decoder=2048,
    d_model=512,
    dropout=0.1,
    max_len=1024,
    num_classes=1000
)
```

**Features:**
- Speech embedding with time reduction (LSTM/Conv/Both)
- Bidirectional encoder for acoustic modeling
- Autoregressive decoder with cross-attention
- CTC auxiliary loss for better alignment
- Optional positional encoding skip for encoder/decoder
- Transfer learning from pretrained decoder

### 4. Sequence Generation

Advanced decoding strategies for text generation:

```python
from models.LanguageModel.sequence_generator import SequenceGenerator

generator = SequenceGenerator(
    score_fn=model.score,
    tokenizer=tokenizer,
    max_length=512,
    device='cuda'
)

# Greedy search
sequences, scores = generator.generate_greedy(
    input_ids,
    temperature=1.0,
    repeat_penalty=1.2
)

# Beam search
sequences, scores = generator.generate_beam(
    input_ids,
    beam_width=5,
    temperature=1.0,
    repeat_penalty=1.2
)

# Sampling with top-k/nucleus filtering
sequences, scores = generator.generate_sample(
    input_ids,
    temperature=0.9,
    top_k=50,
    top_p=0.95
)
```

**Decoding Methods:**
- **Greedy Search**: Fast, deterministic decoding
- **Beam Search**: Better quality with diversity control
- **Sampling**: Stochastic generation with temperature, top-k, and nucleus (top-p) filtering
- **Repetition Penalty**: Reduces repetitive outputs

### 5. Supporting Components

#### Positional Encoding
Sinusoidal positional encoding for sequence position information:
```python
from models.LanguageModel.positional_encoding import PositionalEncoding

pe = PositionalEncoding(d_model=512, max_len=1024)
```

#### Speech Embedding
Specialized embedding for speech features with time reduction:
```python
from models.LanguageModel.speech_embedding import SpeechEmbedding

speech_emb = SpeechEmbedding(
    input_dim=80,
    d_model=512,
    time_reduction=4,
    reduction_method='lstm'  # or 'conv', 'both'
)
```

#### Masking Utilities
- **PadMask**: Masks padding positions in variable-length sequences
- **CausalMask**: Prevents attention to future positions in autoregressive models

## Model Architectures

### Pre-Built Models

#### Multi-Layer Perceptrons
- **MLP0**: 2-layer MLP with configurable hidden size
- **MLP1**: 3-layer MLP with batch normalization
- **MLP4**: Deep MLP with skip connections

#### Sequence Models
- **RNN Classifier**: Multi-layer RNN for sequence classification
- **Character Predictor**: GRU-based character-level language model
- **GRU with CTC**: GRU architecture with CTC loss for ASR

## Training

### CTC Training for ASR

```python
from train_GRU_CTC import train

# Configure hyperparameters
model = GRU(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)

# Training loop with gradient clipping
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

### Transfer Learning

Load pretrained decoder weights into encoder-decoder model:

```python
model, param_info = EncoderDecoderTransformer.from_pretrained_decoder(
    decoder_checkpoint_path='decoder.pt',
    config=encoder_decoder_config
)

# Access transferred and new parameters
transferred_params = param_info['transferred']
new_params = param_info['new']
```

## Key Features

### 1. Efficient Implementations
- Vectorized operations with NumPy/PyTorch
- Optimized attention mechanisms with masking support
- Memory-efficient gradient computation

### 2. Modular Design
- Clean separation between components
- Easy to extend with new layers and models
- Consistent API across all modules

### 3. Production-Ready
- Comprehensive error handling and validation
- Type hints for better code clarity
- Well-documented functions and classes

### 4. Advanced Techniques
- **Layer Dropout**: Regularization by randomly skipping layers during training
- **Weight Tying**: Share weights between embedding and output layers
- **Gradient Clipping**: Prevents exploding gradients in RNNs
- **Mixed Precision Support**: Compatible with PyTorch AMP

## Technical Details

### Attention Mechanism

The implementation follows the standard scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

With support for:
- Key padding masks (ignore padding tokens)
- Attention masks (causal masking for autoregressive models)
- Multi-head parallel attention

### Pre-Layer Normalization

Uses Pre-LN architecture for better training stability:

```
x = x + Sublayer(LayerNorm(x))
```

Instead of Post-LN:
```
x = LayerNorm(x + Sublayer(x))
```

### CTC Loss

Connectionist Temporal Classification for sequence-to-sequence learning:
- Handles variable-length input/output sequences
- Learns alignment automatically
- No forced alignment required

## Requirements

```
numpy>=1.21.0
torch>=2.0.0
```

## Usage Examples

### Example 1: Build and Train a Decoder-Only Transformer

```python
from models.LanguageModel.transformers import DecoderOnlyTransformer
import torch

# Initialize model
model = DecoderOnlyTransformer(
    num_layers=6,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    dropout=0.1,
    max_len=512,
    num_classes=10000,
    weight_tying=True
)

# Prepare data
input_ids = torch.randint(0, 10000, (32, 128))  # (batch_size, seq_len)
target_lengths = torch.full((32,), 128)

# Forward pass
logits, attention_weights = model(input_ids, target_lengths)

# Generate text
with torch.no_grad():
    prompt = torch.randint(0, 10000, (1, 10))
    next_token_logits = model.score(prompt)
```

### Example 2: ASR with Encoder-Decoder

```python
from models.LanguageModel.transformers import EncoderDecoderTransformer

# Initialize model
model = EncoderDecoderTransformer(
    input_dim=80,
    time_reduction=4,
    reduction_method='lstm',
    num_encoder_layers=6,
    num_encoder_heads=8,
    d_ff_encoder=2048,
    num_decoder_layers=6,
    num_decoder_heads=8,
    d_ff_decoder=2048,
    d_model=512,
    dropout=0.1,
    max_len=1024,
    num_classes=1000
)

# Prepare audio features and text
audio_features = torch.randn(16, 1000, 80)  # (batch, time, features)
text_tokens = torch.randint(0, 1000, (16, 50))
audio_lengths = torch.full((16,), 1000)
text_lengths = torch.full((16,), 50)

# Forward pass
logits, attention_weights, ctc_inputs = model(
    audio_features,
    text_tokens,
    audio_lengths,
    text_lengths
)

# Use CTC loss
ctc_loss = torch.nn.CTCLoss()(
    ctc_inputs['log_probs'],
    text_tokens,
    ctc_inputs['lengths'],
    text_lengths
)
```

### Example 3: Text Generation with Beam Search

```python
from models.LanguageModel.sequence_generator import SequenceGenerator

generator = SequenceGenerator(
    score_fn=model.score,
    tokenizer=tokenizer,
    max_length=100,
    device='cuda'
)

# Start with a prompt
prompt = tokenizer.encode("Once upon a time")
prompt_tensor = torch.tensor([prompt]).to('cuda')

# Generate with beam search
sequences, scores = generator.generate_beam(
    prompt_tensor,
    beam_width=5,
    temperature=1.0,
    repeat_penalty=1.2
)

# Post-process (truncate at EOS)
generated = SequenceGenerator.post_process_sequence(sequences[0], tokenizer)
text = tokenizer.decode(generated)
print(text)
```

## Architecture Highlights

### Transformer Components

```
EncoderDecoderTransformer
├── Speech Embedding
│   ├── Linear projection
│   ├── Time reduction (LSTM/Conv)
│   └── Dropout
├── Encoder Stack
│   └── N × Encoder Layer
│       ├── Self-Attention (bidirectional)
│       ├── Feed-Forward Network
│       └── Layer Normalization + Residual
├── CTC Head
│   ├── Linear projection
│   └── Log-Softmax
├── Decoder Stack
│   └── N × Decoder Layer
│       ├── Masked Self-Attention
│       ├── Cross-Attention (to encoder)
│       ├── Feed-Forward Network
│       └── Layer Normalization + Residual
└── Output Projection
```

### Attention Layer Details

```
Multi-Head Attention
├── Query projection (Linear)
├── Key projection (Linear)
├── Value projection (Linear)
├── Split into H heads
├── Scaled Dot-Product Attention
│   ├── QK^T / √d_k
│   ├── Apply masks (padding + causal)
│   ├── Softmax
│   └── Multiply by V
├── Concatenate heads
└── Output projection (Linear)
```

## Contributing

This project was originally developed as a class assignment and has been refactored into a personal project. Contributions are welcome!

## Acknowledgments

Built with inspiration from:
- "Attention Is All You Need" (Vaswani et al., 2017)
- PyTorch and NumPy communities
- Carnegie Mellon Course - Introduction to Deep Learning
