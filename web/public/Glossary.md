# Glossary of LLM Terms

This glossary provides definitions for technical terms used throughout this project.

## A

**Absolute Positional Embeddings**: A technique where each position in a sequence is assigned a unique vector representation up to a maximum sequence length.

**Activation Function**: A mathematical function applied to the output of a neuron to introduce non-linearity, such as ReLU (Rectified Linear Unit).

**Adapter**: A small, trainable module inserted into a pre-trained model to enable parameter-efficient fine-tuning.

**Attention**: A mechanism that allows a model to focus on different parts of an input sequence when generating an output, enabling the model to capture dependencies regardless of their distance.

**Attention Head**: One component of multi-head attention that learns to focus on specific patterns or relationships in the data.

**Attention Matrix**: The matrix resulting from computing attention scores between all pairs of tokens in a sequence.

**Autoregressive**: A property of models that generate one token at a time based on all previously generated tokens.

## B

**Backward Pass**: The process of computing gradients of the loss function with respect to model parameters during training.

**Batch**: A group of examples processed together during model training.

**Batch Size**: The number of examples processed in one forward/backward pass.

**Bidirectional**: The capability to process context from both before and after a token, as in encoder-only models like BERT.

**BPE (Byte-Pair Encoding)**: A subword tokenization algorithm that iteratively merges the most frequent pairs of adjacent tokens.

## C

**Causal Language Modeling (CLM)**: A training objective where the model predicts the next token based on previous tokens, used in decoder-only architectures.

**Checkpoint**: A saved state of the model during training, including weights and optimizer state.

**Chinchilla Scaling Laws**: Research findings about the optimal relationship between model size, dataset size, and compute budget.

**Context Window**: The maximum number of tokens that a model can process in a single forward pass.

**Cross-Attention**: A type of attention where queries come from one sequence and keys/values come from another, used in encoder-decoder models.

**CUDA**: NVIDIA's parallel computing platform used for GPU acceleration.

## D

**Decoder**: A component that generates output tokens, often autoregressively.

**Decoder-Only Architecture**: A transformer architecture that uses only the decoder part of the original transformer, commonly used in modern LLMs like GPT.

**DeepSpeed**: A deep learning optimization library for distributed training.

**Direct Preference Optimization (DPO)**: A technique for aligning language models with human preferences without using a separate reward model.

**Distributed Training**: Training a model across multiple GPUs or machines to handle larger models or datasets.

## E

**Embedding**: A dense vector representation of a token or feature.

**Encoder**: A component that processes input sequences into continuous representations.

**Encoder-Decoder Architecture**: A transformer architecture that uses both encoder and decoder components, designed for sequence-to-sequence tasks.

**Encoder-Only Architecture**: A transformer architecture that uses only the encoder part of the original transformer, used in models like BERT.

## F

**Feed-Forward Network (FFN)**: A fully connected neural network within each transformer block.

**Fine-Tuning**: Adapting a pre-trained model to a specific task by updating its parameters on a smaller, task-specific dataset.

**Flash Attention**: An optimized implementation of attention that reduces memory usage and increases speed.

**Forward Pass**: The process of computing outputs from inputs in a neural network.

## G

**Gradient Accumulation**: Accumulating gradients across multiple batches before updating model weights, allowing for effectively larger batch sizes.

**Gradient Checkpointing**: A technique that trades computation for memory by recomputing activations during backpropagation instead of storing them.

**Gradient Clipping**: Limiting the maximum value of gradients to prevent exploding gradients.

**Grouped-Query Attention (GQA)**: An attention variant that shares key and value projections across multiple query heads to reduce computation.

## H

**HuggingFace Transformers**: A popular library providing implementations of transformer models.

## I

**Inference**: The process of using a trained model to make predictions on new inputs.

## K

**KV-Cache**: A technique that stores key and value states from previous tokens to speed up autoregressive generation.

## L

**Layer Normalization**: A technique that normalizes activations within a layer to stabilize training.

**Local Attention**: An attention variant that restricts each token to attend only to nearby tokens.

**LoRA (Low-Rank Adaptation)**: A parameter-efficient fine-tuning method that adds low-rank matrices to pre-trained weights.

**Loss Function**: A function that measures the difference between the model's predictions and the true values.

## M

**Masked Language Modeling (MLM)**: A training objective where the model predicts masked tokens given surrounding context, used in encoder-only architectures.

**Mixed Precision Training**: Using lower precision (e.g., FP16) for some operations to reduce memory usage and increase speed.

**MMLU (Massive Multitask Language Understanding)**: A benchmark testing knowledge across multiple subjects.

**Multi-Head Attention**: A mechanism that performs attention multiple times in parallel with different learned projections.

## N

**Normalization**: Techniques like Layer Normalization that help stabilize training by normalizing activations.

## O

**Optimizer**: An algorithm that updates model parameters based on gradients to minimize the loss function.

## P

**Parameter**: A trainable weight or bias in a neural network.

**Parameter-Efficient Fine-Tuning (PEFT)**: Methods that update only a small subset of a model's parameters during fine-tuning.

**Perplexity**: A measure of how well a language model predicts a sequence, calculated as the exponential of the average negative log-likelihood.

**Positional Encoding/Embedding**: Information added to token embeddings to represent their position in the sequence.

**Pre-LN**: A transformer architecture variant that applies layer normalization before attention and feed-forward blocks.

**Pre-training**: The initial training phase of a model on a large dataset with a general objective before fine-tuning.

**Prompt**: The input sequence provided to a language model to guide its generation.

**Prompt Tuning**: A parameter-efficient fine-tuning method that optimizes continuous prompt embeddings.

## Q

**QLoRA (Quantized LoRA)**: A variant of LoRA that uses quantization to reduce memory requirements.

**Quantization**: Reducing the precision of a model's weights (e.g., from 32-bit to 8-bit) to decrease memory usage and increase inference speed.

**Query, Key, Value (Q, K, V)**: The three projections used in attention mechanisms.

## R

**Relative Positional Embeddings (RPE)**: A technique that encodes relative distances between tokens rather than absolute positions.

**Reinforcement Learning from Human Feedback (RLHF)**: A method that uses human preferences to guide model fine-tuning.

**Residual Connection**: A connection that adds a layer's input to its output, helping with gradient flow during training.

**Rotary Position Embeddings (RoPE)**: A positional embedding technique that applies a rotation to the embedding space based on position.

## S

**Scaling Laws**: Empirical relationships describing how model performance improves with model size, dataset size, and compute.

**Self-Attention**: An attention mechanism where queries, keys, and values all come from the same sequence.

**SentencePiece**: A tokenization library that works with any language without requiring pre-tokenization.

**Sequence-to-Sequence**: Tasks that transform an input sequence to an output sequence, like translation.

**Sliding Window Attention**: An attention variant that restricts attention to a window of recent tokens.

**Softmax**: An activation function that converts a vector of numbers into a probability distribution.

**Sparse Attention**: An attention variant that computes attention for only a subset of token pairs.

**Supervised Fine-Tuning (SFT)**: Fine-tuning a pre-trained model on task-specific examples with explicit labels or desired outputs.

## T

**Tensor**: A multi-dimensional array, the basic data structure in deep learning frameworks.

**Tensor Parallelism**: A technique for distributing tensor computations across multiple devices.

**Token**: A unit of text in NLP, which can be a word, subword, or character.

**Tokenization**: The process of breaking text into tokens.

**Transformer**: A neural network architecture based on self-attention, introduced in the "Attention is All You Need" paper.

## U

**Unidirectional**: Processing that only considers context from previous tokens, as in decoder-only models.

## V

**Vanishing Gradient**: A problem where gradients become extremely small during backpropagation, hindering training.

**Vocabulary**: The set of all tokens that a model knows.

## W

**Warmup**: A training technique where the learning rate increases gradually at the beginning of training.

**Weight Decay**: A regularization technique that penalizes large weights.

**WordPiece**: A subword tokenization algorithm similar to BPE, used in BERT.

**ZeRO (Zero Redundancy Optimizer)**: An optimization technique that partitions model parameters, gradients, and optimizer states across GPUs. 