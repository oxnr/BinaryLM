import React, { useState } from 'react';
import LLMTutorial from '../components/LLMTutorial';
import './TutorialsPage.css';

// Example code snippets for tutorials
const generateTextWithModel = `
# Example of using sampling parameters with an LLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prompt to generate from
prompt = "Artificial intelligence will"

# Encode the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate with different parameters
def generate_with_params(temperature, top_k, top_p):
    outputs = model.generate(
        input_ids,
        max_length=100,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Compare different parameter settings
print("Low temperature (0.3):")
print(generate_with_params(0.3, 0, 1.0))

print("\\nHigh temperature (1.2):")
print(generate_with_params(1.2, 0, 1.0))

print("\\nWith top_k=50:")
print(generate_with_params(0.7, 50, 1.0))

print("\\nWith top_p=0.9:")
print(generate_with_params(0.7, 0, 0.9))
`;

// Define some shared interfaces for tutorials
interface TutorialStep {
  id: string;
  title: string;
  content: React.ReactNode;
  image?: string;
  codeExample?: string;
  mermaidDiagram?: string;
}

interface TutorialSection {
  id: string;
  title: string;
  description: string;
  steps: TutorialStep[];
}

interface Tutorial {
  id: string;
  title: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedTime: string;
  sections: TutorialSection[];
}

// Define the tokenization tutorial
const TOKENIZATION_TUTORIAL: Tutorial = {
  id: 'tokenization-101',
  title: 'Tokenization 101: How LLMs Process Text',
  description: 'Learn how language models convert text into tokens for processing and why tokenization is crucial for model performance.',
  difficulty: 'beginner',
  estimatedTime: '30 minutes',
  sections: [
    {
      id: 'introduction',
      title: 'Introduction to Tokenization',
      description: 'Understanding the basics of text tokenization',
      steps: [
        {
          id: 'what-is-tokenization',
          title: 'What is Tokenization?',
          content: (
            <div>
              <p>
                Tokenization is the process of converting text into smaller units called tokens. 
                These tokens are the fundamental units that language models process.
              </p>
              <p>
                Think of tokens as the "atoms" of text that the model can understand. 
                For most language models, tokens can be words, parts of words, or even individual characters.
              </p>
              <p>
                For example, the sentence "I love machine learning" might be tokenized into:
              </p>
              <ul>
                <li>["I", "love", "machine", "learning"]</li>
              </ul>
              <p>
                But with subword tokenization (which most modern LLMs use), it might look like:
              </p>
              <ul>
                <li>["I", "love", "mac", "hine", "learn", "ing"]</li>
              </ul>
            </div>
          )
        },
        {
          id: 'why-tokenization-matters',
          title: 'Why Tokenization Matters',
          content: (
            <div>
              <p>
                Tokenization is crucial for several reasons:
              </p>
              <ol>
                <li><strong>Vocabulary Size Management:</strong> Without tokenization, models would need to understand every possible word, requiring enormous vocabularies.</li>
                <li><strong>Handling Unknown Words:</strong> Subword tokenization allows models to process words they've never seen before by breaking them into familiar subword units.</li>
                <li><strong>Efficiency:</strong> Tokens provide a compact representation of text that can be processed more efficiently.</li>
                <li><strong>Cross-Lingual Capabilities:</strong> Tokenization allows models to work across multiple languages with a shared vocabulary.</li>
              </ol>
              <p>
                The quality of tokenization directly affects a model's ability to understand and generate text.
              </p>
            </div>
          )
        }
      ]
    },
    {
      id: 'tokenization-methods',
      title: 'Tokenization Methods',
      description: 'Different approaches to tokenization',
      steps: [
        {
          id: 'word-based',
          title: 'Word-Based Tokenization',
          content: (
            <div>
              <p>
                The simplest form of tokenization is word-based tokenization, where text is split by spaces and punctuation.
              </p>
              <p>
                <strong>Advantages:</strong>
              </p>
              <ul>
                <li>Intuitive and easy to understand</li>
                <li>Preserves word boundaries</li>
              </ul>
              <p>
                <strong>Disadvantages:</strong>
              </p>
              <ul>
                <li>Cannot handle out-of-vocabulary words</li>
                <li>Requires large vocabularies</li>
                <li>Struggles with morphologically rich languages</li>
              </ul>
            </div>
          ),
          mermaidDiagram: `
graph LR
    subgraph "Word-Based Tokenization"
      Text["Hello, world! How are you?"] --> Split["Split by whitespace & punctuation"]
      Split --> Tokens["'Hello', ',', 'world', '!', 'How', 'are', 'you', '?'"]
      Tokens --> IDs["Token IDs: [143, 67, 89, 55, 278, 324, 28, 56]"]
    end
          `
        },
        {
          id: 'character-based',
          title: 'Character-Based Tokenization',
          content: (
            <div>
              <p>
                Character-based tokenization breaks text down into individual characters.
              </p>
              <p>
                <strong>Advantages:</strong>
              </p>
              <ul>
                <li>Tiny vocabulary size</li>
                <li>No out-of-vocabulary problem</li>
                <li>Works well for character-based languages like Chinese</li>
              </ul>
              <p>
                <strong>Disadvantages:</strong>
              </p>
              <ul>
                <li>Very long sequences for processing</li>
                <li>Loses word-level semantics</li>
                <li>Requires more compute for the same text length</li>
              </ul>
            </div>
          ),
          mermaidDiagram: `
graph LR
    subgraph "Character-Based Tokenization"
      Text["Hello"] --> Split["Split by characters"]
      Split --> Tokens["'H', 'e', 'l', 'l', 'o'"]
      Tokens --> IDs["Token IDs: [8, 5, 12, 12, 15]"]
    end
          `
        },
        {
          id: 'subword-tokenization',
          title: 'Subword Tokenization',
          content: (
            <div>
              <p>
                Subword tokenization is a compromise between word-level and character-level approaches. 
                It breaks words into meaningful subword units.
              </p>
              <p>
                The most common subword tokenization methods are:
              </p>
              <ul>
                <li><strong>Byte-Pair Encoding (BPE):</strong> Used by GPT, BERT, and many others</li>
                <li><strong>WordPiece:</strong> Used by BERT and other Google models</li>
                <li><strong>SentencePiece:</strong> Unigram language model, used by XLNet and many multilingual models</li>
              </ul>
              <p>
                These methods break words into subword pieces based on frequency statistics in a corpus.
              </p>
            </div>
          ),
          mermaidDiagram: `
graph LR
    subgraph "Subword Tokenization"
      Text["understanding"] --> Split["Split into subwords"]
      Split --> Tokens["'under', 'stand', 'ing'"]
      Tokens --> IDs["Token IDs: [892, 437, 28]"]
      
      TextOOV["hyperboloid"] --> SplitOOV["Split unfamiliar word"]
      SplitOOV --> TokensOOV["'hyper', 'bo', 'lo', 'id'"]
      TokensOOV --> IDsOOV["Token IDs: [1372, 863, 290, 712]"]
    end
          `
        },
      ]
    },
    {
      id: 'bpe-algorithm',
      title: 'Byte-Pair Encoding (BPE)',
      description: 'Understanding the most common tokenization algorithm',
      steps: [
        {
          id: 'bpe-basics',
          title: 'BPE Algorithm Basics',
          content: (
            <div>
              <p>
                Byte-Pair Encoding (BPE) is one of the most popular subword tokenization algorithms,
                used by models like GPT-3, GPT-4, and many others.
              </p>
              <p>
                The BPE algorithm works as follows:
              </p>
              <ol>
                <li>Start with a vocabulary of individual characters</li>
                <li>Count the frequency of adjacent pairs of characters/subwords</li>
                <li>Merge the most frequent pair into a new token</li>
                <li>Repeat the process until reaching the desired vocabulary size</li>
              </ol>
              <p>
                This creates a vocabulary of common subwords that can be combined to form any word
                in the language, even previously unseen words.
              </p>
            </div>
          )
        },
        {
          id: 'bpe-example',
          title: 'BPE in Action: Step by Step',
          content: (
            <div>
              <p>
                Let's see how BPE works with a simple example:
              </p>
              <p>
                Say we have this tiny corpus: "low lower lowest lowest lowest"
              </p>
              <p>
                Starting with characters:
              </p>
              <ul>
                <li>l, o, w, e, r, s, t, space</li>
                <li>Tokenized: ["l", "o", "w", " ", "l", "o", "w", "e", "r", " ", "l", "o", "w", "e", "s", "t", ...]</li>
              </ul>
              <p>
                After a few BPE merges:
              </p>
              <ol>
                <li>Merge "l" + "o" → "lo"</li>
                <li>Merge "lo" + "w" → "low"</li>
                <li>Merge "e" + "s" → "es"</li>
                <li>Merge "es" + "t" → "est"</li>
              </ol>
              <p>
                Final tokenization:
              </p>
              <ul>
                <li>["low", " ", "low", "er", " ", "low", "est", " ", "low", "est", " ", "low", "est"]</li>
              </ul>
            </div>
          ),
          codeExample: "def learn_bpe(corpus, num_merges):\n    # Start with character vocabulary\n    vocab = set(corpus)\n    \n    # Original corpus split into characters\n    splits = [[c for c in word] for word in corpus.split()]\n    \n    for i in range(num_merges):\n        # Count pairs\n        pairs = get_stats(splits)\n        if not pairs:\n            break\n            \n        # Find most frequent pair\n        best_pair = max(pairs, key=pairs.get)\n        \n        # Create new token from pair\n        new_token = best_pair[0] + best_pair[1]\n        vocab.add(new_token)\n        \n        # Apply merge throughout corpus\n        splits = merge_tokens(best_pair, splits)\n    \n    return vocab, splits"
        }
      ]
    },
    {
      id: 'tokenization-challenges',
      title: 'Challenges in Tokenization',
      description: 'Common problems and solutions in tokenization',
      steps: [
        {
          id: 'multilingual',
          title: 'Multilingual Tokenization',
          content: (
            <div>
              <p>
                Tokenizing across multiple languages presents unique challenges:
              </p>
              <ul>
                <li>Different languages have different character sets and word formation rules</li>
                <li>Some languages like Chinese and Japanese don't use spaces to separate words</li>
                <li>Scripts like Arabic have complex morphology and position-dependent character forms</li>
              </ul>
              <p>
                Solutions:
              </p>
              <ul>
                <li>Train tokenizers on multilingual corpora</li>
                <li>Use byte-level BPE to handle any Unicode character</li>
                <li>Ensure sufficient representation of each language in the training data</li>
              </ul>
            </div>
          )
        },
        {
          id: 'special-tokens',
          title: 'Special Tokens and Control Codes',
          content: (
            <div>
              <p>
                Modern tokenizers include special tokens that serve specific functions:
              </p>
              <ul>
                <li><strong>[CLS]</strong> - Classification token, often used as input for classifiers</li>
                <li><strong>[SEP]</strong> - Separator token, used to separate different segments of text</li>
                <li><strong>[MASK]</strong> - Mask token, used in masked language modeling</li>
                <li><strong>[BOS]/[EOS]</strong> - Beginning/End of sequence tokens</li>
                <li><strong>[PAD]</strong> - Padding token, used to make all sequences the same length</li>
                <li><strong>[UNK]</strong> - Unknown token, used for characters or subwords not in vocabulary</li>
              </ul>
              <p>
                Some models like GPT also use control codes to indicate different tasks or behaviors.
              </p>
            </div>
          )
        },
        {
          id: 'tokenizer-alignment',
          title: 'Tokenization Alignment Issues',
          content: (
            <div>
              <p>
                A critical challenge in tokenization is alignment between tokens and human understanding:
              </p>
              <ul>
                <li>Subword tokens often don't align with linguistic units (morphemes)</li>
                <li>Token counts don't match word counts (important for length constraints)</li>
                <li>Token boundaries can cut across meaningful units</li>
              </ul>
              <p>
                For example, "tokenization" might become ["token", "ization"] instead of ["token", "ization"],
                which doesn't match the morphological structure.
              </p>
              <p>
                This misalignment can sometimes cause problems in understanding and generation, especially
                for technical terms, names, or specialized vocabulary.
              </p>
            </div>
          )
        }
      ]
    },
    {
      id: 'practical-application',
      title: 'Practical Tokenization',
      description: 'Using tokenizers in practice',
      steps: [
        {
          id: 'using-tokenizers',
          title: 'Using Tokenizers in Your Code',
          content: (
            <div>
              <p>
                Most popular language models provide pre-trained tokenizers that you can use directly:
              </p>
              <ul>
                <li>Hugging Face Transformers provides easy access to tokenizers for various models</li>
                <li>OpenAI's tiktoken for GPT models</li>
                <li>SentencePiece library for many multi-lingual models</li>
              </ul>
              <p>
                Using a pre-trained tokenizer is simple and ensures your tokenization matches what the model expects.
              </p>
            </div>
          ),
          codeExample: "# Using Hugging Face tokenizer\nfrom transformers import AutoTokenizer\n\n# Load a pre-trained tokenizer\ntokenizer = AutoTokenizer.from_pretrained(\"gpt2\")\n\n# Tokenize some text\ntext = \"Hello, how are you today?\"\ntokens = tokenizer.encode(text)\nprint(tokens)  # Token IDs\nprint(tokenizer.decode(tokens))  # Back to text\n\n# Get the token strings\ntoken_strings = tokenizer.convert_ids_to_tokens(tokens)\nprint(token_strings)  # ['Hello', ',', 'how', 'are', 'you', 'today', '?']"
        },
        {
          id: 'token-counting',
          title: 'Token Counting and Cost Management',
          content: (
            <div>
              <p>
                Understanding tokenization is crucial for managing API costs and model performance:
              </p>
              <ul>
                <li>API calls for models like GPT are priced per token</li>
                <li>Models have context length limits measured in tokens</li>
                <li>Performance often depends on staying within optimal token ranges</li>
              </ul>
              <p>
                Important considerations:
              </p>
              <ul>
                <li>English text typically averages ~1.3 tokens per word</li>
                <li>Code and technical content often use fewer tokens per character than natural language</li>
                <li>Non-English languages may use significantly more tokens per word</li>
                <li>Special characters, spaces, and formatting all consume tokens</li>
              </ul>
            </div>
          ),
          codeExample: "# Counting tokens for cost estimation\nfrom transformers import AutoTokenizer\n\ntokenizer = AutoTokenizer.from_pretrained(\"gpt2\")\n\ndef estimate_cost(text, input_cost_per_1k=0.001, output_cost_per_1k=0.002, max_output_tokens=500):\n    input_tokens = len(tokenizer.encode(text))\n    \n    # Calculate costs\n    input_cost = (input_tokens / 1000) * input_cost_per_1k\n    max_output_cost = (max_output_tokens / 1000) * output_cost_per_1k\n    \n    print(f\"Input tokens: {input_tokens}\")\n    print(f\"Estimated input cost: ${input_cost:.4f}\")\n    print(f\"Maximum output cost: ${max_output_cost:.4f}\")\n    print(f\"Total maximum cost: ${input_cost + max_output_cost:.4f}\")\n    \n    return input_tokens\n\nsample_text = \"This is a sample text to estimate token count and API cost.\"\nestimate_cost(sample_text)"
        }
      ]
    }
  ]
};

// Define the model architecture tutorial
const ARCHITECTURE_TUTORIAL: Tutorial = {
  id: 'model-architecture-101',
  title: 'Understanding Transformer Architecture',
  description: 'Learn how transformer-based language models are built and how they process information.',
  difficulty: 'intermediate',
  estimatedTime: '45 minutes',
  sections: [
    {
      id: 'transformer-basics',
      title: 'Transformer Basics',
      description: 'Understanding the foundation of modern LLMs',
      steps: [
        {
          id: 'transformer-overview',
          title: 'The Transformer Architecture',
          content: (
            <div>
              <p>
                The transformer architecture, introduced in the 2017 paper "Attention Is All You Need," 
                revolutionized natural language processing and forms the backbone of modern LLMs.
              </p>
              <p>
                Unlike previous sequential models (RNNs, LSTMs), transformers:
              </p>
              <ul>
                <li>Process all tokens in parallel rather than sequentially</li>
                <li>Use attention mechanisms to weigh the importance of different tokens</li>
                <li>Scale better to longer sequences and larger datasets</li>
                <li>Achieve state-of-the-art results on a wide range of language tasks</li>
              </ul>
            </div>
          ),
          mermaidDiagram: `
flowchart TD
    subgraph "Transformer"
      Input[Input Embeddings] --> PE[Positional Encoding]
      PE --> Encoder
      Encoder --> Output[Output Layer]
      
      subgraph Encoder[Encoder Stack]
        direction TB
        E1[Self-Attention] --> E2[Feed Forward]
        E2 --> E3[Layer Norm]
      end
    end
          `
        },
        {
          id: 'attention-mechanism',
          title: 'Self-Attention Mechanism',
          content: (
            <div>
              <p>
                The key innovation in transformers is the self-attention mechanism, which allows the model 
                to weigh the importance of different tokens when processing any given token.
              </p>
              <p>
                For each token, the model computes:
              </p>
              <ul>
                <li><strong>Query (Q):</strong> What the token is looking for</li>
                <li><strong>Key (K):</strong> What each token has to offer</li>
                <li><strong>Value (V):</strong> The information each token provides</li>
              </ul>
              <p>
                The attention score between two tokens is computed by taking the dot product of the 
                query of one token with the key of another, normalized by a scaling factor.
              </p>
            </div>
          ),
          mermaidDiagram: `
graph LR
    subgraph "Self-Attention Mechanism"
      Input[Input Embeddings] --> QW[Q = X·Wq]
      Input --> KW[K = X·Wk]
      Input --> VW[V = X·Wv]
      QW & KW --> Scores[Attention Scores = Q·K^T]
      Scores --> Scaling[Scale by sqrt(dk)]
      Scaling --> Softmax[Softmax]
      Softmax & VW --> Weighted[Weighted Values]
      Weighted --> Output[Output = Sum]
    end
          `
        }
      ]
    },
    {
      id: 'model-components',
      title: 'Model Components',
      description: 'Breaking down the building blocks of transformer models',
      steps: [
        {
          id: 'encoder-decoder',
          title: 'Encoder-Decoder Architecture',
          content: (
            <div>
              <p>
                Transformer models can be categorized based on their architecture:
              </p>
              <ul>
                <li><strong>Encoder-only</strong> (like BERT): Good for understanding tasks</li>
                <li><strong>Decoder-only</strong> (like GPT): Good for generation tasks</li>
                <li><strong>Encoder-decoder</strong> (like T5): Good for translation or summarization</li>
              </ul>
              <p>
                Each architecture has its strengths and is suited for different types of applications.
              </p>
            </div>
          ),
          mermaidDiagram: `
graph TD
    subgraph "Encoder-Only (BERT)"
      EI[Input] --> E1[Encoder Block]
      E1 --> E2[Encoder Block]
      E2 --> EO[Output Representations]
    end
    
    subgraph "Decoder-Only (GPT)"
      DI[Input] --> D1[Decoder Block]
      D1 --> D2[Decoder Block]
      D2 --> DO[Output Probabilities]
    end
    
    subgraph "Encoder-Decoder (T5)"
      EDI[Input] --> ED1[Encoder Block]
      ED1 --> ED2[Encoder Block]
      ED2 --> CrossAttn[Cross-Attention]
      
      T[Target] --> TD1[Decoder Block]
      TD1 --> CrossAttn
      CrossAttn --> TD2[Decoder Block]
      TD2 --> EDO[Output]
    end
          `
        },
        {
          id: 'layer-normalization',
          title: 'Layer Normalization & Residual Connections',
          content: (
            <div>
              <p>
                Two critical components for training deep transformer models are:
              </p>
              <ul>
                <li>
                  <strong>Layer Normalization:</strong> Normalizes the inputs to each sub-layer, 
                  making training more stable and faster
                </li>
                <li>
                  <strong>Residual Connections:</strong> Allow information to flow directly from 
                  earlier layers to later layers, helping with gradient flow during training
                </li>
              </ul>
              <p>
                Without these components, training deep transformer models would be much more difficult.
              </p>
            </div>
          ),
          mermaidDiagram: `
graph TD
    subgraph "Transformer Block with Residual Connections"
      Input --> LN1[Layer Norm]
      LN1 --> Attn[Self-Attention]
      Attn --> Add1[Add]
      Input --> Add1
      Add1 --> LN2[Layer Norm]
      LN2 --> FF[Feed Forward]
      FF --> Add2[Add]
      Add1 --> Add2
      Add2 --> Output
    end
          `
        }
      ]
    }
  ]
};

// Define the model training tutorial
const TRAINING_TUTORIAL: Tutorial = {
  id: 'model-training-101',
  title: 'Training Language Models',
  description: 'Learn the process of training large language models, from data preparation to optimization techniques.',
  difficulty: 'advanced',
  estimatedTime: '60 minutes',
  sections: [
    {
      id: 'training-process',
      title: 'The Training Process',
      description: 'Understanding the end-to-end training pipeline',
      steps: [
        {
          id: 'data-preparation',
          title: 'Data Preparation & Preprocessing',
          content: (
            <div>
              <p>
                Training a language model begins with assembling and preprocessing a large dataset of text.
              </p>
              <p>
                Key steps in data preparation include:
              </p>
              <ol>
                <li>Data collection from diverse sources</li>
                <li>Data cleaning to remove irrelevant content</li>
                <li>Deduplication to eliminate redundant examples</li>
                <li>Tokenization to convert text into model inputs</li>
                <li>Creating training examples with appropriate context lengths</li>
              </ol>
              <p>
                The quality and diversity of training data significantly impacts model performance.
              </p>
            </div>
          ),
          mermaidDiagram: `
flowchart TB
    subgraph "Data Preparation Pipeline"
      Collection["Data Collection<br>Web, Books, Articles, Code"] --> Cleaning["Data Cleaning<br>Remove HTML, Fix Encoding"]
      Cleaning --> Dedup["Deduplication<br>Remove Redundant Text"]
      Dedup --> Filter["Quality Filtering<br>Remove Low-Quality Content"]
      Filter --> Splitting["Train/Val/Test Splitting<br>80/10/10"]
      Splitting --> Tokenization["Tokenization<br>Convert Text to Tokens"]
      Tokenization --> Chunking["Chunking<br>Create Context Windows"]
      Chunking --> Storage["Processed Data Storage<br>Efficient Format"]
    end
          `
        },
        {
          id: 'training-objectives',
          title: 'Training Objectives',
          content: (
            <div>
              <p>
                Language models can be trained with different objectives:
              </p>
              <ul>
                <li>
                  <strong>Next-token prediction (autoregressive):</strong> Predict the next token given previous tokens 
                  (used in GPT models)
                </li>
                <li>
                  <strong>Masked language modeling:</strong> Predict masked tokens in a sentence 
                  (used in BERT)
                </li>
                <li>
                  <strong>Span corruption:</strong> Reconstruct corrupted spans of text 
                  (used in T5)
                </li>
              </ul>
              <p>
                The choice of training objective affects what tasks the model will excel at.
              </p>
            </div>
          ),
          mermaidDiagram: `
graph TD
    subgraph "Training Objectives"
      AR["Autoregressive (GPT)<br>The quick brown [PREDICT]"] --> AR_Out["fox"]
      
      MLM["Masked LM (BERT)<br>The quick [MASK] fox"] --> MLM_Out["brown"]
      
      SC["Span Corruption (T5)<br>The [SPAN] fox --> The quick brown fox"] --> SC_Out["quick brown"]
    end
    
    AR_Out --> AR_Use["Good for:<br>- Text generation<br>- Completion<br>- Chatbots"]
    MLM_Out --> MLM_Use["Good for:<br>- Classification<br>- Sentiment analysis<br>- NER"]
    SC_Out --> SC_Use["Good for:<br>- Translation<br>- Summarization<br>- QA"]
          `
        }
      ]
    },
    {
      id: 'optimization-techniques',
      title: 'Optimization Techniques',
      description: 'Advanced methods to improve training',
      steps: [
        {
          id: 'distributed-training',
          title: 'Distributed Training',
          content: (
            <div>
              <p>
                Large language models are too big to fit on a single GPU, requiring distributed training approaches:
              </p>
              <ul>
                <li><strong>Data Parallelism:</strong> Split batches across multiple GPUs</li>
                <li><strong>Model Parallelism:</strong> Split model layers across multiple GPUs</li>
                <li><strong>Pipeline Parallelism:</strong> Different stages of the forward/backward pass run on different devices</li>
                <li><strong>Tensor Parallelism:</strong> Split individual operations across devices</li>
              </ul>
              <p>
                Modern frameworks like DeepSpeed, Megatron-LM, and PyTorch FSDP help implement these strategies.
              </p>
            </div>
          ),
          mermaidDiagram: `
graph TB
    subgraph "Distributed Training Strategies"
      subgraph "Data Parallelism"
        D1["GPU 1<br>Batch 1<br>Full Model"] 
        D2["GPU 2<br>Batch 2<br>Full Model"]
        D3["GPU 3<br>Batch 3<br>Full Model"]
        D4["GPU 4<br>Batch 4<br>Full Model"]
        
        D1 & D2 & D3 & D4 --> DG["Gradient Sync"]
      end
      
      subgraph "Model Parallelism"
        M1["GPU 1<br>Layers 1-6<br>Full Batch"]
        M2["GPU 2<br>Layers 7-12<br>Full Batch"]
        M3["GPU 3<br>Layers 13-18<br>Full Batch"]
        M4["GPU 4<br>Layers 19-24<br>Full Batch"]
        
        M1 --> M2 --> M3 --> M4
      end
      
      subgraph "Pipeline Parallelism"
        P1["GPU 1<br>Layers 1-6<br>μBatch 1"] --> P2["GPU 2<br>Layers 7-12<br>μBatch 1"] 
        P2 --> P3["GPU 3<br>Layers 13-18<br>μBatch 1"] --> P4["GPU 4<br>Layers 19-24<br>μBatch 1"]
        
        P1 --> P1_2["GPU 1<br>Layers 1-6<br>μBatch 2"]
      end
    end
          `
        },
        {
          id: 'mixed-precision',
          title: 'Mixed Precision Training',
          content: (
            <div>
              <p>
                Mixed precision training uses lower precision formats (like FP16 or BF16) along with FP32 to:
              </p>
              <ul>
                <li>Reduce memory usage by up to 50%</li>
                <li>Speed up matrix multiplications on modern GPUs</li>
                <li>Enable training of larger models or with larger batch sizes</li>
              </ul>
              <p>
                Key techniques include loss scaling to prevent underflow and maintaining master weights in FP32.
              </p>
            </div>
          ),
          mermaidDiagram: `
flowchart TB
    subgraph "Mixed Precision Training"
      FW["Forward Pass (FP16)"] --> Loss["Loss Computation (FP32)"]
      Loss --> Scale["Loss Scaling (Prevent Underflow)"]
      Scale --> BW["Backward Pass (FP16)"]
      BW --> Unscale["Unscale Gradients"]
      Unscale --> Check["Check for Inf/NaN"]
      Check -->|Valid| Update["Update Master Weights (FP32)"]
      Check -->|Invalid| Skip["Skip Update"]
      Update --> Copy["Copy to FP16 Weights"]
    end
          `
        }
      ]
    }
  ]
};

// Define the inference tutorial
const INFERENCE_TUTORIAL: Tutorial = {
  id: 'inference-101',
  title: 'LLM Inference & Deployment',
  description: 'Learn how to efficiently deploy and use trained language models for inference.',
  difficulty: 'intermediate',
  estimatedTime: '40 minutes',
  sections: [
    {
      id: 'inference-basics',
      title: 'Inference Basics',
      description: 'Understanding the inference process for LLMs',
      steps: [
        {
          id: 'generation-methods',
          title: 'Text Generation Methods',
          content: (
            <div>
              <p>
                Different decoding strategies can be used to generate text from language models:
              </p>
              <ul>
                <li><strong>Greedy Decoding:</strong> Always choose the most likely next token</li>
                <li><strong>Beam Search:</strong> Maintain multiple candidate sequences</li>
                <li><strong>Sampling:</strong> Select tokens based on their probability distribution</li>
                <li><strong>Top-k Sampling:</strong> Randomly sample from the k most likely tokens</li>
                <li><strong>Top-p (Nucleus) Sampling:</strong> Sample from tokens comprising top p probability mass</li>
              </ul>
              <p>
                The choice of decoding strategy affects the diversity, coherence, and quality of generated text.
              </p>
            </div>
          ),
          mermaidDiagram: `
graph TD
    Input["Input: 'The cat sat on the'"] --> Model["Language Model"]
    Model --> Probs["Output Probabilities"]
    Probs --> Strategy["Decoding Strategy"]
    
    Strategy --> Greedy["Greedy:<br>mat (0.7)"]
    Strategy --> Beam["Beam Search:<br>mat (0.7)<br>floor (0.2)<br>chair (0.05)"]
    Strategy --> Sampling["Sampling:<br>mat (0.7)<br>floor (0.2)<br>roof (0.02)<br>grass (0.01)"]
    Strategy --> TopK["Top-K (k=3):<br>Sample from:<br>mat (0.7)<br>floor (0.2)<br>chair (0.05)"]
    Strategy --> TopP["Top-P (p=0.9):<br>Sample from:<br>mat (0.7)<br>floor (0.2)"]
    
    Greedy --> Output1["'The cat sat on the mat'"]
    Beam --> Output2["Multiple candidates tracked"]
    Sampling --> Output3["Can be any token<br>based on probabilities"]
    TopK --> Output4["Limited to most likely tokens"]
    TopP --> Output5["Dynamically sized shortlist"]
          `
        },
        {
          id: 'sampling-parameters',
          title: 'Sampling Parameters: Temperature, Top-k, Top-p',
          content: (
            <div>
              <p>
                Key parameters that control the text generation process include:
              </p>
              <ul>
                <li>
                  <strong>Temperature:</strong> Controls the randomness of predictions
                  <ul>
                    <li>Higher temperature (e.g., 0.8) = more random, creative outputs</li>
                    <li>Lower temperature (e.g., 0.2) = more deterministic, focused outputs</li>
                  </ul>
                </li>
                <li>
                  <strong>Top-k:</strong> Limits selection to k most likely tokens
                </li>
                <li>
                  <strong>Top-p (Nucleus):</strong> Dynamically limits to tokens comprising top p (e.g., 0.9) probability mass
                </li>
              </ul>
              <p>
                These parameters can be tuned based on the specific application requirements.
              </p>
            </div>
          ),
          codeExample: generateTextWithModel
        }
      ]
    },
    {
      id: 'deployment-techniques',
      title: 'Deployment Techniques',
      description: 'Methods for efficient model deployment',
      steps: [
        {
          id: 'model-quantization',
          title: 'Model Quantization',
          content: (
            <div>
              <p>
                Quantization reduces the precision of model weights to make inference more efficient:
              </p>
              <ul>
                <li><strong>INT8 Quantization:</strong> Store weights as 8-bit integers instead of 32-bit floats</li>
                <li><strong>4-bit Quantization:</strong> Further reduce to 4 bits per weight (used in QLoRA)</li>
                <li><strong>Mixed-bit Quantization:</strong> Use different precision for different layers</li>
              </ul>
              <p>
                Quantization can reduce model size by 4-8x with minimal impact on quality.
              </p>
            </div>
          ),
          mermaidDiagram: `
graph TD
    subgraph "Quantization for Efficient Inference"
      FP32["FP32 Model<br>Size: 100%<br>Precision: High"] --> INT8["INT8 Quantization<br>Size: 25%<br>Precision: Good"]
      FP32 --> INT4["INT4 Quantization<br>Size: 12.5%<br>Precision: Fair"]
      
      INT8 --> INT8Ben["Benefits:<br>- 4x smaller<br>- 2-4x faster<br>- ~1% quality loss"]
      INT4 --> INT4Ben["Benefits:<br>- 8x smaller<br>- 3-6x faster<br>- 2-5% quality loss"]
    end
          `
        },
        {
          id: 'kv-cache',
          title: 'KV Caching for Efficient Generation',
          content: (
            <div>
              <p>
                Key-Value caching is a technique to speed up autoregressive generation:
              </p>
              <ul>
                <li>Stores the Key (K) and Value (V) tensors from previous generation steps</li>
                <li>Avoids recomputing attention for tokens that have already been processed</li>
                <li>Can provide up to 10x speedup for long sequences</li>
              </ul>
              <p>
                KV caching is essential for making interactive LLM applications responsive.
              </p>
            </div>
          ),
          mermaidDiagram: `
sequenceDiagram
    participant Input as Input Tokens
    participant Model as Transformer
    participant Cache as KV Cache
    participant Output as Generated Tokens
    
    Note over Input,Output: First Forward Pass
    Input->>Model: "The cat sat"
    Model->>Cache: Store K,V for all tokens
    Model->>Output: Predicted next token: "on"
    
    Note over Input,Output: Subsequent Steps
    Output->>Model: Only pass new token: "on"
    Cache->>Model: Retrieve cached K,V for previous tokens
    Model->>Cache: Update cache with K,V for new token
    Model->>Output: Predicted next token: "the"
          `
        }
      ]
    }
  ]
};

// Define the array of available tutorials
const TUTORIALS = [
  TOKENIZATION_TUTORIAL,
  ARCHITECTURE_TUTORIAL,
  TRAINING_TUTORIAL,
  INFERENCE_TUTORIAL
];

const TutorialsPage: React.FC = () => {
  const [selectedTutorial, setSelectedTutorial] = useState<string | null>(null);
  
  // Get the selected tutorial object
  const activeTutorial = TUTORIALS.find(tutorial => tutorial.id === selectedTutorial);
  
  // Render the tutorial catalog if no tutorial is selected
  const renderTutorialCatalog = () => (
    <div className="tutorial-catalog">
      <h1>LLM Tutorial Library</h1>
      <p className="catalog-description">
        Step-by-step tutorials to help you understand and build language models from scratch.
      </p>
      
      <div className="tutorials-grid">
        {TUTORIALS.map(tutorial => (
          <div 
            key={tutorial.id} 
            className="tutorial-card"
            onClick={() => setSelectedTutorial(tutorial.id)}
          >
            <h2>{tutorial.title}</h2>
            <p>{tutorial.description}</p>
            <div className="tutorial-meta">
              <span className={`difficulty-badge ${tutorial.difficulty}`}>{tutorial.difficulty}</span>
              <span className="time-badge">{tutorial.estimatedTime}</span>
            </div>
            <div className="sections-preview">
              {tutorial.sections.length} sections · {tutorial.sections.reduce((total: number, section: TutorialSection) => total + section.steps.length, 0)} lessons
            </div>
            <button className="start-tutorial-button">Start Tutorial</button>
          </div>
        ))}
      </div>
    </div>
  );
  
  return (
    <div className="tutorials-page">
      {activeTutorial ? (
        <div className="active-tutorial-container">
          <button 
            className="back-to-catalog"
            onClick={() => setSelectedTutorial(null)}
          >
            ← Back to Tutorials
          </button>
          <LLMTutorial tutorial={activeTutorial} />
        </div>
      ) : (
        renderTutorialCatalog()
      )}
    </div>
  );
};

export default TutorialsPage; 