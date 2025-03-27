import React, { useState } from 'react';
import LLMTutorial from '../components/LLMTutorial';
import { QuizQuestion } from '../components/TutorialQuiz';
import QuizLibrary from '../components/QuizLibrary';
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
export interface TutorialStep {
  id: string;
  title: string;
  content: React.ReactNode;
  image?: string;
  mermaidDiagram?: string;
  codeExample?: string;
}

export interface TutorialSection {
  id: string;
  title: string;
  description: string;
  steps: TutorialStep[];
  quiz?: QuizQuestion[];
}

export interface Tutorial {
  id: string;
  title: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedTime: string;
  sections: TutorialSection[];
}

// Define the tokenization tutorial
export const TOKENIZATION_TUTORIAL: Tutorial = {
  id: 'tokenization-101',
  title: 'Tokenization 101: How LLMs Process Text',
  description: 'Learn how language models convert text into tokens for processing and why tokenization is crucial for model performance.',
  difficulty: 'beginner',
  estimatedTime: '30 minutes',
  sections: [
    {
      id: 'introduction',
      title: 'Introduction to Tokenization',
      description: 'Understand what tokenization is and why it matters in language models.',
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
      ],
      quiz: [
        {
          id: 'intro-q1',
          question: "What is tokenization in the context of language models?",
          options: [
            "Converting text into images for visualization",
            "Breaking text into smaller units for processing by the model",
            "Encrypting text data to ensure privacy",
            "Converting text to speech for voice recognition"
          ],
          correctOptionIndex: 1,
          explanation: "Tokenization is the process of breaking text into smaller units (tokens) that serve as the basic input units for language models. These tokens might be words, subwords, or characters depending on the tokenization method."
        },
        {
          id: 'intro-q2',
          question: "Why is tokenization necessary for language models?",
          options: [
            "It's not actually necessary, just a preprocessing convenience",
            "To reduce memory usage only",
            "To enable models to process fixed-size inputs and handle vocabulary efficiently",
            "To make the training process slower and more thorough"
          ],
          correctOptionIndex: 2,
          explanation: "Tokenization is essential because neural networks need fixed-size numerical inputs. Tokenization converts variable-length text into sequences of tokens with fixed-size representations (embeddings), allowing models to process text efficiently."
        },
        {
          id: 'intro-q3',
          question: "What problem does tokenization help solve in language modeling?",
          options: [
            "The infinite vocabulary problem",
            "The syntax parsing problem",
            "The semantic understanding problem",
            "The language translation problem"
          ],
          correctOptionIndex: 0,
          explanation: "Tokenization helps solve the infinite vocabulary problem. Languages have practically unlimited possible words, but models need a finite vocabulary. Tokenization strategies like subword tokenization allow models to handle unseen words by breaking them into familiar subword pieces."
        }
      ]
    },
    {
      id: 'tokenization-methods',
      title: 'Tokenization Methods',
      description: 'Explore different approaches to tokenization and their trade-offs.',
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
      ],
      quiz: [
        {
          id: 'methods-q1',
          question: "Which tokenization method suffers most from the out-of-vocabulary (OOV) problem?",
          options: [
            "Character-based tokenization",
            "Word-based tokenization",
            "Subword tokenization (BPE)",
            "Hybrid tokenization"
          ],
          correctOptionIndex: 1,
          explanation: "Word-based tokenization suffers most from the OOV problem because it treats each word as a single token. If a word wasn't in the training vocabulary, the model can't process it properly and must use an <UNK> token, losing all semantic information of that word."
        },
        {
          id: 'methods-q2',
          question: "What is an advantage of character-based tokenization?",
          options: [
            "It produces the most compact representation",
            "It has the most semantic meaning per token",
            "It has no out-of-vocabulary (OOV) problem",
            "It requires the least computational resources"
          ],
          correctOptionIndex: 2,
          explanation: "Character-based tokenization has virtually no OOV problem since the vocabulary is very small (just the set of all characters in the language). Any text can be tokenized using just these characters, so there's never an unknown token problem."
        },
        {
          id: 'methods-q3',
          question: "Why is subword tokenization like BPE often preferred in modern language models?",
          options: [
            "It's the fastest method to implement",
            "It balances vocabulary size and semantic meaning",
            "It uses the least memory during inference",
            "It was proven mathematically to be optimal"
          ],
          correctOptionIndex: 1,
          explanation: "Subword tokenization like BPE strikes a balance between word-based (which preserves semantics but has OOV issues) and character-based (which has no OOV issues but loses semantic connections). Subwords are large enough to capture meaning but small enough to combine into unknown words."
        }
      ]
    },
    {
      id: 'bpe-algorithm',
      title: 'Byte-Pair Encoding (BPE)',
      description: 'Master the most popular subword tokenization algorithm used in modern LLMs.',
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
      ],
      quiz: [
        {
          id: 'bpe-q1',
          question: "What is the first step in the BPE algorithm?",
          options: [
            "Merge the most frequent character pair",
            "Split all words into individual characters",
            "Calculate token frequencies in the corpus",
            "Determine the desired vocabulary size"
          ],
          correctOptionIndex: 1,
          explanation: "BPE starts by splitting all words into individual characters, creating the initial character vocabulary. After this initialization, it begins the iterative merging process based on frequency."
        },
        {
          id: 'bpe-q2',
          question: "How does BPE decide which character pairs to merge?",
          options: [
            "Random selection within the text",
            "Based on linguistic rules of the language",
            "By the semantic meaning of the resulting merge",
            "By selecting the most frequent adjacent pair"
          ],
          correctOptionIndex: 3,
          explanation: "BPE merges character pairs based on frequency. In each iteration, it counts all adjacent pairs and merges the most frequent pair, adding the new merged token to the vocabulary."
        },
        {
          id: 'bpe-q3',
          question: "When does the BPE training process stop?",
          options: [
            "When no more merges are possible",
            "When reaching a predefined number of merge operations",
            "When the algorithm reaches 100% accuracy",
            "When all words can be tokenized as single tokens"
          ],
          correctOptionIndex: 1,
          explanation: "The BPE algorithm stops when it reaches a predefined number of merge operations, which effectively determines the final vocabulary size. This is a hyperparameter that balances vocabulary size and tokenization granularity."
        },
        {
          id: 'bpe-q4',
          question: "How does BPE handle completely new words during tokenization?",
          options: [
            "It always uses the <UNK> token for new words",
            "It breaks them into subwords from its learned vocabulary",
            "It adds them to the vocabulary on the fly",
            "It ignores them completely in the input"
          ],
          correctOptionIndex: 1,
          explanation: "BPE handles new words by breaking them down into the subword units it learned during training. If a word wasn't seen during training, it will be split into the subword pieces that comprise its characters, allowing the model to still process it meaningfully."
        }
      ]
    },
    {
      id: 'tokenization-challenges',
      title: 'Challenges in Tokenization',
      description: 'Explore common challenges and edge cases in tokenization.',
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
      ],
      quiz: [
        {
          id: 'challenges-q1',
          question: "Why is multilingual tokenization particularly challenging?",
          options: [
            "Different languages use different character sets",
            "It requires a much larger vocabulary size",
            "Translation between languages becomes necessary",
            "All of the above"
          ],
          correctOptionIndex: 3,
          explanation: "Multilingual tokenization is challenging for multiple reasons: different languages use different character sets and writing systems, require larger vocabularies to cover multiple languages adequately, and may need to handle translation context. Additionally, some languages have very different morphological structures that affect optimal tokenization strategies."
        },
        {
          id: 'challenges-q2',
          question: "What special token is typically used to mark the beginning of a sequence?",
          options: [
            "<BOT>",
            "<BOS>",
            "<START>",
            "<SEQ>"
          ],
          correctOptionIndex: 1,
          explanation: "The <BOS> (Beginning of Sequence) token is typically used to mark the start of an input sequence. This special token helps models understand where input text begins and provides positional context for the first actual tokens."
        },
        {
          id: 'challenges-q3',
          question: "Which of these is NOT typically a challenge in tokenization?",
          options: [
            "Handling emojis and special characters",
            "Aligning tokens with linguistic units",
            "Managing vocabulary size constraints",
            "Ensuring all tokens have the same byte length"
          ],
          correctOptionIndex: 3,
          explanation: "Ensuring all tokens have the same byte length is not a typical tokenization challenge. In fact, tokens intentionally have different lengths in bytes or characters. The actual challenges include handling special characters, maintaining linguistic meaning, and balancing vocabulary size."
        }
      ]
    },
    {
      id: 'practical-application',
      title: 'Practical Applications',
      description: 'Learn how to use tokenization in real-world LLM applications.',
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
      ],
      quiz: [
        {
          id: 'practice-q1',
          question: "Why is it important to use the same tokenizer for inference as was used during training?",
          options: [
            "To maintain consistent copyright licensing",
            "To ensure input is processed the same way the model expects",
            "To improve processing speed",
            "It doesn't matter which tokenizer is used"
          ],
          correctOptionIndex: 1,
          explanation: "Using the same tokenizer for inference as training is crucial because the model learned patterns based on specific token representations. Different tokenizers produce different token sequences for the same text, which would result in inputs the model wasn't trained to understand."
        },
        {
          id: 'practice-q2',
          question: "What method would you use to estimate the cost of processing a large document with an API-based LLM?",
          options: [
            "Count the words and multiply by 0.75",
            "Use the tokenizer to count tokens in the document",
            "Estimate based on character count only",
            "Run a small test and extrapolate"
          ],
          correctOptionIndex: 1,
          explanation: "The most accurate way to estimate processing cost is to use the same tokenizer as the model to count the exact number of tokens in your document. Since pricing is typically per token, this gives you a precise measure of what processing will cost."
        },
        {
          id: 'practice-q3',
          question: "Which practice helps reduce token usage when working with large documents?",
          options: [
            "Always using the full document regardless of length",
            "Removing all whitespace to save tokens",
            "Chunking the document strategically and using context windows",
            "Converting all text to lowercase"
          ],
          correctOptionIndex: 2,
          explanation: "Chunking documents strategically is the best approach for reducing token usage. By breaking large documents into relevant sections and maintaining necessary context, you can process only what's needed rather than sending the entire document in every request."
        }
      ]
    }
  ]
};

// Define the model architecture tutorial
export const ARCHITECTURE_TUTORIAL: Tutorial = {
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
export const TRAINING_TUTORIAL: Tutorial = {
  id: 'training',
  title: 'LLM Training and Optimization',
  description: 'Learn how large language models are trained from start to finish, exploring key techniques and optimization strategies.',
  difficulty: 'intermediate',
  estimatedTime: '45 minutes',
  sections: [
    {
      id: 'training-process',
      title: 'The Training Process',
      description: 'Understanding how large language models are trained from start to finish.',
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
        }
      ],
      quiz: [
        {
          id: 'training-process-q1',
          question: "What is the first step in the LLM training process?",
          options: [
            "Model architecture design",
            "Data collection and preparation",
            "Defining the loss function",
            "Setting up distributed training"
          ],
          correctOptionIndex: 1,
          explanation: "Data collection and preparation is the first step in training an LLM. Before any modeling can begin, you need a large corpus of text that's cleaned, deduplicated, and preprocessed into the appropriate format."
        },
        {
          id: 'training-process-q2',
          question: "Why is tokenization an essential part of data preparation?",
          options: [
            "It compresses the data to save storage space",
            "It converts text into numerical tokens the model can process",
            "It filters out inappropriate content",
            "It translates text into English"
          ],
          correctOptionIndex: 1,
          explanation: "Tokenization converts text into numerical tokens that the model can process. Neural networks operate on numerical data, not raw text, so tokenization bridges this gap by transforming text into token IDs from a fixed vocabulary."
        },
        {
          id: 'training-process-q3',
          question: "What typically happens after the initial pretraining of an LLM?",
          options: [
            "Immediate deployment to production",
            "Fine-tuning and alignment",
            "Deleting the training data",
            "Reducing the model size"
          ],
          correctOptionIndex: 1,
          explanation: "After pretraining, models typically undergo fine-tuning and alignment phases. Fine-tuning adapts the model to specific tasks or domains, while alignment ensures the model follows instructions, is helpful, harmless, and honest through techniques like RLHF."
        }
      ]
    },
    {
      id: 'training-objectives-section',
      title: 'Training Objectives', 
      description: 'Learn about different pretraining objectives used to train language models.',
      steps: [
        {
          id: 'training-objectives-intro',
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
      ],
      quiz: [
        {
          id: 'training-objectives-q1',
          question: "What is autoregressive pretraining?",
          options: [
            "Training a model to predict previous tokens from future ones",
            "Training a model to predict the next token given previous tokens",
            "Training a model to correct grammatical errors automatically",
            "Training a model to compress text efficiently"
          ],
          correctOptionIndex: 1,
          explanation: "Autoregressive pretraining involves training a model to predict the next token in a sequence given all previous tokens. This approach is used in models like GPT and mimics the natural left-to-right reading process of text."
        },
        {
          id: 'training-objectives-q2',
          question: "How does masked language modeling differ from autoregressive pretraining?",
          options: [
            "It predicts tokens in both directions (bidirectional context)",
            "It only uses the first and last tokens of a sequence",
            "It focuses exclusively on rare words",
            "It doesn't use neural networks"
          ],
          correctOptionIndex: 0,
          explanation: "Masked language modeling (used in models like BERT) predicts masked tokens using context from both directions, making it bidirectional. This differs from autoregressive models that can only use previous tokens to predict the next one."
        },
        {
          id: 'training-objectives-q3',
          question: "Which pretraining objective is most commonly used in modern decoder-only LLMs like GPT?",
          options: [
            "Masked language modeling",
            "Span corruption",
            "Next sentence prediction",
            "Autoregressive (next token prediction)"
          ],
          correctOptionIndex: 3,
          explanation: "Autoregressive (next token prediction) is the primary pretraining objective for decoder-only models like GPT. These models are trained to predict the next token given all previous tokens in the sequence, which enables them to generate coherent text continuations."
        }
      ]
    },
    {
      id: 'optimization-section',
      title: 'Optimization Techniques',
      description: 'Explore strategies to make training more efficient and effective.',
      steps: [
        {
          id: 'optimization-intro',
          title: 'Optimization Techniques for Training',
          content: (
            <div>
              <p>
                Training large language models requires sophisticated optimization techniques:
              </p>
              <ul>
                <li><strong>Data Parallelism:</strong> Split data across multiple GPUs</li>
                <li><strong>Model Parallelism:</strong> Split model across multiple GPUs</li>
                <li><strong>Mixed Precision:</strong> Use lower precision for calculations (FP16, BF16)</li>
                <li><strong>Gradient Checkpointing:</strong> Trade computation for memory efficiency</li>
                <li><strong>Gradient Accumulation:</strong> Simulate larger batch sizes</li>
              </ul>
              <p>
                These techniques make it possible to train models with billions of parameters.
              </p>
            </div>
          ),
          mermaidDiagram: `
flowchart TB
    subgraph "Distributed Training Strategies"
      DP["Data Parallelism"] --> DP_Desc["Same model<br>Different data batches<br>Sync gradients"]
      MP["Model Parallelism"] --> MP_Desc["Different model parts<br>Same data<br>Forward/backward coordination"]
      PP["Pipeline Parallelism"] --> PP_Desc["Sequential model parts<br>Mini-batch bubbling<br>Reduced idle time"]
      ZeRO["ZeRO Optimization"] --> ZeRO_Desc["Partition optimizer states<br>Partition gradients<br>Partition parameters"]
    end
          `
        }
      ],
      quiz: [
        {
          id: 'optimization-q1',
          question: "What is data parallelism in distributed training?",
          options: [
            "Splitting the model across multiple devices",
            "Splitting the data across multiple devices",
            "Training multiple different models simultaneously",
            "Processing multiple batches at once on a single device"
          ],
          correctOptionIndex: 1,
          explanation: "Data parallelism splits the training data (batches) across multiple devices. Each device has a full copy of the model but processes different data samples, with gradients being synchronized periodically to update all model copies."
        },
        {
          id: 'optimization-q2',
          question: "Which technique reduces memory usage by storing activations at only certain layers during the forward pass?",
          options: [
            "Model parallelism",
            "Mixed precision training",
            "Gradient checkpointing",
            "Gradient accumulation"
          ],
          correctOptionIndex: 2,
          explanation: "Gradient checkpointing saves memory by storing activations at only certain layers during the forward pass and recomputing the intermediate activations during the backward pass. This trades additional computation for reduced memory usage."
        },
        {
          id: 'optimization-q3',
          question: "Why is mixed precision training beneficial?",
          options: [
            "It always improves model accuracy",
            "It reduces memory usage and speeds up training",
            "It simplifies the model architecture",
            "It enables training without GPUs"
          ],
          correctOptionIndex: 1,
          explanation: "Mixed precision training uses lower precision formats (like FP16) alongside FP32, reducing memory usage and often speeding up training. Modern GPUs have specialized hardware (like NVIDIA's Tensor Cores) that perform operations much faster in FP16 than in FP32."
        },
        {
          id: 'optimization-q4',
          question: "What problem does gradient accumulation solve?",
          options: [
            "Slow convergence rates",
            "Training with larger effective batch sizes than would fit in memory",
            "Model overfitting",
            "Vanishing gradient problems"
          ],
          correctOptionIndex: 1,
          explanation: "Gradient accumulation allows training with larger effective batch sizes than would fit in memory. It works by accumulating gradients over multiple smaller batches before updating the model weights, effectively simulating a larger batch."
        }
      ]
    }
  ]
};

// Define the inference tutorial
export const INFERENCE_TUTORIAL: Tutorial = {
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
export const TUTORIALS = [
  TOKENIZATION_TUTORIAL,
  ARCHITECTURE_TUTORIAL,
  TRAINING_TUTORIAL,
  INFERENCE_TUTORIAL
];

const TutorialsPage: React.FC = () => {
  const [selectedTutorial, setSelectedTutorial] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'catalog' | 'quizzes'>('catalog');
  
  // Get the selected tutorial object
  const activeTutorial = TUTORIALS.find(tutorial => tutorial.id === selectedTutorial);
  
  // Extract all quizzes for the QuizLibrary
  const allQuizTopics = [
    ...(TOKENIZATION_TUTORIAL.sections
      .filter((section: TutorialSection) => section.quiz && section.quiz.length > 0)
      .map((section: TutorialSection) => ({
        id: `tokenization-${section.id}`,
        title: `${TOKENIZATION_TUTORIAL.title}: ${section.title}`,
        description: section.description,
        questions: section.quiz || [],
        difficulty: TOKENIZATION_TUTORIAL.difficulty
      }))),
    ...(ARCHITECTURE_TUTORIAL.sections
      .filter((section: TutorialSection) => section.quiz && section.quiz.length > 0)
      .map((section: TutorialSection) => ({
        id: `architecture-${section.id}`,
        title: `${ARCHITECTURE_TUTORIAL.title}: ${section.title}`,
        description: section.description,
        questions: section.quiz || [],
        difficulty: ARCHITECTURE_TUTORIAL.difficulty
      }))),
    ...(TRAINING_TUTORIAL.sections
      .filter((section: TutorialSection) => section.quiz && section.quiz.length > 0)
      .map((section: TutorialSection) => ({
        id: `training-${section.id}`,
        title: `${TRAINING_TUTORIAL.title}: ${section.title}`,
        description: section.description,
        questions: section.quiz || [],
        difficulty: TRAINING_TUTORIAL.difficulty
      }))),
    ...(INFERENCE_TUTORIAL.sections
      .filter((section: TutorialSection) => section.quiz && section.quiz.length > 0)
      .map((section: TutorialSection) => ({
        id: `inference-${section.id}`,
        title: `${INFERENCE_TUTORIAL.title}: ${section.title}`,
        description: section.description,
        questions: section.quiz || [],
        difficulty: INFERENCE_TUTORIAL.difficulty
      })))
  ];
  
  // Render the tutorial catalog if no tutorial is selected
  const renderTutorialCatalog = () => (
    <div className="tutorial-catalog">
      <h1>LLM Tutorial Library</h1>
      <p className="catalog-description">
        Step-by-step tutorials to help you understand and build language models from scratch.
      </p>
      
      <div className="tutorial-tabs">
        <button 
          className={`tutorial-tab ${activeTab === 'catalog' ? 'active' : ''}`}
          onClick={() => setActiveTab('catalog')}
        >
          Tutorials
        </button>
        <button 
          className={`tutorial-tab ${activeTab === 'quizzes' ? 'active' : ''}`}
          onClick={() => setActiveTab('quizzes')}
        >
          Quizzes
        </button>
      </div>
      
      {activeTab === 'catalog' ? (
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
      ) : (
        <QuizLibrary quizTopics={allQuizTopics} />
      )}
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