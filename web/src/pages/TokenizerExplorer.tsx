import React, { useState } from 'react';
import TokenizationVisualizer from '../components/TokenizationVisualizer';
import './TokenizerExplorer.css';

const TokenizerExplorer: React.FC = () => {
  return (
    <div className="tokenizer-explorer">
      <h1>Tokenizer Explorer</h1>
      
      <div className="card">
        <h2>How Tokenization Works</h2>
        <p>
          Tokenization is the process of breaking text into smaller units (tokens) 
          that the model can process. Different tokenizers (BPE, WordPiece, etc.) 
          use different algorithms to determine how to split text.
        </p>
        
        <h3>Why Tokenization Matters</h3>
        <p>
          Tokenization is the first step in processing text for language models. It transforms 
          raw text into a format that models can understand. The choice of tokenization method
          affects:
        </p>
        <ul>
          <li>How well the model handles rare and out-of-vocabulary words</li>
          <li>The model's ability to understand subword patterns and morphology</li>
          <li>The maximum sequence length the model can process</li>
          <li>Language support and multilingual capabilities</li>
        </ul>
      </div>

      <div className="card visualization-card">
        <h2>Try Tokenization</h2>
        <TokenizationVisualizer />
          
        <div className="explanation">
          <h3>Byte-Pair Encoding (BPE) Explained</h3>
          <p>
            This simulation demonstrates a simplified BPE tokenization process. In practice, BPE works by:
          </p>
          <ol>
            <li><strong>Starting with a character-level vocabulary</strong>: Each character is a separate token</li>
            <li><strong>Counting frequencies</strong>: Identify the most common adjacent character pairs</li>
            <li><strong>Merging pairs</strong>: Combine the most frequent pair into a new token</li>
            <li><strong>Repeating</strong>: Continue merging until reaching a target vocabulary size</li>
          </ol>
          <p>
            The final tokenization applies these learned merges to segment text optimally. This is why
            common words might be single tokens, while rare words get split into multiple subword tokens.
          </p>
        </div>
      </div>
    </div>
  );
};

export default TokenizerExplorer; 