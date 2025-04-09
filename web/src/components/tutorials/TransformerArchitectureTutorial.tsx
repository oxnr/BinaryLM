import React from 'react';
import '../Tutorial.css';

const TransformerArchitectureTutorial: React.FC = () => {
  return (
    <div className="tutorial-container">
      <h1>Understanding Transformer Architecture</h1>
      
      <section className="tutorial-section">
        <h2>Introduction to Transformers</h2>
        <p>
          The transformer architecture, introduced in the 2017 paper "Attention Is All You Need," 
          revolutionized natural language processing by enabling models to process entire sequences in parallel 
          rather than sequentially as in RNNs and LSTMs.
        </p>
        <div className="key-insight">
          <h3>Key Insight</h3>
          <p>Transformers rely on self-attention mechanisms rather than recurrence, allowing for more efficient training and better modeling of long-range dependencies.</p>
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>Core Components</h2>
        
        <div className="component-explainer">
          <h3>1. Token Embeddings</h3>
          <p>Convert input tokens (words or subwords) into dense vector representations.</p>
          <p>Dimension: Usually between 512-1024 for base models, larger for bigger models.</p>
        </div>
        
        <div className="component-explainer">
          <h3>2. Positional Encodings</h3>
          <p>Add information about the position of each token in the sequence, since transformers process all tokens in parallel.</p>
          <p>Formula: Uses sine and cosine functions of different frequencies.</p>
          <code>PE(pos, 2i) = sin(pos/10000^(2i/d_model))</code><br />
          <code>PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))</code>
        </div>
        
        <div className="component-explainer">
          <h3>3. Multi-Head Attention</h3>
          <p>The core mechanism that allows the model to attend to different parts of the input sequence.</p>
          <ul>
            <li><strong>Query (Q):</strong> What we're looking for</li>
            <li><strong>Key (K):</strong> What we match against</li>
            <li><strong>Value (V):</strong> What we retrieve if there's a match</li>
          </ul>
          <p>Attention formula: <code>Attention(Q, K, V) = softmax(QK^T/√d_k)V</code></p>
          <p>Multiple attention heads allow the model to focus on different aspects of the input simultaneously.</p>
        </div>
        
        <div className="component-explainer">
          <h3>4. Feed-Forward Networks</h3>
          <p>Two linear transformations with a ReLU activation in between:</p>
          <code>FFN(x) = max(0, xW₁ + b₁)W₂ + b₂</code>
          <p>These process each position independently and identically.</p>
        </div>
        
        <div className="component-explainer">
          <h3>5. Layer Normalization</h3>
          <p>Stabilizes training by normalizing the inputs to each sub-layer.</p>
        </div>
        
        <div className="component-explainer">
          <h3>6. Residual Connections</h3>
          <p>Allow for better gradient flow by adding the input of each sub-layer to its output.</p>
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>Transformer Variants</h2>
        
        <div className="variant-explainer">
          <h3>Encoder-Only Models (e.g., BERT)</h3>
          <p>Bidirectional context, good for understanding tasks like classification.</p>
          <ul>
            <li>Can see the entire input at once</li>
            <li>Suitable for tasks like sentiment analysis, named entity recognition</li>
          </ul>
        </div>
        
        <div className="variant-explainer">
          <h3>Decoder-Only Models (e.g., GPT, Claude)</h3>
          <p>Autoregressive, generating one token at a time based on previous tokens.</p>
          <ul>
            <li>Each position can only attend to previous positions</li>
            <li>Used for text generation, completion, chat</li>
          </ul>
        </div>
        
        <div className="variant-explainer">
          <h3>Encoder-Decoder Models (e.g., T5, BART)</h3>
          <p>Use both components for sequence-to-sequence tasks.</p>
          <ul>
            <li>Encoder processes the input sequence</li>
            <li>Decoder generates the output sequence</li>
            <li>Good for translation, summarization</li>
          </ul>
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>Scaling Transformers</h2>
        <p>
          Transformer models scale effectively with more parameters, compute, and data.
          This has led to the development of increasingly larger models with better capabilities.
        </p>
        <ul>
          <li><strong>Parameter scaling:</strong> More layers, wider layers, more attention heads</li>
          <li><strong>Context window scaling:</strong> Longer sequences to process more context</li>
          <li><strong>Training data scaling:</strong> More diverse and comprehensive training data</li>
        </ul>
      </section>
      
      <section className="tutorial-section">
        <h2>Key Advantages</h2>
        <ul>
          <li><strong>Parallelization:</strong> All positions processed simultaneously during training</li>
          <li><strong>Long-range dependencies:</strong> Attention directly connects any positions</li>
          <li><strong>Compositionality:</strong> Can learn hierarchical representations</li>
          <li><strong>Scalability:</strong> Performance improves predictably with size</li>
        </ul>
      </section>
      
      <div className="tutorial-navigation">
        <button className="tutorial-nav-button">Previous: Introduction to LLMs</button>
        <button className="tutorial-nav-button">Next: Self-Attention Mechanism</button>
      </div>
    </div>
  );
};

export default TransformerArchitectureTutorial; 