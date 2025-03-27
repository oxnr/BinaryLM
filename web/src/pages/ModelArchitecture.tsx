import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import TransformerVisualizer from '../components/TransformerVisualizer';
import './ModelArchitecture.css';

const ModelArchitecture: React.FC = () => {
  const [modelType, setModelType] = useState<'decoder-only' | 'encoder-only' | 'encoder-decoder'>('decoder-only');
  const [exampleText, setExampleText] = useState<string>('The transformer architecture is revolutionary');

  // Pre-defined architecture descriptions
  const architectureDescriptions = {
    'decoder-only': {
      name: 'Decoder-Only',
      examples: 'GPT, LLaMA, Claude',
      description: 'Used primarily for text generation. These models process tokens sequentially and generate text one token at a time.',
      strengths: 'Excellent for creative text generation, chat applications, and completing prompts.',
    },
    'encoder-only': {
      name: 'Encoder-Only',
      examples: 'BERT, RoBERTa',
      description: 'Used for understanding and extracting meaning from text. These models can see all tokens at once.',
      strengths: 'Great for classification, sentiment analysis, and understanding input context.',
    },
    'encoder-decoder': {
      name: 'Encoder-Decoder',
      examples: 'T5, BART',
      description: 'Combines both architectures. The encoder processes the input, and the decoder generates the output.',
      strengths: 'Ideal for translation, summarization, and question-answering tasks.',
    }
  };

  const currentArchitecture = architectureDescriptions[modelType];

  return (
    <div>
      <h1>Model Architecture Explorer</h1>
      
      <div className="architecture-selector-container">
        <div className="architecture-options">
          {Object.entries(architectureDescriptions).map(([type, details]) => (
            <div 
              key={type} 
              className={`architecture-option ${modelType === type ? 'selected' : ''}`}
              onClick={() => setModelType(type as any)}
            >
              <h3>{details.name}</h3>
              <div className="architecture-examples">{details.examples}</div>
            </div>
          ))}
        </div>
        
        <div className="selected-architecture-details">
          <h2>{currentArchitecture.name} Architecture</h2>
          <p className="architecture-description">{currentArchitecture.description}</p>
          <div className="architecture-strengths">
            <strong>Strengths:</strong> {currentArchitecture.strengths}
          </div>
        </div>
      </div>
      
      <div className="input-text-control card">
        <label htmlFor="example-text">Input text for visualization:</label>
        <input 
          type="text" 
          id="example-text" 
          value={exampleText} 
          onChange={(e) => setExampleText(e.target.value)}
          className="text-input"
          placeholder="Enter text to visualize the model's processing..."
        />
      </div>

      <div className="card">
        <TransformerVisualizer 
          modelType={modelType} 
          inputText={exampleText}
        />
      </div>

      <div className="attention-explainer-card">
        <h2>Understanding Attention Mechanism</h2>
        
        <div className="attention-explainer-content">
          <div className="attention-intro">
            <p>
              The attention mechanism is the core innovation of transformer models, allowing them to understand 
              relationships between words regardless of their distance from each other.
            </p>
          </div>
          
          <div className="attention-process-visual">
            <div className="process-step">
              <div className="step-number">01</div>
              <div className="step-content">
                <h3>Input Projection</h3>
                <p>Each token is projected into Query (Q), Key (K), and Value (V) vectors</p>
                <div className="step-visualization">
                  <div className="token-representation">Token</div>
                  <div className="projection-arrows">
                    <div className="arrow-line">→ Q</div>
                    <div className="arrow-line">→ K</div>
                    <div className="arrow-line">→ V</div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="process-step">
              <div className="step-number">02</div>
              <div className="step-content">
                <h3>Attention Scoring</h3>
                <p>Query and Key vectors are used to calculate attention scores</p>
                <div className="step-formula">Score = Q·K<sup>T</sup> / √d<sub>k</sub></div>
              </div>
            </div>
            
            <div className="process-step">
              <div className="step-number">03</div>
              <div className="step-content">
                <h3>Softmax Distribution</h3>
                <p>Scores are converted to probabilities with softmax</p>
                <div className="step-visualization">
                  <div className="attention-distribution">
                    {[0.1, 0.05, 0.7, 0.15].map((weight, i) => (
                      <div 
                        key={i} 
                        className="attention-weight"
                        style={{height: `${weight * 100}px`}}
                        title={`Weight: ${weight}`}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </div>
            
            <div className="process-step">
              <div className="step-number">04</div>
              <div className="step-content">
                <h3>Value Weighting</h3>
                <p>Value vectors are weighted by attention scores and summed</p>
                <div className="step-visualization">
                  <div className="weighted-values">
                    <div className="value-vector v1">V1 × 0.1</div>
                    <div className="value-vector v2">V2 × 0.05</div>
                    <div className="value-vector v3">V3 × 0.7</div>
                    <div className="value-vector v4">V4 × 0.15</div>
                    <div className="sum-symbol">↓</div>
                    <div className="output-vector">Output</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="attention-key-insight">
            <div className="insight-header">Why Attention Matters</div>
            <p>
              Attention allows models to selectively focus on relevant parts of input, creating direct connections between words 
              regardless of distance. For example, in "The cat, which was very old, sat on the mat," attention helps connect "cat" and "sat" 
              despite the words between them.
            </p>
            <p>
              Each attention head can learn different linguistic patterns such as syntax, semantics, entity relationships, or co-reference resolution.
            </p>
          </div>
          
          <div className="multi-head-explanation">
            <h3>Multi-Head Attention</h3>
            <p>
              Transformers use multiple attention heads in parallel, each learning different relationship patterns in the data.
              The outputs from all heads are concatenated and projected to form the final output.
            </p>
            <div className="multi-head-visual">
              {[1, 2, 3, 4].map(head => (
                <div key={head} className={`attention-head head-${head}`}>
                  Head {head}
                </div>
              ))}
              <div className="concat-arrow">⟶</div>
              <div className="concatenated-output">Combined Output</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelArchitecture; 