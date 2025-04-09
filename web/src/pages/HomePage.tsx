import React from 'react';
import { Link } from 'react-router-dom';
import './HomePage.css';

const HomePage: React.FC = () => {
  return (
    <div className="home-page">
      <section className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">BinaryLM</h1>
          <p className="hero-tagline">Educational platform for large language models</p>
          <div className="hero-buttons">
            <Link to="/model-training" className="btn btn-primary">Start Training</Link>
            <a href="https://github.com/oxnr/BinaryLM" target="_blank" rel="noopener noreferrer" className="btn btn-secondary">GitHub</a>
          </div>
        </div>
      </section>

      <section className="process-section">
        <h2 className="section-title">The End-to-End LLM Process</h2>
        <p className="section-description">
          Understanding language models requires seeing how all components work together. Follow the complete pipeline from raw text to generated responses.
        </p>
        
        <div className="process-flow">
          <Link to="#" className="process-step-link">
            <div className="process-step">
              <div className="step-content">
                <h3>
                  <span className="step-number">1</span>
                  Data Collection
                </h3>
                <p>Large datasets of text are gathered from books, websites, and other sources to train the model.</p>
              </div>
            </div>
          </Link>
          
          <Link to="/tokenization" className="process-step-link">
            <div className="process-step">
              <div className="step-content">
                <h3>
                  <span className="step-number">2</span>
                  Tokenization
                </h3>
                <p>Raw text is split into tokens (words, subwords, or characters) that the model can process.</p>
              </div>
            </div>
          </Link>
          
          <Link to="/model-architecture" className="process-step-link">
            <div className="process-step">
              <div className="step-content">
                <h3>
                  <span className="step-number">3</span>
                  Model Architecture
                </h3>
                <p>Transformer-based neural networks process token embeddings through attention mechanisms.</p>
              </div>
            </div>
          </Link>
          
          <Link to="/model-training" className="process-step-link">
            <div className="process-step">
              <div className="step-content">
                <h3>
                  <span className="step-number">4</span>
                  Training
                </h3>
                <p>Model learns by predicting the next token in sequences, adjusting weights through backpropagation.</p>
              </div>
            </div>
          </Link>
          
          <Link to="/inference" className="process-step-link">
            <div className="process-step">
              <div className="step-content">
                <h3>
                  <span className="step-number">5</span>
                  Inference
                </h3>
                <p>Trained model generates responses by predicting tokens one at a time based on input prompts.</p>
              </div>
            </div>
          </Link>
        </div>
        
        <div className="process-connections">
          <h3>How It All Connects</h3>
          
          <div className="connection-diagram">
            <div className="connection-flow">
              <div className="connection-node">
                <div className="node-label">Raw Text Data</div>
              </div>
              
              <div className="connection-arrow">
                <span>Tokenization</span>
              </div>
              
              <div className="connection-node">
                <div className="node-label">Token Embeddings</div>
              </div>
              
              <div className="connection-arrow">
                <span>Processing</span>
              </div>
              
              <div className="connection-node">
                <div className="node-label">Neural Network</div>
              </div>
              
              <div className="connection-arrow">
                <span>Learning</span>
              </div>
              
              <div className="connection-node">
                <div className="node-label">Trained Model</div>
              </div>
              
              <div className="connection-arrow">
                <span>Predicting</span>
              </div>
              
              <div className="connection-node">
                <div className="node-label">Generated Text</div>
              </div>
            </div>
            
            <div className="connection-details">
              <p>
                Language models transform text into numerical representations (tokens), process them through transformer layers
                with attention mechanisms, and generate output tokens that are converted back to human-readable text.
              </p>
              <p>
                Each component builds on the previous, creating a pipeline where the quality of each step affects the
                final output. By understanding these connections, you can optimize each part of the process.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="features-section">
        <h2 className="section-title">Understand how LLMs work</h2>
        
        <div className="features-grid">
          <Link to="/tokenization" className="feature-card-link">
            <div className="feature-card">
              <h3>Tokenization</h3>
              <p>Visualize how text is broken down into tokens that models can understand.</p>
            </div>
          </Link>
          
          <Link to="/model-architecture" className="feature-card-link">
            <div className="feature-card">
              <h3>Model Architecture</h3>
              <p>Interactive visualizations of transformer architecture and attention mechanisms.</p>
            </div>
          </Link>
          
          <Link to="/model-training" className="feature-card-link">
            <div className="feature-card">
              <h3>Training Process</h3>
              <p>Understand the training process and how models learn patterns from data.</p>
            </div>
          </Link>
          
          <Link to="/inference" className="feature-card-link">
            <div className="feature-card">
              <h3>Inference</h3>
              <p>Generate completions with your trained models and visualize the process.</p>
            </div>
          </Link>
        </div>
      </section>
    </div>
  );
};

export default HomePage; 