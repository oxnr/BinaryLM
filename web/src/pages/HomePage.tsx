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
          <div className="process-step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h3>Data Collection</h3>
              <p>Large datasets of text are gathered from books, websites, and other sources to train the model.</p>
              <div className="step-icon">
                <svg viewBox="0 0 24 24" width="36" height="36" stroke="currentColor" fill="none">
                  <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z" />
                </svg>
              </div>
            </div>
          </div>
          
          <div className="flow-arrow">→</div>
          
          <div className="process-step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h3>Tokenization</h3>
              <p>Raw text is split into tokens (words, subwords, or characters) that the model can process.</p>
              <Link to="/tokenization" className="step-link">Explore Tokenization</Link>
              <div className="step-icon">
                <svg viewBox="0 0 24 24" width="36" height="36" stroke="currentColor" fill="none">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="17 8 12 3 7 8"></polyline>
                  <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
              </div>
            </div>
          </div>
          
          <div className="flow-arrow">→</div>
          
          <div className="process-step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h3>Model Architecture</h3>
              <p>Transformer-based neural networks process token embeddings through attention mechanisms.</p>
              <Link to="/model-architecture" className="step-link">Explore Architecture</Link>
              <div className="step-icon">
                <svg viewBox="0 0 24 24" width="36" height="36" stroke="currentColor" fill="none">
                  <rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect>
                  <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path>
                </svg>
              </div>
            </div>
          </div>
          
          <div className="flow-arrow">→</div>
          
          <div className="process-step">
            <div className="step-number">4</div>
            <div className="step-content">
              <h3>Training</h3>
              <p>Model learns by predicting the next token in sequences, adjusting weights through backpropagation.</p>
              <Link to="/model-training" className="step-link">Explore Training</Link>
              <div className="step-icon">
                <svg viewBox="0 0 24 24" width="36" height="36" stroke="currentColor" fill="none">
                  <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path>
                </svg>
              </div>
            </div>
          </div>
          
          <div className="flow-arrow">→</div>
          
          <div className="process-step">
            <div className="step-number">5</div>
            <div className="step-content">
              <h3>Inference</h3>
              <p>Trained model generates responses by predicting tokens one at a time based on input prompts.</p>
              <Link to="/inference" className="step-link">Try Inference</Link>
              <div className="step-icon">
                <svg viewBox="0 0 24 24" width="36" height="36" stroke="currentColor" fill="none">
                  <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon>
                </svg>
              </div>
            </div>
          </div>
        </div>
        
        <div className="process-connections">
          <h3>How It All Connects</h3>
          
          <div className="connection-diagram">
            <div className="connection-flow">
              <div className="connection-node">
                <div className="node-icon">📚</div>
                <div className="node-label">Raw Text Data</div>
              </div>
              
              <div className="connection-arrow">
                <span>Tokenization</span>
                <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" fill="none">
                  <line x1="5" y1="12" x2="19" y2="12"></line>
                  <polyline points="12 5 19 12 12 19"></polyline>
                </svg>
              </div>
              
              <div className="connection-node">
                <div className="node-icon">🔢</div>
                <div className="node-label">Token Embeddings</div>
              </div>
              
              <div className="connection-arrow">
                <span>Processing</span>
                <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" fill="none">
                  <line x1="5" y1="12" x2="19" y2="12"></line>
                  <polyline points="12 5 19 12 12 19"></polyline>
                </svg>
              </div>
              
              <div className="connection-node">
                <div className="node-icon">🧠</div>
                <div className="node-label">Neural Network</div>
              </div>
              
              <div className="connection-arrow">
                <span>Learning</span>
                <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" fill="none">
                  <line x1="5" y1="12" x2="19" y2="12"></line>
                  <polyline points="12 5 19 12 12 19"></polyline>
                </svg>
              </div>
              
              <div className="connection-node">
                <div className="node-icon">💡</div>
                <div className="node-label">Trained Model</div>
              </div>
              
              <div className="connection-arrow">
                <span>Predicting</span>
                <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" fill="none">
                  <line x1="5" y1="12" x2="19" y2="12"></line>
                  <polyline points="12 5 19 12 12 19"></polyline>
                </svg>
              </div>
              
              <div className="connection-node">
                <div className="node-icon">💬</div>
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
          <div className="feature-card">
            <h3>Tokenization</h3>
            <p>Visualize how text is broken down into tokens that models can understand.</p>
            <Link to="/tokenization" className="feature-link">Explore Tokenization →</Link>
          </div>
          
          <div className="feature-card">
            <h3>Model Architecture</h3>
            <p>Interactive visualizations of transformer architecture and attention mechanisms.</p>
            <Link to="/model-architecture" className="feature-link">Explore Architecture →</Link>
          </div>
          
          <div className="feature-card">
            <h3>Training Process</h3>
            <p>Understand the training process and how models learn patterns from data.</p>
            <Link to="/model-training" className="feature-link">Explore Training →</Link>
          </div>
          
          <div className="feature-card">
            <h3>Inference</h3>
            <p>Generate completions with your trained models and visualize the process.</p>
            <Link to="/inference" className="feature-link">Try Inference →</Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default HomePage; 