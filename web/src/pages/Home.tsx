import React from 'react';
import { Link } from 'react-router-dom';

const Home: React.FC = () => {
  return (
    <div>
      <h1>Welcome to BinaryLM Explorer</h1>
      
      <div className="card">
        <h2>Learn How LLMs Work</h2>
        <p>
          This interactive platform helps you understand the inner workings of Large Language Models (LLMs)
          through visualizations, explanations, and interactive examples.
        </p>
      </div>

      <div className="card">
        <h2>Explore the Components</h2>
        <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '20px' }}>
          <div style={{ flex: 1, minWidth: '250px' }}>
            <h3>Tokenization</h3>
            <p>Understand how text is broken down into tokens, the fundamental unit of LLM processing.</p>
            <Link to="/tokenizer">
              <button>Explore Tokenization</button>
            </Link>
          </div>

          <div style={{ flex: 1, minWidth: '250px' }}>
            <h3>Model Architecture</h3>
            <p>Visualize the transformer architecture and how each component processes information.</p>
            <Link to="/model">
              <button>Explore Architecture</button>
            </Link>
          </div>

          <div style={{ flex: 1, minWidth: '250px' }}>
            <h3>Training Process</h3>
            <p>See how models learn from data and improve over time with custom training.</p>
            <Link to="/training">
              <button>Explore Training</button>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home; 