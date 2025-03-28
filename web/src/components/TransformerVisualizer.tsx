import React, { useState, useEffect } from 'react';
import './TransformerVisualizer.css';

interface TransformerVisualizerProps {
  inputText: string;
  modelType?: 'decoder-only' | 'encoder-only' | 'encoder-decoder';
  onUpdate?: (output: string) => void;
}

const TransformerVisualizer: React.FC<TransformerVisualizerProps> = ({ 
  inputText, 
  modelType = 'decoder-only',
  onUpdate 
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationSpeed, setAnimationSpeed] = useState(1000); // ms per step
  const [error, setError] = useState<string | null>(null);

  // Validate input
  useEffect(() => {
    if (!inputText || inputText.trim() === '') {
      setError('Please enter some text to visualize');
    } else {
      setError(null);
    }
  }, [inputText]);

  const steps = [
    { name: 'Input Tokenization', description: 'Converting input text into tokens' },
    { name: 'Token Embeddings', description: 'Converting tokens into vector representations' },
    { name: 'Position Embeddings', description: 'Adding position information to embeddings' },
    { name: 'Self-Attention', description: 'Computing attention scores between tokens' },
    { name: 'Attention Weights', description: 'Applying attention weights to value vectors' },
    { name: 'Feed Forward', description: 'Processing through feed-forward network' },
    { name: 'Output Generation', description: 'Generating next token prediction' }
  ];

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isAnimating && currentStep < steps.length) {
      interval = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= steps.length - 1) {
            setIsAnimating(false);
            return prev;
          }
          return prev + 1;
        });
      }, animationSpeed);
    }
    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isAnimating, currentStep, animationSpeed, steps.length]);

  const handlePlayPause = () => {
    if (error) return;
    setIsAnimating(!isAnimating);
  };

  const handleReset = () => {
    setCurrentStep(0);
    setIsAnimating(false);
  };

  const handleSpeedChange = (speed: number) => {
    setAnimationSpeed(speed);
  };

  if (error) {
    return (
      <div className="transformer-visualizer">
        <div className="error-message">{error}</div>
      </div>
    );
  }

  return (
    <div className="transformer-visualizer">
      <div className="visualization-controls">
        <button onClick={handlePlayPause}>
          {isAnimating ? 'Pause' : 'Play'} Animation
        </button>
        <button onClick={handleReset}>Reset</button>
        <div className="speed-control">
          <label>Animation Speed:</label>
          <select 
            value={animationSpeed} 
            onChange={(e) => handleSpeedChange(Number(e.target.value))}
          >
            <option value={500}>Fast</option>
            <option value={1000}>Normal</option>
            <option value={2000}>Slow</option>
          </select>
        </div>
      </div>

      <div className="visualization-container">
        <div className="transformer-architecture">
          {/* Input Layer */}
          <div className={`layer input-layer ${currentStep >= 0 ? 'active' : ''}`}>
            <div className="layer-title">Input</div>
            <div className="tokens">
              {inputText.split(' ').map((token, i) => (
                <div key={i} className="token">{token}</div>
              ))}
            </div>
          </div>

          {/* Embedding Layer */}
          <div className={`layer embedding-layer ${currentStep >= 1 ? 'active' : ''}`}>
            <div className="layer-title">Embeddings</div>
            <div className="embeddings">
              {inputText.split(' ').map((_, i) => (
                <div key={i} className="embedding-vector" />
              ))}
            </div>
          </div>

          {/* Position Embeddings */}
          <div className={`layer position-layer ${currentStep >= 2 ? 'active' : ''}`}>
            <div className="layer-title">Position</div>
            <div className="position-embeddings">
              {inputText.split(' ').map((_, i) => (
                <div key={i} className="position-vector" />
              ))}
            </div>
          </div>

          {/* Self-Attention Layer */}
          <div className={`layer attention-layer ${currentStep >= 3 ? 'active' : ''}`}>
            <div className="layer-title">Self-Attention</div>
            <div className="attention-matrix">
              {inputText.split(' ').map((_, i) => (
                <div key={i} className="attention-row">
                  {inputText.split(' ').map((_, j) => (
                    <div key={j} className="attention-cell" />
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Feed Forward Layer */}
          <div className={`layer ff-layer ${currentStep >= 5 ? 'active' : ''}`}>
            <div className="layer-title">Feed Forward</div>
            <div className="ff-network">
              {inputText.split(' ').map((_, i) => (
                <div key={i} className="ff-unit" />
              ))}
            </div>
          </div>

          {/* Output Layer */}
          <div className={`layer output-layer ${currentStep >= 6 ? 'active' : ''}`}>
            <div className="layer-title">Output</div>
            <div className="output-tokens">
              {inputText.split(' ').map((_, i) => (
                <div key={i} className="output-token" />
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="step-description">
        <h3>{steps[currentStep].name}</h3>
        <p>{steps[currentStep].description}</p>
      </div>

      <div className="progress-bar">
        <div 
          className="progress-fill"
          style={{ width: `${(currentStep / (steps.length - 1)) * 100}%` }}
        />
      </div>
    </div>
  );
};

export default TransformerVisualizer; 