import React, { useState } from 'react';
import './InferencePage.css';

const InferencePage: React.FC = () => {
  const [prompt, setPrompt] = useState<string>('');
  const [response, setResponse] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [selectedModel, setSelectedModel] = useState<string>('binarylm-small');
  const [temperature, setTemperature] = useState<number>(0.7);
  const [maxTokens, setMaxTokens] = useState<number>(512);
  const [showParams, setShowParams] = useState<boolean>(false);

  // The available models - would be fetched from an API in a real implementation
  const availableModels = [
    { id: 'binarylm-small', name: 'BinaryLM Small (2B)', description: 'Fast, lightweight model for basic text generation' },
    { id: 'binarylm-medium', name: 'BinaryLM Medium (7B)', description: 'Balanced performance and quality' },
    { id: 'binarylm-large', name: 'BinaryLM Large (13B)', description: 'High-quality responses for complex tasks' },
  ];

  const handleGenerate = () => {
    if (!prompt.trim()) return;
    
    setIsGenerating(true);
    setResponse('');
    
    // Simulated streaming response for demo purposes
    let generatedText = '';
    const sampleResponse = "This is a simulated response from the model. In a real application, this would connect to an API that interfaces with the language model. The model would receive the tokenized version of your prompt, run inference through its transformer layers, and generate a response token by token.\n\nEach token is selected based on the probability distribution output by the model, modified by the temperature parameter. Higher temperature (>1.0) results in more random responses, while lower values (<1.0) make the model more deterministic and focused.\n\nThe maximum tokens parameter limits how long the generated text can be. This is important because transformer models have a fixed context window size.";
    const words = sampleResponse.split(' ');
    
    let i = 0;
    const intervalId = setInterval(() => {
      if (i < words.length) {
        generatedText += ' ' + words[i];
        setResponse(generatedText);
        i++;
      } else {
        clearInterval(intervalId);
        setIsGenerating(false);
      }
    }, 50);
  };

  return (
    <div className="inference-page">
      <h1>Model Inference</h1>
      <p className="page-description">
        Test the language model by providing a prompt and generating text. This simulates how an LLM processes your input and generates responses.
      </p>
      
      <div className="model-selection">
        <h3>Select Model</h3>
        <div className="model-options">
          {availableModels.map((model) => (
            <div
              key={model.id}
              className={`model-option ${selectedModel === model.id ? 'selected' : ''}`}
              onClick={() => setSelectedModel(model.id)}
            >
              <h4>{model.name}</h4>
              <p>{model.description}</p>
            </div>
          ))}
        </div>
      </div>
      
      <div className="inference-controls">
        <div className="generation-params">
          <button 
            className="params-toggle" 
            onClick={() => setShowParams(!showParams)}
          >
            {showParams ? 'Hide Parameters' : 'Show Parameters'}
          </button>
          
          {showParams && (
            <div className="params-controls">
              <div className="param-control">
                <label htmlFor="temperature">Temperature: {temperature}</label>
                <input
                  id="temperature"
                  type="range"
                  min="0.1"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                />
                <span className="param-hint">Higher = more creative, Lower = more focused</span>
              </div>
              
              <div className="param-control">
                <label htmlFor="max-tokens">Max Tokens: {maxTokens}</label>
                <input
                  id="max-tokens"
                  type="range"
                  min="64"
                  max="2048"
                  step="64"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                />
                <span className="param-hint">Maximum length of generated text</span>
              </div>
            </div>
          )}
        </div>
        
        <div className="prompt-input">
          <label htmlFor="prompt">Enter your prompt:</label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Type a prompt to generate text..."
            rows={5}
          />
        </div>
        
        <button
          className="generate-button"
          onClick={handleGenerate}
          disabled={isGenerating || !prompt.trim()}
        >
          {isGenerating ? 'Generating...' : 'Generate'}
        </button>
      </div>
      
      {(response || isGenerating) && (
        <div className="model-response">
          <h3>Model Response</h3>
          <div className="response-content">
            <p>{response}</p>
            {isGenerating && <span className="typing-indicator">▌</span>}
          </div>
        </div>
      )}
      
      <div className="inference-explainer">
        <h3>How Inference Works</h3>
        <div className="inference-steps">
          <div className="inference-step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h4>Tokenization</h4>
              <p>Your prompt is converted to token IDs using the model's vocabulary</p>
            </div>
          </div>
          
          <div className="inference-step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h4>Token Embedding</h4>
              <p>Each token ID is converted to a vector representation</p>
            </div>
          </div>
          
          <div className="inference-step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h4>Attention Layers</h4>
              <p>Token vectors flow through multiple transformer layers</p>
            </div>
          </div>
          
          <div className="inference-step">
            <div className="step-number">4</div>
            <div className="step-content">
              <h4>Next Token Prediction</h4>
              <p>Model predicts the most likely next token based on context</p>
            </div>
          </div>
          
          <div className="inference-step">
            <div className="step-number">5</div>
            <div className="step-content">
              <h4>Token Selection</h4>
              <p>A token is chosen based on its probability and the temperature</p>
            </div>
          </div>
          
          <div className="inference-step">
            <div className="step-number">6</div>
            <div className="step-content">
              <h4>Auto-regressive Generation</h4>
              <p>The process repeats, adding the new token to context</p>
            </div>
          </div>
        </div>
        
        <div className="github-link">
          <p>
            Want to learn more or contribute? Visit the project on 
            <a href="https://github.com/oxnr/BinaryLM" target="_blank" rel="noopener noreferrer">
              GitHub
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default InferencePage; 