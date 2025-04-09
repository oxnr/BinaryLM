import React, { useState, useEffect, useRef } from 'react';
import PredictionVisualizer from '../components/PredictionVisualizer';
import './InferencePage.css';

interface TokenPrediction {
  token: string;
  probability: number;
}

interface TokenStep {
  token: string;
  predictions: TokenPrediction[];
}

// Fixed demonstration data
const DEMO_PROMPT = "How do language models learn?";
const DEMO_RESPONSE = "Language models learn through reinforcement learning techniques where the model parameters are iteratively adjusted to maximize prediction accuracy. This process involves training on vast corpora of text data, where each token prediction contributes to updating the weights of the neural network. By using reward signals to guide parameter tuning, the model gradually improves its ability to generate coherent and contextually appropriate text.";

// Token vocabulary for generating random alternatives
const VOCABULARY = [
  'the', 'of', 'and', 'to', 'in', 'a', 'is', 'that', 'for', 'it', 
  'with', 'as', 'was', 'be', 'by', 'on', 'not', 'this', 'but', 'from',
  'are', 'or', 'an', 'they', 'which', 'you', 'their', 'has', 'have', 'had',
  'its', 'at', 'been', 'if', 'more', 'when', 'will', 'would', 'who', 'what',
  'language', 'model', 'transformer', 'neural', 'network', 'learning', 'deep',
  'attention', 'embedding', 'token', 'layer', 'training', 'reinforcement',
  'parameter', 'gradient', 'backpropagation', 'optimization', 'loss', 'function',
  'algorithm', 'sequence', 'prediction', 'probability', 'distribution', 'sampling',
  'generation', 'fine-tuning', 'weight', 'bias', 'activation', 'tensor', 'vector'
];

const InferencePage: React.FC = () => {
  const [generatedText, setGeneratedText] = useState<string>('');
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [isDemoRunning, setIsDemoRunning] = useState<boolean>(false);
  const [currentPredictions, setCurrentPredictions] = useState<TokenPrediction[]>([]);
  const [selectedToken, setSelectedToken] = useState<string>('');
  const [temperature, setTemperature] = useState<number>(0.7);
  const [generationSpeed, setGenerationSpeed] = useState<number>(2000); // ms between tokens
  const [showSummary, setShowSummary] = useState<boolean>(false);
  const [generationHistory, setGenerationHistory] = useState<TokenStep[]>([]);
  
  // Add ref for tracking running state
  const isRunningRef = useRef<boolean>(false);
  
  // Add cleanup effect
  useEffect(() => {
    return () => {
      // Cleanup when component unmounts
      isRunningRef.current = false;
      setIsDemoRunning(false);
      setIsGenerating(false);
    };
  }, []);
  
  // Split the completion into tokens (words in this simplified case)
  const completionTokens = DEMO_RESPONSE.split(/\s+/);
  
  // Function to generate realistic-looking token probabilities
  const generateTokenPredictions = (correctToken: string, step: number): TokenPrediction[] => {
    try {
      const predictions: TokenPrediction[] = [];
      
      // Add the correct token with high probability
      // Earlier tokens are more "certain" than later ones
      const certaintyFactor = Math.max(0.55, 0.9 - (step * 0.01));
      const correctProb = certaintyFactor + (Math.random() * 0.15);
      
      predictions.push({
        token: correctToken,
        probability: correctProb
      });
      
      // Generate 8-15 alternative predictions
      const numAlternatives = 8 + Math.floor(Math.random() * 8);
      // Instead of distributing remaining probability (1-correctProb),
      // we'll use a value that will make the total exceed 100%
      const totalAlternativeProbability = 0.8; // This will make total exceed 100%
      
      for (let i = 0; i < numAlternatives; i++) {
        // Randomly sample from vocabulary, making sure not to duplicate
        let randomToken: string;
        let attempts = 0;
        const maxAttempts = 100;
        
        do {
          randomToken = VOCABULARY[Math.floor(Math.random() * VOCABULARY.length)];
          attempts++;
          if (attempts > maxAttempts) {
            console.warn('Could not find unique token after max attempts');
            break;
          }
        } while (predictions.some(p => p.token === randomToken));
        
        if (attempts <= maxAttempts) {
          // Distribute probabilities using a decay function
          // Earlier alternatives get higher probabilities
          const decayFactor = 0.7 ** i;
          let altProb = (totalAlternativeProbability * decayFactor) / numAlternatives;
          
          // Small random variation
          altProb *= (0.8 + Math.random() * 0.4);
          
          predictions.push({
            token: randomToken,
            probability: altProb
          });
        }
      }
      
      // We're intentionally NOT normalizing probabilities to let them exceed 100%
      return predictions.sort((a, b) => b.probability - a.probability);
    } catch (error) {
      console.error('Error generating token predictions:', error);
      // Return a simple prediction with just the correct token
      return [{
        token: correctToken,
        probability: 1.0
      }];
    }
  };
  
  // Demonstration of the generation process
  const runDemo = async () => {
    console.log("runDemo started");
    
    // Set states for running
    isRunningRef.current = true;
    setIsGenerating(true);
    setIsDemoRunning(true);
    setCurrentStep(0);
    setShowSummary(false);
    setGenerationHistory([]);
    
    // Start with an empty string
    setGeneratedText('');
    let builtText = '';
    let lastTokenTime = performance.now();
    let currentTokenIndex = 0;
    let animationFrameId: number;
    
    console.log("Starting token generation loop, tokens:", completionTokens.length);
    
    const generateNextToken = (timestamp: number) => {
      if (!isRunningRef.current) {
        console.log("Demo stopped during generation");
        setIsGenerating(false);
        setIsDemoRunning(false);
        if (animationFrameId) {
          cancelAnimationFrame(animationFrameId);
        }
        return;
      }

      const timeSinceLastToken = timestamp - lastTokenTime;
      console.log(`Time since last token: ${timeSinceLastToken}ms, Required: ${generationSpeed}ms`);

      if (timeSinceLastToken >= generationSpeed) {
        // Get the current token
        const currentToken = completionTokens[currentTokenIndex];
        console.log(`Generating token ${currentTokenIndex + 1}: "${currentToken}"`);
        
        // Set the current step (1-based for display)
        setCurrentStep(currentTokenIndex + 1);
        
        // First, generate predictions for this token
        const predictions = generateTokenPredictions(currentToken, currentTokenIndex);
        console.log(`Generated ${predictions.length} predictions for token "${currentToken}"`);
        
        // Add to built text
        if (currentTokenIndex === 0) {
          builtText = currentToken;
        } else {
          builtText += ' ' + currentToken;
        }
        
        // Display the text with the new token
        setGeneratedText(builtText);
        console.log(`Updated generated text (${builtText.length} chars)`);
        
        // Set the predictions and selected token
        setCurrentPredictions(predictions);
        setSelectedToken(currentToken);
        
        // Add this step to history
        setGenerationHistory(prev => [...prev, {
          token: currentToken,
          predictions: predictions
        }]);

        lastTokenTime = timestamp;
        currentTokenIndex++;
      }

      if (currentTokenIndex < completionTokens.length && isRunningRef.current) {
        animationFrameId = requestAnimationFrame(generateNextToken);
      } else {
        // Generation complete
        console.log("Token generation complete");
        setCurrentStep(completionTokens.length);
        setShowSummary(true);
        setIsGenerating(false);
        setIsDemoRunning(false);
        isRunningRef.current = false;
      }
    };

    // Start the animation frame loop
    animationFrameId = requestAnimationFrame(generateNextToken);
    
    // Cleanup function
    return () => {
      isRunningRef.current = false;
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  };
  
  const handleStopGeneration = () => {
    isRunningRef.current = false;
    setIsDemoRunning(false);
    setIsGenerating(false);
  };
  
  const handleRestartDemo = () => {
    // Reset and restart the demo
    console.log("handleRestartDemo called");
    
    // Make sure we reset any existing state
    isRunningRef.current = false;
    setIsGenerating(false);
    setIsDemoRunning(false);
    setCurrentStep(0);
    setGeneratedText('');
    setShowSummary(false);
    setGenerationHistory([]);
    setCurrentPredictions([]);
    
    // Add debugging
    console.log("State reset complete");
    console.log("DEMO_RESPONSE length:", DEMO_RESPONSE.length);
    console.log("completionTokens:", completionTokens);
    
    // Start the demo immediately instead of using setTimeout
    try {
      console.log("Starting demo...");
      runDemo();
      console.log("Demo started successfully");
    } catch (error) {
      console.error("Error starting demo:", error);
    }
  };
  
  const handleToggleSummary = () => {
    setShowSummary(!showSummary);
  };
  
  // Handle manual token selection
  const handleTokenClick = (token: string) => {
    if (!isGenerating) return;
    
    // Update the selected token
    setSelectedToken(token);
    
    // In a real implementation, this would change the generation path
    console.log(`Token selected: ${token}`);
  };
  
  return (
    <div className="inference-page">
      <header>
        <h1>BinaryLM Inference</h1>
        <p className="subtitle">
          See how the model generates text one token at a time, with probability distributions for each step.
        </p>
      </header>
      
      <div className="inference-container">
        <div className="inference-controls">
          <div className="control-group">
            <label htmlFor="prompt">Demo Prompt:</label>
            <div className="fixed-prompt">{DEMO_PROMPT}</div>
            <p className="demo-explanation">
              This demonstration shows how a language model generates text one token at a time,
              with probabilities for each possible next token.
            </p>
          </div>
          
          <div className="control-row">
            <div className="control-group">
              <label htmlFor="temperature">Temperature:</label>
              <input
                id="temperature"
                type="range"
                min="0.1"
                max="1.5"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                disabled={isGenerating}
              />
              <span className="param-value">{temperature.toFixed(1)}</span>
              <p className="param-explanation">
                Controls randomness: higher values produce more diverse outputs.
              </p>
            </div>
            
            <div className="control-group">
              <label htmlFor="speed">Generation Speed:</label>
              <input
                id="speed"
                type="range"
                min="500"
                max="3000"
                step="100"
                value={generationSpeed}
                onChange={(e) => setGenerationSpeed(parseInt(e.target.value))}
                disabled={isGenerating}
              />
              <span className="param-value">{(generationSpeed / 1000).toFixed(1)}s</span>
              <p className="param-explanation">
                Time between token generations (slower = easier to follow).
              </p>
            </div>
          </div>
          
          <div className="button-group">
            {!isGenerating ? (
              <>
                <button 
                  className="generate-button"
                  onClick={handleRestartDemo}
                >
                  {currentStep === 0 ? 'Start Demo' : 'Restart Demo'}
                </button>
                {generationHistory.length > 0 && (
                  <button 
                    className="summary-button"
                    onClick={handleToggleSummary}
                  >
                    {showSummary ? 'Show Latest Prediction' : 'Show Summary'}
                  </button>
                )}
              </>
            ) : (
              <button 
                className="stop-button"
                onClick={handleStopGeneration}
              >
                Stop Demo
              </button>
            )}
          </div>
        </div>
        
        <div className="generation-results">
          <h2>Generated Text <span className="step-counter">Token: {currentStep}/{completionTokens.length}</span></h2>
          
          <div className="prompt-display">
            <strong>Prompt:</strong> {DEMO_PROMPT}
          </div>
          
          <div className="current-token-label">Generated Text:</div>
          <div className="generated-content">
            {generatedText || <span className="placeholder">Start the demo to see generation in action...</span>}
            {isGenerating && <span className="cursor"></span>}
          </div>
          
          <div className="distribution-heading">
            {currentPredictions.length > 0 && (
              <div className="current-distribution-info">
                <span>Current token: <strong>{selectedToken}</strong></span>
                <span className="distribution-subtitle">Showing prediction distribution for this token</span>
              </div>
            )}
          </div>
          
          <div className="predictions-section">
            {showSummary && generationHistory.length > 0 ? (
              <div className="summary-view">
                <h3>Token Generation Summary</h3>
                <div className="summary-scroll">
                  {generationHistory.map((step, index) => (
                    <div key={index} className="summary-item">
                      <div className="summary-token">Token #{index + 1}: <strong>{step.token}</strong></div>
                      <div className="summary-distribution">
                        <PredictionVisualizer
                          predictions={step.predictions}
                          selectedToken={step.token}
                          maxTokensToShow={5}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <PredictionVisualizer
                predictions={currentPredictions}
                selectedToken={selectedToken}
                onTokenClick={handleTokenClick}
                maxTokensToShow={12}
              />
            )}
          </div>
        </div>
      </div>
      
      <div className="inference-explanation">
        <h3>How It Works</h3>
        <p>
          This visualization demonstrates how a language model generates text. At each step, the model predicts 
          probabilities for the next token based on the context, and then selects one token (typically using 
          temperature sampling). The bars above show the top token candidates and their relative probabilities 
          at the current step.
        </p>
        <p>
          <strong>Reinforcement Learning in LLMs:</strong> The model's parameters are tuned through 
          reinforcement learning techniques to maximize the likelihood of generating high-quality, 
          accurate predictions. This process involves optimizing the model to predict tokens that 
          lead to better outcomes based on various reward signals.
        </p>
        <p>
          In production models, the visualization might differ slightly as real models work at the token level, 
          not word level, and can include special tokens and subword units. The token probabilities often 
          exceed 100% in total because they represent the model's confidence in each possible continuation, 
          before normalization.
        </p>
      </div>
    </div>
  );
};

export default InferencePage; 