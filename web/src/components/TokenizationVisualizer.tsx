import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './TokenizationVisualizer.css';

// Mock data types
interface Token {
  text: string;
  id?: number;
  type: string;
  vector?: number[];
}

interface TokenStep {
  stage: string;
  tokens: Token[];
}

interface TokenizationVisualizerProps {
  text?: string;
}

// Generate mock tokenization data based on input text
const generateMockTokenizationData = (text: string): TokenStep[] => {
  // Split into raw characters
  const rawTokens: Token[] = text.split('').map(char => ({
    text: char,
    type: 'raw'
  }));
  
  // Split into words
  const wordTokens: Token[] = text.split(/\b/)
    .filter(word => word.trim().length > 0)
    .map(word => ({
      text: word,
      type: 'word'
    }));
  
  // Mock subword tokenization (simplified)
  const subwordTokens: Token[] = [];
  wordTokens.forEach(token => {
    if (token.text.length > 3) {
      // Split longer words into subwords
      const firstPart = token.text.substring(0, Math.ceil(token.text.length / 2));
      const secondPart = token.text.substring(Math.ceil(token.text.length / 2));
      
      subwordTokens.push({
        text: firstPart,
        type: 'subword-start',
        id: Math.floor(Math.random() * 10000)
      });
      
      subwordTokens.push({
        text: secondPart,
        type: 'subword-continuation',
        id: Math.floor(Math.random() * 10000)
      });
    } else {
      // Keep short words intact
      subwordTokens.push({
        text: token.text,
        type: 'subword-start',
        id: Math.floor(Math.random() * 10000)
      });
    }
  });
  
  // Generate mock vectors for tokens
  const finalTokens: Token[] = subwordTokens.map(token => ({
    ...token,
    vector: Array.from({ length: 5 }, () => (Math.random() * 2 - 1))
  }));
  
  return [
    {
      stage: "Raw Text",
      tokens: rawTokens
    },
    {
      stage: "Word Tokenization",
      tokens: wordTokens
    },
    {
      stage: "Subword Tokenization & IDs",
      tokens: subwordTokens
    },
    {
      stage: "Token Embedding Vectors",
      tokens: finalTokens
    }
  ];
};

const TokenizationVisualizer: React.FC<TokenizationVisualizerProps> = ({ text: initialText }) => {
  const [inputText, setInputText] = useState<string>(initialText || "The transformer architecture revolutionized natural language processing.");
  const [steps, setSteps] = useState<TokenStep[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [showExplanation, setShowExplanation] = useState<boolean>(false);

  const handleTokenize = () => {
    if (!inputText.trim()) return;
    
    setIsLoading(true);
    
    // Simulate API delay
    setTimeout(() => {
      const mockSteps = generateMockTokenizationData(inputText);
      setSteps(mockSteps);
      setIsLoading(false);
    }, 800);
  };

  // Generate initial tokenization data on component mount
  useEffect(() => {
    if (initialText) {
      handleTokenize();
    }
  }, [initialText, handleTokenize]);

  return (
    <div className="tokenization-visualizer">
      <h3>Tokenization Process</h3>
      
      <div className="tokenization-input">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Enter text to tokenize..."
          onKeyPress={(e) => e.key === 'Enter' && handleTokenize()}
        />
      </div>
      
      <button className="tokenize-button" onClick={handleTokenize}>
        Tokenize
      </button>
      
      <button 
        className="explanation-toggle-button" 
        onClick={() => setShowExplanation(!showExplanation)}
      >
        {showExplanation ? "Hide Explanation" : "What is Tokenization?"}
      </button>
      
      {showExplanation && (
        <div className="tokenization-explanation">
          <h4>What is Tokenization?</h4>
          <p>
            Tokenization is the <strong>first step</strong> in the LLM processing pipeline. It converts raw text into 
            numerical tokens that the model can understand.
          </p>
          
          <h4>Where It Fits in the LLM Process</h4>
          <ol>
            <li><strong>Tokenization</strong>: Text → Token IDs (what you see here)</li>
            <li><strong>Embedding</strong>: Token IDs → Vector representations</li>
            <li><strong>Model Processing</strong>: Vector processing through transformer layers</li>
            <li><strong>Prediction</strong>: Output vectors → Token probabilities</li>
            <li><strong>Detokenization</strong>: Token IDs → Text (for generating output)</li>
          </ol>
          
          <h4>Where Are Tokens Stored?</h4>
          <p>
            The vocabulary (mapping between words and token IDs) is stored separately from the model as a 
            <strong> vocabulary file</strong>. The model itself doesn't store the words—only their numeric representations.
            During preprocessing, the tokenizer converts text to these IDs using the vocabulary.
          </p>
          
          <h4>Important Facts About Tokenization</h4>
          <ul>
            <li>Tokens aren't always complete words; they can be subwords, characters, or byte pairs</li>
            <li>The vocabulary size is fixed (typically between 30K-100K tokens)</li>
            <li>Rare words get split into multiple subword tokens</li>
            <li>Tokenization affects the model's performance, especially for multilingual text</li>
            <li>Common tokenization methods include BPE (Byte-Pair Encoding), WordPiece, and SentencePiece</li>
          </ul>
        </div>
      )}
      
      {isLoading ? (
        <div className="loading">Processing...</div>
      ) : steps.length > 0 ? (
        <div className="tokenization-steps">
          {steps.map((step, stepIdx) => (
            <div key={stepIdx} className="tokenization-step">
              <h4>{step.stage}</h4>
              <div className="tokens-container">
                {step.tokens.map((token, tokenIdx) => (
                  <div 
                    key={tokenIdx} 
                    className={`token token-${token.type}`}
                    title={token.id ? `ID: ${token.id}` : undefined}
                  >
                    <span className="token-text">{token.text}</span>
                    {token.id !== undefined && <span className="token-id">[{token.id}]</span>}
                    
                    {token.vector && (
                      <div className="token-vector">
                        <small>Vector: [{token.vector.map(v => v.toFixed(2)).join(', ')}]</small>
                        <div className="vector-visualization">
                          {token.vector.map((value, i) => (
                            <div 
                              key={i}
                              className="vector-element"
                              style={{ 
                                width: '20px', 
                                height: '20px',
                                backgroundColor: `rgba(${value > 0 ? '74, 144, 226' : '226, 74, 74'}, ${Math.abs(value)})`,
                                margin: '1px'
                              }}
                              title={`Dimension ${i}: ${value.toFixed(3)}`}
                            />
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="tokenization-placeholder">
          <p>Enter some text and click "Tokenize" to see how it's broken down into tokens.</p>
        </div>
      )}
    </div>
  );
};

export default TokenizationVisualizer; 