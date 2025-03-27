import React, { useState, useEffect } from 'react';
import './TransformerVisualizer.css';

interface TransformerVisualizerProps {
  modelType?: 'encoder-decoder' | 'decoder-only' | 'encoder-only';
  inputText?: string;
}

const TransformerVisualizer: React.FC<TransformerVisualizerProps> = ({
  modelType = 'decoder-only',
  inputText = 'The transformer architecture is revolutionary'
}) => {
  const [activeLayer, setActiveLayer] = useState<number>(0);
  const [showDetails, setShowDetails] = useState<boolean>(false);
  const [activePath, setActivePath] = useState<string | null>(null);
  
  // Generated token colors for visualization
  const tokenColors = [
    '#FF6B6B', '#48DBFB', '#1DD1A1', '#F368E0', '#FF9F43', 
    '#00D2D3', '#FECA57', '#54A0FF', '#C8D6E5'
  ];
  
  // Simulate tokens from the input text
  const tokens = inputText.split(/\s+/).map((token, index) => ({
    text: token,
    color: tokenColors[index % tokenColors.length],
    id: index
  }));
  
  const numLayers = modelType === 'encoder-only' ? 6 
    : modelType === 'decoder-only' ? 8 
    : 12; // encoder-decoder
  
  const layerBlocks = Array.from({ length: numLayers }, (_, i) => i);

  // Simulated attention patterns between tokens
  const generateAttentionPatterns = () => {
    const patterns: Record<number, Record<number, number>> = {};
    
    tokens.forEach((_, sourceIdx) => {
      patterns[sourceIdx] = {};
      tokens.forEach((_, targetIdx) => {
        // In decoder models, can only attend to previous tokens
        if (modelType.includes('decoder') && targetIdx > sourceIdx) {
          patterns[sourceIdx][targetIdx] = 0;
        } else {
          // Random attention scores, higher for nearby tokens
          const distance = Math.abs(sourceIdx - targetIdx);
          const baseScore = Math.max(0, 1 - distance * 0.25);
          patterns[sourceIdx][targetIdx] = baseScore * (0.5 + Math.random() * 0.5);
        }
      });
    });
    
    return patterns;
  };
  
  const attentionPatterns = generateAttentionPatterns();
  
  const handleTokenHover = (tokenId: number) => {
    setActivePath(`token-${tokenId}`);
  };
  
  const handleTokenLeave = () => {
    setActivePath(null);
  };

  return (
    <div className="transformer-visualizer">
      <div className="model-controls">
        <h3>Transformer {modelType.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join('-')} Model</h3>
        
        <div className="control-options">
          <div className="control-group">
            <label>Layer: </label>
            <select 
              value={activeLayer} 
              onChange={(e) => setActiveLayer(parseInt(e.target.value))}
            >
              {layerBlocks.map((layer) => (
                <option key={layer} value={layer}>Layer {layer + 1}</option>
              ))}
            </select>
          </div>
          
          <div className="control-group">
            <label>
              <input 
                type="checkbox" 
                checked={showDetails} 
                onChange={() => setShowDetails(!showDetails)}
              />
              Show Details
            </label>
          </div>
        </div>
      </div>
      
      <div className="architecture-visualization">
        {/* Input tokens */}
        <div className="layer input-layer">
          <div className="layer-label">Input Tokens</div>
          <div className="tokens-container">
            {tokens.map((token) => (
              <div 
                key={token.id}
                className={`token ${activePath === `token-${token.id}` ? 'active' : ''}`}
                style={{ backgroundColor: token.color }}
                onMouseEnter={() => handleTokenHover(token.id)}
                onMouseLeave={handleTokenLeave}
              >
                {token.text}
              </div>
            ))}
          </div>
        </div>
        
        {/* Embedding layer */}
        <div className="layer embedding-layer">
          <div className="layer-label">Token + Positional Embeddings</div>
          <div className="tokens-container">
            {tokens.map((token) => (
              <div 
                key={token.id}
                className={`token embedding ${activePath === `token-${token.id}` ? 'active' : ''}`}
                style={{ 
                  backgroundColor: token.color,
                  backgroundImage: 'repeating-linear-gradient(45deg, transparent, transparent 5px, rgba(255,255,255,0.2) 5px, rgba(255,255,255,0.2) 10px)'
                }}
                onMouseEnter={() => handleTokenHover(token.id)}
                onMouseLeave={handleTokenLeave}
              >
                {token.text} + pos
              </div>
            ))}
          </div>
        </div>
        
        {/* Self-attention visualization */}
        <div className="layer attention-layer">
          <div className="layer-label">Self-Attention (Layer {activeLayer + 1})</div>
          
          {showDetails && (
            <div className="details-container">
              <div className="attention-detail-container">
                <div className="attention-header">Attention Matrix</div>
                <div className="attention-matrix">
                  {tokens.map((source, sourceIdx) => (
                    <div key={sourceIdx} className="attention-row">
                      {tokens.map((target, targetIdx) => {
                        const score = attentionPatterns[sourceIdx][targetIdx] || 0;
                        const isMasked = modelType.includes('decoder') && targetIdx > sourceIdx;
                        
                        return (
                          <div 
                            key={targetIdx}
                            className={`attention-cell ${isMasked ? 'masked' : ''}`}
                            style={{ 
                              backgroundColor: !isMasked 
                                ? `rgba(74, 144, 226, ${score})` 
                                : 'rgba(200, 200, 200, 0.2)',
                              border: `1px solid ${isMasked ? '#ddd' : 'rgba(74, 144, 226, 0.5)'}`
                            }}
                            title={`${source.text} → ${target.text}: ${isMasked ? 'Masked' : score.toFixed(2)}`}
                          />
                        );
                      })}
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="computation-detail">
                <div className="computation-step">
                  <div className="step-name">1. Project to Q, K, V</div>
                  <div className="step-detail">Linear projection of each embedding</div>
                </div>
                <div className="computation-step">
                  <div className="step-name">2. Compute QK^T</div>
                  <div className="step-detail">Matrix multiply Q and K transpose</div>
                </div>
                <div className="computation-step">
                  <div className="step-name">3. Scale & Softmax</div>
                  <div className="step-detail">Divide by √d_k, apply softmax</div>
                </div>
                <div className="computation-step">
                  <div className="step-name">4. Apply to V</div>
                  <div className="step-detail">Matrix multiply with V</div>
                </div>
              </div>
            </div>
          )}
          
          <div className="tokens-container">
            {tokens.map((token) => (
              <div 
                key={token.id}
                className={`token attention-output ${activePath === `token-${token.id}` ? 'active' : ''}`}
                style={{ backgroundColor: token.color }}
                onMouseEnter={() => handleTokenHover(token.id)}
                onMouseLeave={handleTokenLeave}
              >
                {token.text}*
              </div>
            ))}
          </div>
        </div>
        
        {/* Feed forward layer */}
        <div className="layer ffn-layer">
          <div className="layer-label">Feed-Forward Network</div>
          <div className="tokens-container">
            {tokens.map((token) => (
              <div 
                key={token.id}
                className={`token ffn-output ${activePath === `token-${token.id}` ? 'active' : ''}`}
                style={{ backgroundColor: token.color }}
                onMouseEnter={() => handleTokenHover(token.id)}
                onMouseLeave={handleTokenLeave}
              >
                {token.text}**
              </div>
            ))}
          </div>
        </div>
        
        {/* Output layer */}
        <div className="layer output-layer">
          <div className="layer-label">Layer Output</div>
          <div className="tokens-container">
            {tokens.map((token) => (
              <div 
                key={token.id}
                className={`token layer-output ${activePath === `token-${token.id}` ? 'active' : ''}`}
                style={{ backgroundColor: token.color }}
                onMouseEnter={() => handleTokenHover(token.id)}
                onMouseLeave={handleTokenLeave}
              >
                {token.text}***
              </div>
            ))}
          </div>
        </div>
      </div>
      
      <div className="next-token-prediction">
        <h4>Next Token Prediction (for Decoder Models)</h4>
        <div className="prediction-container">
          <div className="predicted-token">
            <div className="prediction-score">87%</div>
            <div className="prediction-text">is</div>
          </div>
          <div className="predicted-token">
            <div className="prediction-score">8%</div>
            <div className="prediction-text">was</div>
          </div>
          <div className="predicted-token">
            <div className="prediction-score">3%</div>
            <div className="prediction-text">has</div>
          </div>
          <div className="predicted-token">
            <div className="prediction-score">1%</div>
            <div className="prediction-text">becomes</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TransformerVisualizer; 