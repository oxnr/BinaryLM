import React, { useState, useEffect } from 'react';
import './PredictionVisualizer.css';

interface TokenPrediction {
  token: string;
  probability: number;
}

interface PredictionVisualizerProps {
  predictions: TokenPrediction[];
  selectedToken: string;
  onTokenClick?: (token: string) => void;
  maxTokensToShow?: number;
}

const PredictionVisualizer: React.FC<PredictionVisualizerProps> = ({
  predictions,
  selectedToken,
  onTokenClick,
  maxTokensToShow = 10
}) => {
  const [sortedPredictions, setSortedPredictions] = useState<TokenPrediction[]>([]);

  // Update predictions immediately when they change
  useEffect(() => {
    if (!predictions || predictions.length === 0) {
      console.log("No predictions received");
      setSortedPredictions([]);
      return;
    }

    console.log("PredictionVisualizer received predictions:", predictions);
    console.log("Selected token:", selectedToken);
    
    // Sort predictions by probability (highest first)
    const sorted = [...predictions].sort((a, b) => b.probability - a.probability);
    
    // Take only the top N tokens to display
    const topPredictions = sorted.slice(0, maxTokensToShow);
    console.log("Sorted and filtered predictions:", topPredictions);
    
    // Update state immediately
    setSortedPredictions(topPredictions);
  }, [predictions, maxTokensToShow]);

  // Calculate the maximum probability for scaling
  const maxProbability = sortedPredictions.length > 0 
    ? Math.max(...sortedPredictions.map(p => p.probability))
    : 1.0;

  console.log("Rendering PredictionVisualizer with", sortedPredictions.length, "predictions");
  console.log("Max probability:", maxProbability);
  
  return (
    <div className="prediction-visualizer">
      <h3>Token Prediction Distribution</h3>
      
      <div className="prediction-container">
        {sortedPredictions.length > 0 ? (
          sortedPredictions.map((pred, index) => {
            const width = (pred.probability / maxProbability) * 100;
            console.log(`Rendering prediction ${index + 1}: "${pred.token}" (${width.toFixed(2)}%)`);
            
            return (
              <div 
                key={`${pred.token}-${index}`}
                className={`prediction-bar ${pred.token === selectedToken ? 'selected' : ''}`}
                onClick={() => onTokenClick && onTokenClick(pred.token)}
              >
                <div 
                  className="probability-bar" 
                  style={{ width: `${width}%` }}
                />
                <div className="token-info">
                  <span className="token-text">{pred.token}</span>
                  <span className="token-probability">{(pred.probability * 100).toFixed(2)}%</span>
                </div>
              </div>
            );
          })
        ) : (
          <div className="no-predictions">
            No token predictions available.
          </div>
        )}
      </div>
      
      {sortedPredictions.length > 0 && (
        <div className="prediction-legend">
          <div className="legend-item">
            <div className="legend-color selected" />
            <span>Selected Token</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" />
            <span>Alternative Token</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionVisualizer; 