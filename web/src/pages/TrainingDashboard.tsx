import React, { useState } from 'react';

interface TrainingConfig {
  learningRate: number;
  batchSize: number;
  epochs: number;
  datasetSize: number;
}

const TrainingDashboard: React.FC = () => {
  const [config, setConfig] = useState<TrainingConfig>({
    learningRate: 0.0001,
    batchSize: 32,
    epochs: 3,
    datasetSize: 10000
  });
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);

  const handleConfigChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setConfig(prev => ({
      ...prev,
      [name]: name === 'learningRate' ? parseFloat(value) : parseInt(value, 10)
    }));
  };

  const handleStartTraining = () => {
    setIsTraining(true);
    setProgress(0);
    
    // Simulate training progress
    const interval = setInterval(() => {
      setProgress(prev => {
        const newProgress = prev + 5;
        if (newProgress >= 100) {
          clearInterval(interval);
          setIsTraining(false);
          return 100;
        }
        return newProgress;
      });
    }, 500);
  };

  return (
    <div>
      <h1>Training Dashboard</h1>
      
      <div className="card">
        <h2>Training Configuration</h2>
        <div className="form-control">
          <label htmlFor="learningRate">Learning Rate:</label>
          <input
            type="number"
            id="learningRate"
            name="learningRate"
            value={config.learningRate}
            onChange={handleConfigChange}
            step="0.0001"
            min="0.0001"
            max="0.01"
            disabled={isTraining}
          />
        </div>

        <div className="form-control">
          <label htmlFor="batchSize">Batch Size:</label>
          <input
            type="number"
            id="batchSize"
            name="batchSize"
            value={config.batchSize}
            onChange={handleConfigChange}
            step="1"
            min="1"
            max="128"
            disabled={isTraining}
          />
        </div>

        <div className="form-control">
          <label htmlFor="epochs">Epochs:</label>
          <input
            type="number"
            id="epochs"
            name="epochs"
            value={config.epochs}
            onChange={handleConfigChange}
            step="1"
            min="1"
            max="20"
            disabled={isTraining}
          />
        </div>

        <div className="form-control">
          <label htmlFor="datasetSize">Dataset Size:</label>
          <input
            type="number"
            id="datasetSize"
            name="datasetSize"
            value={config.datasetSize}
            onChange={handleConfigChange}
            step="1000"
            min="1000"
            max="100000"
            disabled={isTraining}
          />
        </div>

        <button 
          onClick={handleStartTraining}
          disabled={isTraining}
        >
          {isTraining ? 'Training...' : 'Start Training'}
        </button>
      </div>

      {isTraining && (
        <div className="card">
          <h2>Training Progress</h2>
          <div className="visualizer">
            <div style={{ marginBottom: '10px' }}>
              <strong>Progress: {progress}%</strong>
            </div>
            <div style={{ 
              height: '20px', 
              backgroundColor: '#e9ecef', 
              borderRadius: '4px',
              overflow: 'hidden'
            }}>
              <div style={{
                height: '100%',
                width: `${progress}%`,
                backgroundColor: '#4a90e2',
                transition: 'width 0.3s ease'
              }} />
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <h2>Training Visualization</h2>
        <div className="visualizer">
          <p>Training metrics visualization will be implemented here.</p>
          {/* Placeholder for training visualization */}
          <div style={{ 
            height: '300px', 
            backgroundColor: '#f8f9fa', 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center',
            border: '1px dashed #ccc'
          }}>
            Training Metrics Visualization Placeholder
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingDashboard; 