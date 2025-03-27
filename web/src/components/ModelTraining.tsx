import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import './ModelTraining.css';

interface DatasetInfo {
  id: string;
  name: string;
  description: string;
  status: string;
  size?: string;
  num_tokens?: string;
  sample?: string;
  domain?: string;
}

interface ModelSize {
  id: string;
  name: string;
  params: string;
  description: string;
  training_time: string;
  capabilities: string[];
}

interface TrainingParams {
  dataset_id: string;
  model_size: string;
  seq_length: number;
  batch_size: number;
  learning_rate: number;
  epochs: number;
  train_split: number;
}

interface TrainingJob {
  id: string;
  status: string;
  progress: number;
  dataset: string;
  model_size: string;
  start_time: string;
  elapsed_time: string;
  estimated_completion?: string;
  metrics: {
    train_loss: number;
    val_loss?: number;
    learning_rate: number;
    step: number;
    total_steps: number;
    perplexity?: number;
  };
}

interface SamplePrompt {
  id: string;
  text: string;
}

interface SampleResponse {
  prompt: string;
  response: string;
  metrics: {
    coherence: number;
    relevance: number;
    creativity: number;
  }
}

const MODEL_SIZES: ModelSize[] = [
  {
    id: 'tiny',
    name: 'Tiny',
    params: '~2M',
    description: 'Very small model for quick experimentation and testing',
    training_time: '~5 min',
    capabilities: ['Basic text completion', 'Simple pattern recognition']
  },
  {
    id: 'small',
    name: 'Small',
    params: '~30M',
    description: 'Small model that can learn basic patterns',
    training_time: '~15 min',
    capabilities: ['Text completion', 'Basic Q&A', 'Simple summarization']
  },
  {
    id: 'base',
    name: 'Base',
    params: '~110M',
    description: 'Medium-sized model with decent language understanding',
    training_time: '~30 min',
    capabilities: ['Good text generation', 'Question answering', 'Summarization']
  },
  {
    id: 'medium',
    name: 'Medium',
    params: '~350M',
    description: 'Large model with good language capabilities',
    training_time: '~1 hour',
    capabilities: ['Strong text generation', 'Nuanced understanding', 'Good contextual responses']
  },
  {
    id: 'large',
    name: 'Large',
    params: '~1.3B',
    description: 'Very large model with strong language understanding',
    training_time: '~3 hours',
    capabilities: ['Excellent text generation', 'Complex reasoning', 'Detailed responses']
  },
];

// Sample prompts for evaluation
const SAMPLE_PROMPTS: SamplePrompt[] = [
  { id: 'intro', text: 'Write an introduction about language models.' },
  { id: 'explain', text: 'Explain how transformers work in simple terms.' },
  { id: 'summarize', text: 'Summarize the key ideas behind attention mechanisms.' },
  { id: 'create', text: 'Create a short story about a robot learning to understand human emotions.' },
];

// Define training process steps
const TRAINING_PROCESS_STEPS = [
  { id: 'dataset', name: 'Dataset Selection', description: 'Choose or upload a dataset for training' },
  { id: 'preprocessing', name: 'Data Preprocessing', description: 'Tokenization and preparation of training data' },
  { id: 'model', name: 'Model Configuration', description: 'Select model size and training parameters' },
  { id: 'training', name: 'Model Training', description: 'Train the model with selected parameters' },
  { id: 'evaluation', name: 'Model Evaluation', description: 'Test the trained model with prompts' },
];

const ModelTraining: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('tiny');
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadFileName, setUploadFileName] = useState<string>('');
  const [trainingParams, setTrainingParams] = useState<TrainingParams>({
    dataset_id: '',
    model_size: 'tiny',
    seq_length: 1024,
    batch_size: 8,
    learning_rate: 5e-5,
    epochs: 3,
    train_split: 0.9,
  });
  const [activeJobs, setActiveJobs] = useState<TrainingJob[]>([]);
  const [completedJobs, setCompletedJobs] = useState<TrainingJob[]>([]);
  const [activeTab, setActiveTab] = useState<'datasets' | 'model' | 'training' | 'evaluation' | 'explanation'>('explanation');
  const [sampleResponses, setSampleResponses] = useState<SampleResponse[]>([]);
  const [selectedModelForEval, setSelectedModelForEval] = useState<string>('');
  const [customPrompt, setCustomPrompt] = useState<string>('');
  const [customResponse, setCustomResponse] = useState<string>('');
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);
  const [fullTrainingProgress, setFullTrainingProgress] = useState<number>(0);
  
  const navigate = useNavigate();

  // Mock API calls - in a real implementation these would connect to the backend
  useEffect(() => {
    // Mock loading datasets
    const mockDatasets: DatasetInfo[] = [
      {
        id: 'tiny_shakespeare',
        name: 'Tiny Shakespeare',
        description: 'A collection of Shakespeare\'s works',
        status: 'available',
        size: '~1MB',
        num_tokens: '~1M',
        domain: 'Literature',
        sample: 'To be, or not to be, that is the question: Whether \'tis nobler in the mind to suffer The slings and arrows of outrageous fortune...'
      },
      {
        id: 'code_examples',
        name: 'Python Code Examples',
        description: 'Collection of Python code snippets',
        status: 'available',
        size: '~5MB',
        num_tokens: '~3M',
        domain: 'Programming',
        sample: 'def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)'
      },
      {
        id: 'wikipedia_articles',
        name: 'Wikipedia Articles',
        description: 'Sample of knowledge-focused articles',
        status: 'available',
        size: '~20MB',
        num_tokens: '~15M',
        domain: 'Knowledge',
        sample: 'Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence displayed by humans or other animals...'
      },
      {
        id: 'custom_dataset',
        name: 'My Custom Dataset',
        description: 'Uploaded text dataset',
        status: 'downloaded',
        size: '~100KB',
        num_tokens: '~50K',
        domain: 'Custom',
        sample: 'Your custom dataset content...'
      },
    ];
    
    setDatasets(mockDatasets);
    
    // Mock active training job
    const mockJob: TrainingJob = {
      id: 'job_123456',
      status: 'running',
      progress: 45,
      dataset: 'tiny_shakespeare',
      model_size: 'small',
      start_time: '2023-03-27T10:30:00Z',
      elapsed_time: '01:15:32',
      estimated_completion: '01:45:00',
      metrics: {
        train_loss: 2.34,
        val_loss: 2.56,
        learning_rate: 4.23e-5,
        step: 4500,
        total_steps: 10000,
        perplexity: 15.6
      },
    };
    
    // Mock completed job
    const mockCompletedJob: TrainingJob = {
      id: 'job_123455',
      status: 'completed',
      progress: 100,
      dataset: 'code_examples',
      model_size: 'tiny',
      start_time: '2023-03-26T14:30:00Z',
      elapsed_time: '00:30:12',
      metrics: {
        train_loss: 1.87,
        val_loss: 2.05,
        learning_rate: 5e-5,
        step: 5000,
        total_steps: 5000,
        perplexity: 12.3
      },
    };
    
    setActiveJobs([mockJob]);
    setCompletedJobs([mockCompletedJob]);
    setSelectedModelForEval(mockCompletedJob.id);
    
    // Mock sample responses
    const mockResponses: SampleResponse[] = [
      {
        prompt: SAMPLE_PROMPTS[0].text,
        response: "I'm a language model trained on a diverse range of texts. Based on your prompt, I can generate relevant and coherent responses that simulate human-like text.",
        metrics: {
          coherence: 0.92,
          relevance: 0.95,
          creativity: 0.78
        }
      },
      {
        prompt: SAMPLE_PROMPTS[1].text,
        response: "Transformers are like message-passing networks. Each word looks at all other words and decides how important they are to its own meaning. This 'attention' mechanism helps the model understand context much better than older approaches. It's like having a conversation where everyone can hear and remember everything that's been said.",
        metrics: {
          coherence: 0.89,
          relevance: 0.91,
          creativity: 0.82
        }
      }
    ];
    
    setSampleResponses(mockResponses);
  }, []);
  
  // Handle dataset selection
  const handleDatasetSelect = (datasetId: string) => {
    setSelectedDataset(datasetId);
    setTrainingParams({
      ...trainingParams,
      dataset_id: datasetId,
    });
    // Update progress bar when dataset is selected
    setFullTrainingProgress(20);
  };
  
  // Handle model selection
  const handleModelSelect = (modelId: string) => {
    setSelectedModel(modelId);
    setTrainingParams({
      ...trainingParams,
      model_size: modelId,
    });
    // Update progress bar when model is selected and dataset was already selected
    if (selectedDataset) {
      setFullTrainingProgress(40);
    }
  };
  
  // Handle file upload
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    setIsUploading(true);
    setUploadFileName(file.name);
    
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      setIsUploading(false);
      
      // In a real app, we would upload this to the server
      // For now, we'll just pretend we did and add it to the list
      const newDatasetId = file.name.replace(/\.[^/.]+$/, "").toLowerCase().replace(/\s+/g, '_');
      const newDataset: DatasetInfo = {
        id: newDatasetId,
        name: file.name.replace(/\.[^/.]+$/, ""),
        description: 'Uploaded text dataset',
        status: 'downloaded',
        size: `~${Math.round(file.size / 1024)} KB`,
        num_tokens: 'Unknown',
        domain: 'Custom',
        sample: content.slice(0, 150) + '...'
      };
      
      setDatasets([...datasets, newDataset]);
      setSelectedDataset(newDatasetId);
      setTrainingParams({
        ...trainingParams,
        dataset_id: newDatasetId,
      });
      
      // Update progress bar when dataset is uploaded
      setFullTrainingProgress(20);
    };
    
    reader.readAsText(file);
  };
  
  // Handle training parameter changes
  const handleParamChange = (paramName: keyof TrainingParams, value: any) => {
    setTrainingParams({
      ...trainingParams,
      [paramName]: value,
    });
  };
  
  // Start training
  const startTraining = () => {
    // In a real app, we would send a request to the server
    // For now, we'll just create a mock job
    const dataset = datasets.find(d => d.id === trainingParams.dataset_id);
    const modelSize = MODEL_SIZES.find(m => m.id === trainingParams.model_size);
    
    if (!dataset || !modelSize) return;
    
    const totalSteps = trainingParams.epochs * 1000; // Mock calculation
    
    const newJob: TrainingJob = {
      id: `job_${Date.now()}`,
      status: 'starting',
      progress: 0,
      dataset: trainingParams.dataset_id,
      model_size: trainingParams.model_size,
      start_time: new Date().toISOString(),
      elapsed_time: '00:00:00',
      estimated_completion: modelSize.training_time.replace('~', ''),
      metrics: {
        train_loss: 0,
        learning_rate: trainingParams.learning_rate,
        step: 0,
        total_steps: totalSteps,
      },
    };
    
    setActiveJobs([...activeJobs, newJob]);
    setActiveTab('training');
    
    // Set to preprocessing stage when starting training
    setFullTrainingProgress(40);
    
    // Simulate job progress
    let progress = 0;
    const interval = setInterval(() => {
      progress += 5;
      if (progress <= 100) {
        setActiveJobs(prev => 
          prev.map(job => 
            job.id === newJob.id 
              ? {
                  ...job,
                  status: progress < 100 ? 'running' : 'completed',
                  progress,
                  elapsed_time: `00:${String(Math.floor(progress / 5)).padStart(2, '0')}:00`,
                  metrics: {
                    ...job.metrics,
                    train_loss: 4 - (progress / 100) * 2,
                    val_loss: 4.2 - (progress / 100) * 1.8,
                    step: Math.floor((progress / 100) * totalSteps),
                    perplexity: 30 - (progress / 100) * 15
                  }
                }
              : job
          )
        );
        
        // Update global training progress based on job progress
        // Map job progress (0-100) to the training and evaluation stages (40-100)
        const mappedProgress = 40 + Math.floor((progress / 100) * 60);
        setFullTrainingProgress(mappedProgress);
      } else {
        clearInterval(interval);
        // Move to completed jobs
        setActiveJobs(prev => prev.filter(job => job.id !== newJob.id));
        setCompletedJobs(prev => [
          ...prev, 
          {
            ...newJob,
            status: 'completed',
            progress: 100,
            elapsed_time: modelSize.training_time.replace('~', ''),
            metrics: {
              ...newJob.metrics,
              train_loss: 2.1,
              val_loss: 2.4,
              step: totalSteps,
              total_steps: totalSteps,
              perplexity: 15.2
            }
          }
        ]);
        
        // Set to fully complete when training is done
        setFullTrainingProgress(100);
      }
    }, 1000);
  };
  
  // Generate response to a prompt
  const generateResponse = useCallback((prompt: string) => {
    setIsEvaluating(true);
    
    // Simulate API call delay
    setTimeout(() => {
      const responses = [
        "I'm a language model trained on a diverse range of texts. Based on your prompt, I can generate relevant and coherent responses that simulate human-like text.",
        "Language models like me work by predicting the next word in a sequence based on patterns learned during training. This allows me to generate text that follows the context and style of your input.",
        "The concept of attention in transformer models allows me to focus on different parts of your input when generating each word of my response, creating more contextually relevant outputs.",
        "Transformer models consist of multiple layers that process information iteratively, refining my understanding of your prompt with each pass through the network."
      ];
      
      const newResponse = {
        prompt,
        response: responses[Math.floor(Math.random() * responses.length)],
        metrics: {
          coherence: 0.7 + Math.random() * 0.25,
          relevance: 0.7 + Math.random() * 0.25,
          creativity: 0.6 + Math.random() * 0.3
        }
      };
      
      setSampleResponses(prev => [newResponse, ...prev]); // Add to beginning instead of end
      setCustomResponse(newResponse.response);
      setIsEvaluating(false);
    }, 2000);
  }, []);
  
  // Render the full training process progress bar
  const renderProgressBar = () => (
    <div className="full-training-progress">
      <h3>Training Pipeline Progress</h3>
      <div className="process-steps">
        {TRAINING_PROCESS_STEPS.map((step, index) => {
          // Calculate if this step is active or complete based on overall progress
          const stepThreshold = index * 20; // 20% per step
          const isActive = fullTrainingProgress > stepThreshold;
          const isComplete = fullTrainingProgress >= stepThreshold + 20;
          
          return (
            <div 
              key={step.id} 
              className={`process-step-indicator ${isActive ? 'active' : ''} ${isComplete ? 'complete' : ''}`}
            >
              <div className="step-label">{step.name}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
  
  // Render explanation tab
  const renderExplanationTab = () => (
    <div className="explanation-tab">
      <div className="tab-header">
        <h2>Understanding LLM Training</h2>
        <p>Learn how language models are trained and generate text</p>
      </div>
      
      <div className="learning-process">
        <div className="process-step">
          <h3>1. Dataset Selection</h3>
          <p>
            The first step in training a language model is selecting the right dataset. 
            The data you choose determines what your model will learn.
          </p>
          <div className="key-points">
            <div className="key-point">
              <div className="point-icon">📚</div>
              <div className="point-text">Models learn patterns, styles, and information present in training data</div>
            </div>
            <div className="key-point">
              <div className="point-icon">🔍</div>
              <div className="point-text">Domain-specific data produces specialized models</div>
            </div>
            <div className="key-point">
              <div className="point-icon">⚖️</div>
              <div className="point-text">Diverse data creates more versatile models</div>
            </div>
          </div>
        </div>
        
        <div className="process-step">
          <h3>2. Model Architecture</h3>
          <p>
            The size and structure of your model determine its capabilities and training requirements.
          </p>
          <div className="model-comparison">
            <div className="model-size">
              <div className="size-label">Small</div>
              <div className="size-description">
                <ul>
                  <li>Fast to train</li>
                  <li>Lower resource needs</li>
                  <li>Basic capabilities</li>
                </ul>
              </div>
            </div>
            <div className="model-size">
              <div className="size-label">Medium</div>
              <div className="size-description">
                <ul>
                  <li>Balanced training time</li>
                  <li>Moderate resource needs</li>
                  <li>Good capabilities</li>
                </ul>
              </div>
            </div>
            <div className="model-size">
              <div className="size-label">Large</div>
              <div className="size-description">
                <ul>
                  <li>Longer training time</li>
                  <li>Higher resource needs</li>
                  <li>Advanced capabilities</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
        
        <div className="process-step">
          <h3>3. Training Process</h3>
          <p>
            During training, the model learns to predict the next token in a sequence by analyzing patterns in the data.
          </p>
          <div className="training-visualization">
            <div className="training-step">
              <div className="step-title">Input:</div>
              <div className="token-sequence">
                <div className="token input">The</div>
                <div className="token input">cat</div>
                <div className="token input">sat</div>
                <div className="token input">on</div>
                <div className="token input">the</div>
              </div>
            </div>
            <div className="training-step">
              <div className="step-title">Prediction Task:</div>
              <div className="token-sequence">
                <div className="token input">The</div>
                <div className="arrow">→</div>
                <div className="token output">cat</div>
              </div>
              <div className="token-sequence">
                <div className="token input">The</div>
                <div className="token input">cat</div>
                <div className="arrow">→</div>
                <div className="token output">sat</div>
              </div>
              <div className="token-sequence">
                <div className="token input">The</div>
                <div className="token input">cat</div>
                <div className="token input">sat</div>
                <div className="arrow">→</div>
                <div className="token output">on</div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="process-step">
          <h3>4. Evaluation & Fine-tuning</h3>
          <p>
            Once trained, the model is evaluated and can be fine-tuned for specific tasks or improved performance.
          </p>
          <div className="evaluation-metrics">
            <div className="metric">
              <div className="metric-name">Perplexity</div>
              <div className="metric-description">Measures how well the model predicts text (lower is better)</div>
            </div>
            <div className="metric">
              <div className="metric-name">Loss</div>
              <div className="metric-description">Quantifies prediction errors during training</div>
            </div>
            <div className="metric">
              <div className="metric-name">Human Evaluation</div>
              <div className="metric-description">Assesses coherence, relevance, and quality of generated content</div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="action-buttons">
        <button 
          className="next-button"
          onClick={() => setActiveTab('datasets')}
        >
          Start Training Your Model
        </button>
      </div>
    </div>
  );
  
  // Render dataset selection tab
  const renderDatasetTab = () => (
    <div className="datasets-tab">
      <div className="tab-header">
        <h2>1. Select a Dataset</h2>
        <p>Choose a dataset to train your language model on. The dataset will determine what kind of text your model can generate.</p>
      </div>
      
      {renderProgressBar()}
      
      <div className="dataset-grid">
        {datasets.map((dataset) => (
          <div 
            key={dataset.id}
            className={`dataset-card ${selectedDataset === dataset.id ? 'selected' : ''}`}
            onClick={() => handleDatasetSelect(dataset.id)}
          >
            <h3>{dataset.name}</h3>
            <p>{dataset.description}</p>
            <div className="dataset-meta">
              <span className="domain-tag">{dataset.domain}</span>
              <span>{dataset.size}</span>
              <span>{dataset.num_tokens}</span>
              <span className={`status ${dataset.status}`}>{dataset.status}</span>
            </div>
            {selectedDataset === dataset.id && (
              <div className="dataset-sample">
                <h4>Sample:</h4>
                <p className="sample-text">{dataset.sample}</p>
              </div>
            )}
          </div>
        ))}
        
        <div className="dataset-card upload-card">
          <h3>Upload Your Own Dataset</h3>
          <p>Train on your own text data by uploading a file.</p>
          <input 
            type="file" 
            id="dataset-upload" 
            className="file-input" 
            accept=".txt,.csv,.json"
            onChange={handleFileUpload}
          />
          <label htmlFor="dataset-upload" className="upload-button">
            {isUploading ? 'Uploading...' : 'Select File'}
          </label>
          {uploadFileName && <p className="file-name">{uploadFileName}</p>}
        </div>
      </div>
      
      <div className="navigation-buttons">
        <button 
          className="back-button"
          onClick={() => setActiveTab('explanation')}
        >
          Back
        </button>
        <button 
          className="next-button"
          onClick={() => setActiveTab('model')}
          disabled={!selectedDataset}
        >
          Next: Choose Model
        </button>
      </div>
    </div>
  );
  
  // Render model selection tab
  const renderModelTab = () => (
    <div className="model-tab">
      <div className="tab-header">
        <h2>2. Choose Model Size</h2>
        <p>Select the size and complexity of your language model. Larger models can learn more complex patterns but take longer to train.</p>
      </div>
      
      {renderProgressBar()}
      
      <div className="model-grid">
        {MODEL_SIZES.map((model) => (
          <div 
            key={model.id}
            className={`model-card ${selectedModel === model.id ? 'selected' : ''}`}
            onClick={() => handleModelSelect(model.id)}
          >
            <div className="model-header">
              <h3>{model.name}</h3>
              <span className="params-badge">{model.params}</span>
            </div>
            <p>{model.description}</p>
            <div className="model-meta">
              <div className="training-time">
                <span className="meta-label">Training Time:</span>
                <span>{model.training_time}</span>
              </div>
              <div className="capabilities">
                <span className="meta-label">Capabilities:</span>
                <ul className="capabilities-list">
                  {model.capabilities.map((capability, index) => (
                    <li key={index}>{capability}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="training-parameters">
        <h3>Advanced Parameters</h3>
        <div className="params-grid">
          <div className="param-group">
            <label htmlFor="seq-length">Sequence Length</label>
            <input
              type="number"
              id="seq-length"
              value={trainingParams.seq_length}
              onChange={(e) => handleParamChange('seq_length', Number(e.target.value))}
              min={64}
              max={2048}
              step={64}
            />
            <div className="param-description">Maximum length of input sequences (longer = more context, slower training)</div>
          </div>
          
          <div className="param-group">
            <label htmlFor="batch-size">Batch Size</label>
            <input
              type="number"
              id="batch-size"
              value={trainingParams.batch_size}
              onChange={(e) => handleParamChange('batch_size', Number(e.target.value))}
              min={1}
              max={64}
            />
            <div className="param-description">Number of examples processed at once (higher = faster, but more memory)</div>
          </div>
          
          <div className="param-group">
            <label htmlFor="learning-rate">Learning Rate</label>
            <input
              type="number"
              id="learning-rate"
              value={trainingParams.learning_rate}
              onChange={(e) => handleParamChange('learning_rate', Number(e.target.value))}
              min={1e-6}
              max={1e-3}
              step={1e-6}
            />
            <div className="param-description">Rate at which the model learns (too high = unstable, too low = slow)</div>
          </div>
          
          <div className="param-group">
            <label htmlFor="epochs">Training Epochs</label>
            <input
              type="number"
              id="epochs"
              value={trainingParams.epochs}
              onChange={(e) => handleParamChange('epochs', Number(e.target.value))}
              min={1}
              max={10}
            />
            <div className="param-description">Number of times to iterate through the dataset</div>
          </div>
        </div>
      </div>
      
      <div className="navigation-buttons">
        <button 
          className="back-button"
          onClick={() => setActiveTab('datasets')}
        >
          Back
        </button>
        <button 
          className="start-button"
          onClick={startTraining}
          disabled={!selectedDataset || !selectedModel}
        >
          Start Training
        </button>
      </div>
    </div>
  );
  
  // Render training tab
  const renderTrainingTab = () => (
    <div className="training-tab">
      <div className="tab-header">
        <h2>3. Model Training</h2>
        <p>Monitor your model's training progress and performance metrics.</p>
      </div>
      
      {renderProgressBar()}
      
      <div className="training-jobs">
        <h3>Active Training Jobs</h3>
        {activeJobs.length > 0 ? (
          activeJobs.map((job) => (
            <div key={job.id} className="job-card">
              <div className="job-header">
                <h4>Training {MODEL_SIZES.find(m => m.id === job.model_size)?.name} Model on {datasets.find(d => d.id === job.dataset)?.name}</h4>
                <div className={`job-status ${job.status}`}>{job.status}</div>
              </div>
              
              <div className="job-meta">
                <div>Started: {new Date(job.start_time).toLocaleString()}</div>
                <div>Elapsed: {job.elapsed_time}</div>
                <div>Est. Completion: {job.estimated_completion}</div>
              </div>
              
              <div className="progress-container">
                <div className="progress-label">
                  Progress: {job.progress}% ({job.metrics.step}/{job.metrics.total_steps} steps)
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ width: `${job.progress}%` }}
                  ></div>
                </div>
              </div>
              
              <div className="metrics">
                <div className="metric">
                  <div className="metric-label">Training Loss</div>
                  <div className="metric-value">{job.metrics.train_loss.toFixed(3)}</div>
                </div>
                
                {job.metrics.val_loss && (
                  <div className="metric">
                    <div className="metric-label">Validation Loss</div>
                    <div className="metric-value">{job.metrics.val_loss.toFixed(3)}</div>
                  </div>
                )}
                
                {job.metrics.perplexity && (
                  <div className="metric">
                    <div className="metric-label">Perplexity</div>
                    <div className="metric-value">{job.metrics.perplexity.toFixed(2)}</div>
                  </div>
                )}
                
                <div className="metric">
                  <div className="metric-label">Learning Rate</div>
                  <div className="metric-value">{job.metrics.learning_rate.toExponential(2)}</div>
                </div>
              </div>
              
              <div className="training-chart">
                <div className="chart-title">Training Progress</div>
                <div className="placeholder-chart">
                  <svg width="100%" height="100%" viewBox="0 0 500 200" preserveAspectRatio="none">
                    <path d="M0,200 C50,180 100,100 150,90 C200,80 250,120 300,100 C350,80 400,60 450,50 L500,30" 
                          stroke="#0070f3" 
                          strokeWidth="3" 
                          fill="none" />
                    <path d="M0,200 C50,190 100,150 150,130 C200,120 250,140 300,130 C350,120 400,100 450,90 L500,70" 
                          stroke="#ff0000" 
                          strokeWidth="2" 
                          strokeDasharray="5,5" 
                          fill="none" />
                  </svg>
                  <div className="chart-legend">
                    <div className="legend-item">
                      <div className="legend-color" style={{backgroundColor: "#0070f3"}}></div>
                      <div>Training Loss</div>
                    </div>
                    <div className="legend-item">
                      <div className="legend-color" style={{backgroundColor: "#ff0000"}}></div>
                      <div>Validation Loss</div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="job-actions">
                <button className="view-button">View Details</button>
                <button className="stop-button">Stop Training</button>
              </div>
            </div>
          ))
        ) : (
          <div className="no-jobs-message">No active training jobs</div>
        )}
        
        <h3>Completed Models</h3>
        {completedJobs.length > 0 ? (
          <div className="completed-models-grid">
            {completedJobs.map((job) => (
              <div key={job.id} className="completed-model-card">
                <div className="model-title">
                  <h4>{datasets.find(d => d.id === job.dataset)?.name} - {MODEL_SIZES.find(m => m.id === job.model_size)?.name}</h4>
                  <div className="model-date">{new Date(job.start_time).toLocaleDateString()}</div>
                </div>
                <div className="model-metrics">
                  <div className="model-metric">
                    <span className="metric-label">Final Loss:</span>
                    <span className="metric-value">{job.metrics.train_loss.toFixed(3)}</span>
                  </div>
                  <div className="model-metric">
                    <span className="metric-label">Perplexity:</span>
                    <span className="metric-value">{job.metrics.perplexity?.toFixed(2) || "N/A"}</span>
                  </div>
                </div>
                <div className="model-actions">
                  <button 
                    className="evaluate-button"
                    onClick={() => {
                      setSelectedModelForEval(job.id);
                      setActiveTab('evaluation');
                    }}
                  >
                    Evaluate
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="no-models-message">No completed models yet</div>
        )}
      </div>
    </div>
  );
  
  // Render evaluation tab
  const renderEvaluationTab = () => (
    <div className="evaluation-tab">
      <div className="tab-header">
        <h2>4. Model Evaluation</h2>
        <p>Test your trained model with sample prompts to evaluate its performance.</p>
      </div>
      
      {renderProgressBar()}
      
      <div className="selected-model-info">
        {completedJobs.find(job => job.id === selectedModelForEval) && (
          <div className="model-details">
            <h3>Selected Model</h3>
            <div className="details-container">
              <div className="detail">
                <span className="detail-label">Dataset:</span>
                <span className="detail-value">
                  {datasets.find(d => d.id === completedJobs.find(job => job.id === selectedModelForEval)?.dataset)?.name}
                </span>
              </div>
              <div className="detail">
                <span className="detail-label">Model Size:</span>
                <span className="detail-value">
                  {MODEL_SIZES.find(m => m.id === completedJobs.find(job => job.id === selectedModelForEval)?.model_size)?.name}
                </span>
              </div>
              <div className="detail">
                <span className="detail-label">Perplexity:</span>
                <span className="detail-value">
                  {completedJobs.find(job => job.id === selectedModelForEval)?.metrics.perplexity?.toFixed(2) || "N/A"}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
      
      <div className="sample-prompts">
        <h3>Sample Prompts</h3>
        <div className="prompts-container">
          {SAMPLE_PROMPTS.map((prompt) => (
            <div 
              key={prompt.id}
              className="prompt-card"
              onClick={() => !sampleResponses.some(r => r.prompt === prompt.text) && generateResponse(prompt.text)}
            >
              <p>{prompt.text}</p>
              <button 
                className="generate-button"
                disabled={sampleResponses.some(r => r.prompt === prompt.text)}
              >
                {sampleResponses.some(r => r.prompt === prompt.text) ? "Generated" : "Generate"}
              </button>
            </div>
          ))}
        </div>
      </div>
      
      <div className="custom-prompt">
        <h3>Try Your Own Prompt</h3>
        <textarea
          value={customPrompt}
          onChange={(e) => setCustomPrompt(e.target.value)}
          placeholder="Enter your prompt here..."
          rows={3}
          className="custom-prompt-input"
        />
        <button 
          className="generate-button"
          onClick={() => generateResponse(customPrompt)}
          disabled={!customPrompt || isEvaluating}
        >
          {isEvaluating ? "Generating..." : "Generate Response"}
        </button>
        
        {customResponse && (
          <div className="custom-response">
            <h4>Response:</h4>
            <p>{customResponse}</p>
          </div>
        )}
      </div>
      
      <div className="generated-responses">
        <h3>Model Responses</h3>
        {sampleResponses.length > 0 ? (
          <div className="responses-list">
            {sampleResponses.map((response, index) => (
              <div key={index} className="response-card">
                <div className="prompt-text">
                  <strong>Prompt:</strong> {response.prompt}
                </div>
                <div className="response-text">
                  <strong>Response:</strong> {response.response}
                </div>
                <div className="response-metrics">
                  <div className="metric-item">
                    <div className="metric-name">Coherence</div>
                    <div className="metric-bar">
                      <div 
                        className="metric-fill"
                        style={{ width: `${response.metrics.coherence * 100}%` }}
                      ></div>
                    </div>
                    <div className="metric-value">{Math.round(response.metrics.coherence * 100)}</div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-name">Relevance</div>
                    <div className="metric-bar">
                      <div 
                        className="metric-fill"
                        style={{ width: `${response.metrics.relevance * 100}%` }}
                      ></div>
                    </div>
                    <div className="metric-value">{Math.round(response.metrics.relevance * 100)}</div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-name">Creativity</div>
                    <div className="metric-bar">
                      <div 
                        className="metric-fill"
                        style={{ width: `${response.metrics.creativity * 100}%` }}
                      ></div>
                    </div>
                    <div className="metric-value">{Math.round(response.metrics.creativity * 100)}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="no-responses-message">No responses generated yet. Try a sample prompt!</div>
        )}
      </div>
      
      <div className="navigation-buttons">
        <button 
          className="back-button"
          onClick={() => setActiveTab('training')}
        >
          Back to Training
        </button>
      </div>
    </div>
  );
  
  return (
    <div className="model-training-page">
      <div className="page-header">
        <h1>Train Your Language Model</h1>
        <p>Create, train and evaluate custom language models with your own data</p>
      </div>
      
      {/* Global progress bar - always visible */}
      {renderProgressBar()}
      
      <div className="tab-navigation">
        <button 
          className={`tab-button ${activeTab === 'explanation' ? 'active' : ''}`}
          onClick={() => setActiveTab('explanation')}
        >
          Overview
        </button>
        <button 
          className={`tab-button ${activeTab === 'datasets' ? 'active' : ''}`}
          onClick={() => setActiveTab('datasets')}
        >
          1. Dataset
        </button>
        <button 
          className={`tab-button ${activeTab === 'model' ? 'active' : ''}`}
          onClick={() => setActiveTab('model')}
        >
          2. Model
        </button>
        <button 
          className={`tab-button ${activeTab === 'training' ? 'active' : ''}`}
          onClick={() => setActiveTab('training')}
        >
          3. Training
        </button>
        <button 
          className={`tab-button ${activeTab === 'evaluation' ? 'active' : ''}`}
          onClick={() => activeTab === 'evaluation' || completedJobs.length > 0 ? setActiveTab('evaluation') : null}
          disabled={completedJobs.length === 0 && activeTab !== 'evaluation'}
        >
          4. Evaluation
        </button>
      </div>
      
      <div className="tab-content">
        {activeTab === 'explanation' && renderExplanationTab()}
        {activeTab === 'datasets' && renderDatasetTab()}
        {activeTab === 'model' && renderModelTab()}
        {activeTab === 'training' && renderTrainingTab()}
        {activeTab === 'evaluation' && renderEvaluationTab()}
      </div>
    </div>
  );
};

export default ModelTraining; 