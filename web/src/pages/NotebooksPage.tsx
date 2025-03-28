import React, { useState } from 'react';
import SimpleNotebookViewer from '../components/SimpleNotebookViewer';
import './NotebooksPage.css';

interface Notebook {
  id: string;
  title: string;
  description: string;
  path: string;
}

const AVAILABLE_NOTEBOOKS: Notebook[] = [
  {
    id: 'build-lm-from-scratch',
    title: 'Building a Language Model from Scratch',
    description: 'Learn how to build a simple transformer-based language model from scratch using PyTorch.',
    path: 'notebooks/Build_LM_From_Scratch.ipynb'
  },
  {
    id: 'building-with-transformers',
    title: 'Building LMs with the Transformers Library',
    description: 'Learn how to leverage the Hugging Face Transformers library to fine-tune and use pre-trained language models.',
    path: 'notebooks/Building_with_Transformers.ipynb'
  },
  {
    id: 'sample-notebook',
    title: 'Sample Notebook',
    description: 'A basic sample notebook to demonstrate the notebook viewer functionality.',
    path: 'notebooks/sample_notebook.ipynb'
  }
];

const NotebooksPage: React.FC = () => {
  const [selectedNotebook, setSelectedNotebook] = useState<Notebook | null>(null);

  const handleSelectNotebook = (notebook: Notebook) => {
    setSelectedNotebook(notebook);
    // Preload the notebook file to ensure it's available in browser cache
    fetch(notebook.path).catch(err => console.error(`Error preloading notebook: ${err.message}`));
  };

  const handleBack = () => {
    setSelectedNotebook(null);
  };

  return (
    <div className="notebooks-page">
      {selectedNotebook ? (
        <div className="notebook-view">
          <button 
            className="back-button"
            onClick={handleBack}
          >
            ← Back to notebooks
          </button>
          <SimpleNotebookViewer 
            notebookPath={selectedNotebook.path} 
            title={selectedNotebook.title} 
          />
        </div>
      ) : (
        <div className="notebooks-list">
          <h1>Jupyter Notebooks</h1>
          <p className="notebooks-intro">
            These interactive notebooks demonstrate how to build language models from scratch and how to use existing libraries.
            You can view them here or download to run them locally in Jupyter.
          </p>
          
          <div className="notebooks-grid">
            {AVAILABLE_NOTEBOOKS.map(notebook => (
              <div 
                key={notebook.id} 
                className="notebook-card"
                onClick={() => handleSelectNotebook(notebook)}
              >
                <div className="notebook-card-content">
                  <h2>{notebook.title}</h2>
                  <p>{notebook.description}</p>
                  <div className="jupyter-icon">
                    <span role="img" aria-label="Jupyter Notebook">📓</span>
                  </div>
                </div>
                <button className="view-notebook-button">
                  Open Notebook
                </button>
              </div>
            ))}
          </div>
          
          <div className="notebooks-info">
            <h2>Running Notebooks Locally</h2>
            <p>
              To run these notebooks locally, you'll need to:
            </p>
            <ol>
              <li>Download the notebooks from the <a href="https://github.com/oxnr/BinaryLM/tree/main/notebooks" target="_blank" rel="noopener noreferrer">GitHub repository</a></li>
              <li>Install Jupyter: <code>pip install jupyter</code></li>
              <li>Install dependencies: <code>pip install -r requirements.txt</code></li>
              <li>Launch Jupyter: <code>jupyter notebook</code></li>
            </ol>
            <p>
              These notebooks provide a hands-on way to understand how language models work and how to build your own from scratch.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default NotebooksPage; 