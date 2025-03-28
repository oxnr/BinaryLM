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
    id: 'llm-demo',
    title: 'End-to-End Language Model Creation',
    description: 'A comprehensive notebook demonstrating how to create a language model from scratch, including data preparation, tokenization, model architecture (Transformer), training, and text generation.',
    path: 'notebooks/LLM_Demo.ipynb'
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
            View them here or open them in Colab/Binder to run them interactively.
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