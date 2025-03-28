import React, { useState, useEffect } from 'react';
import JupyterNotebook from 'react-jupyter-notebook';
import './NotebookViewer.css';

interface NotebookViewerProps {
  notebookPath: string;
  title?: string;
}

const NotebookViewer: React.FC<NotebookViewerProps> = ({ notebookPath, title }) => {
  const [notebook, setNotebook] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchNotebook = async () => {
      try {
        setLoading(true);
        const response = await fetch(notebookPath);
        if (!response.ok) {
          throw new Error(`Failed to fetch notebook from ${notebookPath}`);
        }
        const data = await response.json();
        setNotebook(data);
        setError(null);
      } catch (err) {
        console.error('Error loading notebook:', err);
        setError(`Failed to load notebook: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        setLoading(false);
      }
    };

    fetchNotebook();
  }, [notebookPath]);

  return (
    <div className="notebook-viewer">
      {title && <h1 className="notebook-title">{title}</h1>}
      
      {loading && (
        <div className="notebook-loading">
          <div className="loading-spinner"></div>
          <p>Loading notebook...</p>
        </div>
      )}
      
      {error && (
        <div className="notebook-error">
          <h3>Error</h3>
          <p>{error}</p>
          <p>Please make sure the notebook exists and is correctly formatted.</p>
        </div>
      )}
      
      {!loading && !error && notebook && (
        <div className="notebook-container">
          <JupyterNotebook 
            rawIpynb={notebook} 
            showLineNumbers={true}
            displaySource="show"
            displayOutput="show"
          />
        </div>
      )}
    </div>
  );
};

export default NotebookViewer; 