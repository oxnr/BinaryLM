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
        console.log(`Fetching notebook from: ${notebookPath}`);
        
        const response = await fetch(notebookPath);
        if (!response.ok) {
          throw new Error(`Failed to fetch notebook from ${notebookPath}: ${response.statusText}`);
        }
        
        const text = await response.text();
        console.log(`Received data length: ${text.length} characters`);
        
        if (!text || text.trim() === '') {
          throw new Error('Received empty response');
        }
        
        // Create a synthetic notebook if we can't load the real one
        try {
          const data = JSON.parse(text);
          setNotebook(data);
          setError(null);
        } catch (parseError) {
          console.error('JSON parse error:', parseError);
          
          // Create a basic notebook structure for debugging
          const fallbackNotebook = {
            cells: [
              {
                cell_type: "markdown",
                metadata: {},
                source: [
                  "# Unable to load notebook\n",
                  "There was an error loading the notebook from the server. Please try again later or download the notebooks from GitHub."
                ]
              }
            ],
            metadata: {
              kernelspec: {
                display_name: "Python 3",
                language: "python",
                name: "python3"
              }
            },
            nbformat: 4,
            nbformat_minor: 5
          };
          
          setNotebook(fallbackNotebook);
          setError(`Failed to parse notebook JSON: ${parseError instanceof Error ? parseError.message : String(parseError)}`);
        }
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
          <p>
            Please make sure the notebook exists and is correctly formatted.
            You can download the notebooks directly from the <a href="https://github.com/oxnr/BinaryLM/tree/main/notebooks" target="_blank" rel="noopener noreferrer">GitHub repository</a>.
          </p>
        </div>
      )}
      
      {!loading && notebook && (
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