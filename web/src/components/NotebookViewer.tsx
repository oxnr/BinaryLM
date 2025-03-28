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
  
  // Use absolute URL to ensure path is correct
  const getAbsoluteNotebookPath = (path: string) => {
    // Remove leading slash if it exists to avoid double slashes
    const cleanPath = path.startsWith('/') ? path.substring(1) : path;
    return `${window.location.origin}/${cleanPath}`;
  };

  useEffect(() => {
    const fetchNotebook = async () => {
      try {
        setLoading(true);
        const absolutePath = getAbsoluteNotebookPath(notebookPath);
        console.log(`Fetching notebook from: ${absolutePath}`);
        
        const response = await fetch(absolutePath);
        console.log(`Fetch response status: ${response.status} ${response.statusText}`);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch notebook: ${response.status} ${response.statusText}`);
        }
        
        const text = await response.text();
        console.log(`Received data length: ${text.length} characters`);
        console.log(`First 50 characters: ${text.substring(0, 50)}...`);
        
        if (!text || text.trim() === '') {
          throw new Error('Received empty response');
        }
        
        try {
          const data = JSON.parse(text);
          console.log('Successfully parsed notebook JSON');
          setNotebook(data);
          setError(null);
        } catch (parseError) {
          console.error('JSON parse error:', parseError);
          console.error('Raw text received:', text);
          
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
        
        // Set a fallback notebook
        setNotebook({
          cells: [
            {
              cell_type: "markdown",
              metadata: {},
              source: [
                "# Error Loading Notebook\n",
                `\n`,
                `We encountered an error while loading this notebook: ${err instanceof Error ? err.message : String(err)}\n`,
                `\n`,
                "Please try again later or download the notebooks directly from our GitHub repository."
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
        });
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