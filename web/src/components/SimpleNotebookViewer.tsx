import React, { useState, useEffect } from 'react';
import './NotebookViewer.css';

interface NotebookViewerProps {
  notebookPath: string;
  title?: string;
}

interface Cell {
  cell_type: string;
  metadata: any;
  source: string[];
  execution_count?: number | null;
  outputs?: any[];
}

interface NotebookData {
  cells: Cell[];
  metadata: any;
  nbformat: number;
  nbformat_minor: number;
}

const SimpleNotebookViewer: React.FC<NotebookViewerProps> = ({ notebookPath, title }) => {
  const [notebook, setNotebook] = useState<NotebookData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  // Use absolute URL to ensure path is correct
  const getAbsoluteNotebookPath = (path: string) => {
    // Remove leading slash if it exists to avoid double slashes
    const cleanPath = path.startsWith('/') ? path.substring(1) : path;
    return `${window.location.origin}/${cleanPath}`;
  };

  // Get github URLs for the notebook
  const getGitHubUrl = (path: string) => {
    const repoOwner = 'oxnr';
    const repoName = 'BinaryLM';
    const branch = 'main';
    const notebookPath = path.replace('notebooks/', '');
    return `https://github.com/${repoOwner}/${repoName}/blob/${branch}/notebooks/${notebookPath}`;
  };

  // Get Google Colab URL for the notebook
  const getColabUrl = (path: string) => {
    const githubUrl = getGitHubUrl(path);
    return `https://colab.research.google.com/github/oxnr/BinaryLM/blob/main/notebooks/${path.replace('notebooks/', '')}`;
  };

  // Get Binder URL for the notebook
  const getBinderUrl = (path: string) => {
    const repoOwner = 'oxnr';
    const repoName = 'BinaryLM';
    const branch = 'main';
    const notebookPath = path.replace('notebooks/', '');
    return `https://mybinder.org/v2/gh/${repoOwner}/${repoName}/${branch}?filepath=notebooks/${notebookPath}`;
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

  // Function to render markdown content
  const renderMarkdown = (source: string[]) => {
    const content = source.join('');
    // Very simple markdown rendering for now - this could be improved with a markdown library
    return (
      <div 
        className="markdown-cell"
        dangerouslySetInnerHTML={{ 
          __html: content
            .replace(/^# (.*$)/gm, '<h1>$1</h1>')
            .replace(/^## (.*$)/gm, '<h2>$1</h2>')
            .replace(/^### (.*$)/gm, '<h3>$1</h3>')
            .replace(/^#### (.*$)/gm, '<h4>$1</h4>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/```([a-z]*)\n([\s\S]*?)```/g, '<pre class="code-block"><code>$2</code></pre>')
            .replace(/^- (.*$)/gm, '<li>$1</li>').replace(/<\/li>\n<li>/g, '</li><li>')
            .replace(/^\d+\. (.*$)/gm, '<li>$1</li>').replace(/<\/li>\n<li>/g, '</li><li>')
            .replace(/\n\n/g, '<br/><br/>')
        }} 
      />
    );
  };

  // Function to render code content
  const renderCode = (cell: Cell) => {
    const content = cell.source.join('');
    return (
      <div className="code-cell">
        {cell.execution_count !== null && (
          <div className="execution-count">In [{cell.execution_count || ' '}]:</div>
        )}
        <pre className="code-block">
          <code>{content}</code>
        </pre>
        {cell.outputs && cell.outputs.length > 0 && (
          <div className="cell-outputs">
            {cell.outputs.map((output, i) => (
              <div key={i} className="output">
                {output.output_type === 'stream' && (
                  <pre className="output-text">{output.text.join('')}</pre>
                )}
                {output.output_type === 'execute_result' && (
                  <div>
                    {output.data && output.data['text/plain'] && (
                      <pre className="output-text">{Array.isArray(output.data['text/plain']) 
                        ? output.data['text/plain'].join('') 
                        : output.data['text/plain']}
                      </pre>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  // Render the notebook content
  const renderNotebook = () => {
    if (!notebook) return null;
    
    return notebook.cells.map((cell, index) => (
      <div key={index} className={`cell ${cell.cell_type}-cell`}>
        {cell.cell_type === 'markdown' && renderMarkdown(cell.source)}
        {cell.cell_type === 'code' && renderCode(cell)}
      </div>
    ));
  };

  return (
    <div className="notebook-viewer">
      {title && <h1 className="notebook-title">{title}</h1>}
      
      <div className="notebook-actions">
        <a 
          href={getColabUrl(notebookPath)} 
          target="_blank" 
          rel="noopener noreferrer"
          className="notebook-action-button colab-button"
        >
          <img 
            src="https://colab.research.google.com/assets/colab-badge.svg" 
            alt="Open In Colab" 
            className="button-icon"
          />
          <span>Run in Colab</span>
        </a>
        <a 
          href={getBinderUrl(notebookPath)} 
          target="_blank" 
          rel="noopener noreferrer"
          className="notebook-action-button binder-button"
        >
          <img 
            src="https://mybinder.org/badge_logo.svg" 
            alt="Launch Binder" 
            className="button-icon"
          />
          <span>Launch in Binder</span>
        </a>
        <a 
          href={getGitHubUrl(notebookPath)} 
          target="_blank" 
          rel="noopener noreferrer"
          className="notebook-action-button github-button"
        >
          <svg viewBox="0 0 16 16" width="16" height="16" className="button-icon">
            <path fillRule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
          </svg>
          <span>View on GitHub</span>
        </a>
      </div>
      
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
          {renderNotebook()}
        </div>
      )}
    </div>
  );
};

export default SimpleNotebookViewer; 