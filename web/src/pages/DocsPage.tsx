import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import { Link, useParams, useNavigate } from 'react-router-dom';
import mermaid from 'mermaid';
import DocsExplainer from '../components/DocsExplainer';
import './DocsPage.css';

// Configure Mermaid
mermaid.initialize({
  startOnLoad: true,
  theme: 'neutral',
  securityLevel: 'loose',
  fontFamily: 'JetBrains Mono, monospace',
});

interface DocFile {
  id: string;
  title: string;
  path: string;
}

// API URL - change this to match your backend API
const API_URL = 'http://localhost:5001';

const DocsPage: React.FC = () => {
  const { docId = 'readme' } = useParams<{ docId: string }>();
  const [markdown, setMarkdown] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [showExplainer, setShowExplainer] = useState<string | null>(null);
  const [docFiles, setDocFiles] = useState<DocFile[]>([
    { id: 'readme', title: 'README', path: '/README.md' },
    { id: 'glossary', title: 'Glossary', path: '/Glossary.md' },
    { id: 'masking', title: 'Masking', path: '/masking' },
    { id: 'softmax', title: 'Softmax', path: '/softmax' },
  ]);
  const navigate = useNavigate();
  const mermaidRef = useRef<HTMLDivElement>(null);

  // Fetch markdown content
  useEffect(() => {
    const fetchMarkdown = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const selectedDoc = docFiles.find(doc => doc.id === docId) || docFiles[0];
        
        // Check if this is a special term that should show the explainer
        if (['masking', 'softmax'].includes(selectedDoc.id)) {
          setShowExplainer(selectedDoc.id);
          setMarkdown(''); // No need to fetch markdown
          setLoading(false);
          return;
        }
        
        setShowExplainer(null);
        
        // For development, serve from the public directory using API
        // Instead of directly accessing the file, use the API
        let markdownContent = '';
        
        try {
          // First try to fetch directly from public folder (for development)
          console.log(`Trying to fetch from public folder: ${selectedDoc.path}`);
          const response = await fetch(selectedDoc.path);
          
          if (response.ok) {
            markdownContent = await response.text();
          } else {
            throw new Error("Couldn't load from public folder");
          }
        } catch (publicError) {
          console.log("Failed to load from public, trying API...");
          
          // If that fails, try API fallback
          try {
            // Remove leading slash if present
            const filename = selectedDoc.path.startsWith('/') ? selectedDoc.path.slice(1) : selectedDoc.path;
            const apiResponse = await fetch(`${API_URL}/api/docs/${filename}`);
            
            if (!apiResponse.ok) {
              throw new Error(`API error: ${apiResponse.status} ${apiResponse.statusText}`);
            }
            
            const apiResult = await apiResponse.json();
            markdownContent = apiResult.content;
          } catch (apiError) {
            console.error('API error:', apiError);
            
            // Last resort: use embedded markdown
            if (selectedDoc.id === 'readme') {
              markdownContent = `# BinaryLM Documentation\n\nThis is a fallback README content. The actual markdown file could not be loaded.\n\nPlease make sure the API server is running or the markdown files are properly placed in the public directory.`;
            } else if (selectedDoc.id === 'glossary') {
              markdownContent = `# Glossary\n\nThis is a fallback Glossary content. The actual markdown file could not be loaded.\n\nPlease make sure the API server is running or the markdown files are properly placed in the public directory.`;
            } else {
              throw new Error(`Failed to load documentation: ${apiError instanceof Error ? apiError.message : String(apiError)}`);
            }
          }
        }
        
        setMarkdown(markdownContent);
      } catch (err) {
        console.error('Error fetching markdown:', err);
        setError('Failed to load documentation. Please try again later.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchMarkdown();
  }, [docId, docFiles]);

  // Process Mermaid diagrams after markdown is rendered
  useEffect(() => {
    if (!loading && mermaidRef.current) {
      try {
        // Find all mermaid code blocks
        const mermaidBlocks = mermaidRef.current.querySelectorAll('.language-mermaid');
        
        // Process each mermaid diagram
        mermaidBlocks.forEach((block, index) => {
          const code = block.textContent || '';
          
          // Create a container for the rendered diagram
          const diagramContainer = document.createElement('div');
          diagramContainer.className = 'mermaid-diagram';
          diagramContainer.id = `mermaid-diagram-${index}`;
          
          // Replace the code block with the container
          block.parentNode?.replaceChild(diagramContainer, block);
          
          // Render the diagram
          mermaid.render(`mermaid-${index}`, code)
            .then(({ svg }) => {
              const container = document.getElementById(`mermaid-diagram-${index}`);
              if (container) {
                container.innerHTML = svg;
              }
            })
            .catch(err => {
              console.error('Mermaid rendering error:', err);
            });
        });
      } catch (err) {
        console.error('Error processing mermaid diagrams:', err);
      }
    }
  }, [markdown, loading]);

  // Custom components for ReactMarkdown
  const components = {
    a: ({ href, children }: { href?: string; children?: React.ReactNode }) => {
      // If it's an internal link to another markdown file, use React Router
      if (href && href.endsWith('.md')) {
        const targetId = href.replace('.md', '').toLowerCase();
        return (
          <Link to={`/docs/${targetId}`}>
            {children}
          </Link>
        );
      }
      
      // External links
      return (
        <a href={href} target="_blank" rel="noopener noreferrer">
          {children}
        </a>
      );
    }
  };

  return (
    <div className="docs-page">
      <aside className="docs-sidebar">
        <h2>Documentation</h2>
        <nav className="docs-nav">
          <ul>
            {docFiles.map(doc => (
              <li key={doc.id} className={docId === doc.id ? 'active' : ''}>
                <Link to={`/docs/${doc.id}`}>{doc.title}</Link>
              </li>
            ))}
          </ul>
        </nav>
        <div className="docs-github-link">
          <a 
            href="https://github.com/oxnr/BinaryLM" 
            target="_blank" 
            rel="noopener noreferrer"
          >
            View on GitHub
          </a>
        </div>
      </aside>
      
      <main className="docs-content" ref={mermaidRef}>
        {loading ? (
          <div className="docs-loading">Loading documentation...</div>
        ) : error ? (
          <div className="docs-error">{error}</div>
        ) : showExplainer ? (
          <>
            <div className="docs-actions">
              <select 
                value={docId} 
                onChange={(e) => navigate(`/docs/${e.target.value}`)}
                className="docs-mobile-nav"
              >
                {docFiles.map(doc => (
                  <option key={doc.id} value={doc.id}>{doc.title}</option>
                ))}
              </select>
            </div>
            
            <DocsExplainer term={showExplainer} />
          </>
        ) : (
          <>
            <div className="docs-actions">
              <select 
                value={docId} 
                onChange={(e) => navigate(`/docs/${e.target.value}`)}
                className="docs-mobile-nav"
              >
                {docFiles.map(doc => (
                  <option key={doc.id} value={doc.id}>{doc.title}</option>
                ))}
              </select>
            </div>
            
            <div className="markdown-content">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeRaw]}
                components={components}
              >
                {markdown}
              </ReactMarkdown>
            </div>
          </>
        )}
      </main>
    </div>
  );
};

export default DocsPage; 