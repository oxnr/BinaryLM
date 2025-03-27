import React, { useEffect, useRef } from 'react';
import mermaid from 'mermaid';

interface MermaidDiagramProps {
  chart: string;
  className?: string;
}

const MermaidDiagram: React.FC<MermaidDiagramProps> = ({ chart, className }) => {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current) {
      mermaid.initialize({ 
        startOnLoad: true,
        theme: 'neutral',
        securityLevel: 'loose',
        themeVariables: {
          primaryColor: '#5d8aa8',
          primaryTextColor: '#fff',
          primaryBorderColor: '#5d8aa8',
          lineColor: '#555',
          secondaryColor: '#006100',
          tertiaryColor: '#fff'
        }
      });
      
      // Clear previous rendered diagrams
      ref.current.innerHTML = '';
      
      try {
        const id = 'mermaid-diagram-' + Math.random().toString(36).substring(2, 9);
        mermaid.render(id, chart).then(({ svg }) => {
          if (ref.current) {
            ref.current.innerHTML = svg;
          }
        });
      } catch (error) {
        console.error('Mermaid rendering error:', error);
        if (ref.current) {
          ref.current.innerHTML = '<div class="mermaid-error">Diagram rendering error</div>';
        }
      }
    }
  }, [chart]);

  return <div ref={ref} className={`mermaid-diagram ${className || ''}`}></div>;
};

export default MermaidDiagram; 