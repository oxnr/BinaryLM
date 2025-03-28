import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import ModelTraining from './components/ModelTraining';
import TokenizationVisualizer from './components/TokenizationVisualizer';
import ModelArchitecture from './pages/ModelArchitecture';
import HomePage from './pages/HomePage';
import NotFoundPage from './pages/NotFoundPage';
import TokenizerExplorer from './pages/TokenizerExplorer';
import InferencePage from './pages/InferencePage';
import DocsPage from './pages/DocsPage';
import TutorialsPage from './pages/TutorialsPage';
import NotebooksPage from './pages/NotebooksPage';
import './App.css';

function App() {
  return (
    <div className="app">
      <Header />
      <main className="main-content">
        <div className="container">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/tokenization" element={
              <div className="page-content">
                <h1>Tokenization Visualizer</h1>
                <p>See how text gets broken down into tokens that a language model can understand.</p>
                <TokenizationVisualizer />
              </div>
            } />
            <Route path="/tokenizer" element={<TokenizerExplorer />} />
            <Route path="/model-architecture" element={<ModelArchitecture />} />
            <Route path="/model-training" element={<ModelTraining />} />
            <Route path="/inference" element={<InferencePage />} />
            <Route path="/docs" element={<DocsPage />} />
            <Route path="/docs/:docId" element={<DocsPage />} />
            <Route path="/tutorials" element={<TutorialsPage />} />
            <Route path="/notebooks" element={<NotebooksPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </div>
      </main>
      <footer className="main-footer">
        <div className="container">
          <p>© 2023 BinaryLM - Educational LLM Platform | <a href="https://github.com/oxnr/BinaryLM" target="_blank" rel="noopener noreferrer">GitHub</a></p>
        </div>
      </footer>
    </div>
  );
}

export default App; 