import React from 'react';
import { Link } from 'react-router-dom';
import './Header.css';

const Header: React.FC = () => {
  return (
    <header className="main-header">
      <div className="header-container">
        <div className="logo-container">
          <Link to="/" className="logo-link">
            <span className="logo-text">BinaryLM</span>
          </Link>
        </div>
        
        <nav className="main-nav">
          <ul className="nav-list">
            <li className="nav-item">
              <Link to="/tokenization" className="nav-link">Tokenization</Link>
            </li>
            <li className="nav-item">
              <Link to="/model-architecture" className="nav-link">Architecture</Link>
            </li>
            <li className="nav-item">
              <Link to="/model-training" className="nav-link">Training</Link>
            </li>
            <li className="nav-item">
              <Link to="/inference" className="nav-link">Inference</Link>
            </li>
            <li className="nav-item">
              <Link to="/docs" className="nav-link">Docs</Link>
            </li>
            <li className="nav-item">
              <a href="https://github.com/oxnr/BinaryLM" target="_blank" rel="noopener noreferrer" className="nav-link">GitHub</a>
            </li>
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Header; 