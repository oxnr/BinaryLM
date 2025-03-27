import React from 'react';
import { NavLink } from 'react-router-dom';

const Navbar: React.FC = () => {
  return (
    <header className="header">
      <h1>BinaryLM Explorer</h1>
      <nav className="nav-links">
        <NavLink to="/">Home</NavLink>
        <NavLink to="/tokenizer">Tokenizer</NavLink>
        <NavLink to="/model">Model Architecture</NavLink>
        <NavLink to="/training">Training</NavLink>
      </nav>
    </header>
  );
};

export default Navbar; 