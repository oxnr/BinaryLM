import React from 'react';
import { Link } from 'react-router-dom';
import './NotFoundPage.css';

const NotFoundPage: React.FC = () => {
  return (
    <div className="not-found-page">
      <div className="not-found-content">
        <h1 className="error-code">404</h1>
        <h2 className="error-title">Page Not Found</h2>
        <p className="error-message">
          The page you are looking for doesn't exist or has been moved.
        </p>
        <Link to="/" className="btn back-home-btn">
          Back to Home
        </Link>
      </div>
    </div>
  );
};

export default NotFoundPage; 