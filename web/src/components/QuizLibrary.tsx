import React, { useState } from 'react';
import TutorialQuiz, { QuizQuestion } from './TutorialQuiz';
import './QuizLibrary.css';

interface QuizTopic {
  id: string;
  title: string;
  description: string;
  questions: QuizQuestion[];
  difficulty: 'beginner' | 'intermediate' | 'advanced';
}

const QuizLibrary: React.FC<{ quizTopics: QuizTopic[] }> = ({ quizTopics }) => {
  const [selectedQuizId, setSelectedQuizId] = useState<string | null>(null);
  const [completedQuizzes, setCompletedQuizzes] = useState<Record<string, { score: number, total: number }>>({});
  
  const selectedQuiz = quizTopics.find(quiz => quiz.id === selectedQuizId);
  
  const handleQuizComplete = (score: number, totalQuestions: number) => {
    if (selectedQuizId) {
      setCompletedQuizzes(prev => ({
        ...prev,
        [selectedQuizId]: { score, total: totalQuestions }
      }));
    }
  };
  
  const handleBackToLibrary = () => {
    setSelectedQuizId(null);
  };
  
  const renderQuizCatalog = () => (
    <div className="quiz-catalog">
      <h1>Quiz Library</h1>
      <p className="catalog-description">
        Test your knowledge of language model concepts with these quizzes
      </p>
      
      <div className="quiz-grid">
        {quizTopics.map(quiz => (
          <div 
            key={quiz.id} 
            className="quiz-card"
            onClick={() => setSelectedQuizId(quiz.id)}
          >
            <h2>{quiz.title}</h2>
            <p>{quiz.description}</p>
            <div className="quiz-meta">
              <span className={`difficulty-badge ${quiz.difficulty}`}>{quiz.difficulty}</span>
              <span className="questions-count">{quiz.questions.length} questions</span>
            </div>
            {completedQuizzes[quiz.id] && (
              <div className="quiz-score-badge">
                Score: {completedQuizzes[quiz.id].score}/{completedQuizzes[quiz.id].total}
              </div>
            )}
            <button className="start-quiz-button">Start Quiz</button>
          </div>
        ))}
      </div>
    </div>
  );
  
  return (
    <div className="quiz-library-container">
      {selectedQuiz ? (
        <div className="active-quiz-container">
          <button 
            className="back-to-library"
            onClick={handleBackToLibrary}
          >
            ← Back to Quiz Library
          </button>
          <h2>{selectedQuiz.title}</h2>
          <TutorialQuiz 
            questions={selectedQuiz.questions} 
            onComplete={handleQuizComplete} 
          />
        </div>
      ) : (
        renderQuizCatalog()
      )}
    </div>
  );
};

export default QuizLibrary; 