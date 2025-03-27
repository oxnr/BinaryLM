import React, { useState } from 'react';
import './TutorialQuiz.css';

export interface QuizQuestion {
  id: string;
  question: string;
  options: string[];
  correctOptionIndex: number;
  explanation: string;
}

interface TutorialQuizProps {
  questions: QuizQuestion[];
  onComplete: (score: number, totalQuestions: number) => void;
}

const TutorialQuiz: React.FC<TutorialQuizProps> = ({ questions, onComplete }) => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedOptionIndices, setSelectedOptionIndices] = useState<number[]>(Array(questions.length).fill(-1));
  const [showExplanation, setShowExplanation] = useState(false);
  const [quizCompleted, setQuizCompleted] = useState(false);
  
  const currentQuestion = questions[currentQuestionIndex];
  const hasSelectedOption = selectedOptionIndices[currentQuestionIndex] !== -1;
  const isCorrect = selectedOptionIndices[currentQuestionIndex] === currentQuestion.correctOptionIndex;
  
  const handleOptionSelect = (optionIndex: number) => {
    if (showExplanation) return; // Don't allow changing answer after checking
    
    const newSelectedOptions = [...selectedOptionIndices];
    newSelectedOptions[currentQuestionIndex] = optionIndex;
    setSelectedOptionIndices(newSelectedOptions);
  };
  
  const handleCheckAnswer = () => {
    setShowExplanation(true);
  };
  
  const handleNextQuestion = () => {
    setShowExplanation(false);
    
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    } else {
      // All questions answered
      const correctAnswers = selectedOptionIndices.filter(
        (selectedIndex, i) => selectedIndex === questions[i].correctOptionIndex
      ).length;
      
      setQuizCompleted(true);
      onComplete(correctAnswers, questions.length);
    }
  };
  
  const calculateProgress = () => {
    return Math.round(((currentQuestionIndex + 1) / questions.length) * 100);
  };
  
  if (quizCompleted) {
    const score = selectedOptionIndices.filter(
      (selectedIndex, i) => selectedIndex === questions[i].correctOptionIndex
    ).length;
    
    return (
      <div className="tutorial-quiz-completed">
        <h2>Quiz Completed!</h2>
        <div className="quiz-score">
          <p>Your score: <strong>{score}</strong> out of <strong>{questions.length}</strong></p>
          <p className="score-percentage">({Math.round((score / questions.length) * 100)}%)</p>
        </div>
        {score === questions.length ? (
          <div className="perfect-score">
            <span className="score-icon">🎉</span>
            <p>Great job! You've mastered this topic!</p>
          </div>
        ) : score >= questions.length * 0.7 ? (
          <div className="good-score">
            <span className="score-icon">👍</span>
            <p>Good work! You have a solid understanding of this topic.</p>
          </div>
        ) : (
          <div className="needs-review">
            <span className="score-icon">📚</span>
            <p>You might want to review this section again to strengthen your understanding.</p>
          </div>
        )}
      </div>
    );
  }
  
  return (
    <div className="tutorial-quiz">
      <div className="quiz-progress">
        <div className="progress-bar">
          <div 
            className="progress-fill"
            style={{ width: `${calculateProgress()}%` }}
          ></div>
        </div>
        <div className="progress-text">
          Question {currentQuestionIndex + 1} of {questions.length}
        </div>
      </div>
      
      <div className="quiz-question">
        <h3>{currentQuestion.question}</h3>
        
        <div className="quiz-options">
          {currentQuestion.options.map((option, index) => (
            <div 
              key={index}
              className={`quiz-option ${selectedOptionIndices[currentQuestionIndex] === index ? 'selected' : ''} ${
                showExplanation ? (
                  index === currentQuestion.correctOptionIndex ? 'correct' : 
                  selectedOptionIndices[currentQuestionIndex] === index ? 'incorrect' : ''
                ) : ''
              }`}
              onClick={() => handleOptionSelect(index)}
            >
              <div className="option-marker">{String.fromCharCode(65 + index)}</div>
              <div className="option-content">{option}</div>
              {showExplanation && index === currentQuestion.correctOptionIndex && (
                <div className="correct-indicator">✓</div>
              )}
            </div>
          ))}
        </div>
        
        {showExplanation && (
          <div className="answer-explanation">
            <h4>{isCorrect ? 'Correct!' : 'Not quite right.'}</h4>
            <p>{currentQuestion.explanation}</p>
          </div>
        )}
        
        <div className="quiz-actions">
          {!showExplanation ? (
            <button 
              className="check-answer-button"
              onClick={handleCheckAnswer}
              disabled={!hasSelectedOption}
            >
              Check Answer
            </button>
          ) : (
            <button 
              className="next-question-button"
              onClick={handleNextQuestion}
            >
              {currentQuestionIndex < questions.length - 1 ? 'Next Question' : 'Complete Quiz'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default TutorialQuiz; 