import React, { useState } from 'react';
import '../Quiz.css';

interface Question {
  id: number;
  text: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
}

const SoftmaxQuiz: React.FC = () => {
  const questions: Question[] = [
    {
      id: 1,
      text: "What is the primary purpose of the softmax function in language models?",
      options: [
        "To reduce model parameters",
        "To convert raw logits into a probability distribution",
        "To normalize the input embeddings",
        "To compute attention weights between tokens"
      ],
      correctAnswer: 1,
      explanation: "The softmax function converts a vector of raw scores (logits) into a probability distribution where all values are between 0 and 1 and sum to exactly 1. This makes it suitable for selecting the next token during generation."
    },
    {
      id: 2,
      text: "What mathematical property do the outputs of softmax always have?",
      options: [
        "They are all integers between 0 and 100",
        "They are all between 0 and 1 and sum to exactly 1",
        "They are all positive or negative real numbers",
        "They follow a normal distribution with mean 0"
      ],
      correctAnswer: 1,
      explanation: "Softmax outputs are always between 0 and 1 (making them valid probabilities) and they sum to exactly 1 (making them a valid probability distribution)."
    },
    {
      id: 3,
      text: "In the softmax formula, what is the purpose of the exponential function (e^x)?",
      options: [
        "To ensure all values are positive",
        "To normalize the values to be between 0 and 1",
        "To increase computation speed",
        "To make the formula differentiable"
      ],
      correctAnswer: 0,
      explanation: "The exponential function (e^x) ensures that all values in the output are positive, which is necessary for a probability distribution. Even negative inputs become positive after applying e^x."
    },
    {
      id: 4,
      text: "Where is softmax applied in the attention mechanism?",
      options: [
        "To the input embeddings",
        "To the query vectors only",
        "To the scaled dot products of queries and keys",
        "To the output of the feed-forward networks"
      ],
      correctAnswer: 2,
      explanation: "In the attention mechanism, softmax is applied to the scaled dot products of queries and keys: Attention(Q, K, V) = softmax(QK^T/√d_k)V. This converts the raw attention scores into attention weights."
    },
    {
      id: 5,
      text: "What happens when temperature is applied to logits before softmax?",
      options: [
        "Lower temperature makes the distribution more uniform (random)",
        "Higher temperature makes the distribution more peaky (deterministic)",
        "Temperature has no effect on the probabilities after softmax",
        "Lower temperature makes the distribution more peaky (deterministic)"
      ],
      correctAnswer: 3,
      explanation: "Lower temperature (T < 1) makes the probability distribution more 'peaky', giving more weight to higher-probability tokens. As T approaches 0, the selection becomes more deterministic (like argmax)."
    },
    {
      id: 6,
      text: "If we have logits [2.0, 1.0, 0.1] and apply softmax, which token has the highest probability?",
      options: [
        "The token with logit 0.1",
        "The token with logit 1.0",
        "The token with logit 2.0",
        "All tokens have equal probability"
      ],
      correctAnswer: 2,
      explanation: "Softmax preserves the relative ordering of the input logits, so the token with the highest logit (2.0) will have the highest probability after applying softmax."
    },
    {
      id: 7,
      text: "What is a common numerical issue when computing softmax directly?",
      options: [
        "Integer overflow due to large sums",
        "Numerical underflow due to small products",
        "Numerical overflow due to large exponentials",
        "Division by zero when normalizing"
      ],
      correctAnswer: 2,
      explanation: "Computing softmax directly can lead to numerical overflow due to large exponentials. The standard solution is to subtract the maximum value from all elements before applying the exponential, which doesn't change the result but improves numerical stability."
    },
    {
      id: 8,
      text: "What is the 'temperature' parameter in sampling with softmax?",
      options: [
        "A measure of how hot the GPU gets during computation",
        "A divisor applied to logits before softmax to control randomness",
        "The number of tokens to consider in the distribution",
        "The cooling rate for the learning process"
      ],
      correctAnswer: 1,
      explanation: "Temperature is a parameter used to divide the logits before applying softmax: softmax(logits/T). It controls the randomness of the sampling process by adjusting how 'peaky' or 'flat' the probability distribution becomes."
    },
    {
      id: 9,
      text: "What happens if you set the temperature to a very high value (e.g., T = 100) when sampling?",
      options: [
        "The model will always select the highest probability token",
        "The model will crash due to numerical overflow",
        "The probability distribution becomes nearly uniform (random selection)",
        "The generation speed increases significantly"
      ],
      correctAnswer: 2,
      explanation: "As temperature approaches infinity, the probability distribution becomes more and more uniform, approaching random selection where all tokens have equal probability regardless of their logits."
    },
    {
      id: 10,
      text: "What is the log_softmax function and why is it useful?",
      options: [
        "It's a faster approximation of softmax with lower accuracy",
        "It directly computes the logarithm of softmax for better numerical stability",
        "It's used only for classification tasks, not for language generation",
        "It allows softmax to work with negative numbers"
      ],
      correctAnswer: 1,
      explanation: "log_softmax directly computes the logarithm of the softmax values. This is numerically more stable for computing cross-entropy loss, which involves taking the log of probabilities anyway, and helps avoid potential underflow issues."
    }
  ];

  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedOption, setSelectedOption] = useState<number | null>(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [score, setScore] = useState(0);
  const [quizComplete, setQuizComplete] = useState(false);

  const handleOptionSelect = (optionIndex: number) => {
    if (selectedOption !== null) return; // Prevent changing answer
    setSelectedOption(optionIndex);
    setShowExplanation(true);
    
    if (optionIndex === questions[currentQuestionIndex].correctAnswer) {
      setScore(score + 1);
    }
  };

  const handleNextQuestion = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
      setSelectedOption(null);
      setShowExplanation(false);
    } else {
      setQuizComplete(true);
    }
  };

  const restartQuiz = () => {
    setCurrentQuestionIndex(0);
    setSelectedOption(null);
    setShowExplanation(false);
    setScore(0);
    setQuizComplete(false);
  };

  const currentQuestion = questions[currentQuestionIndex];

  return (
    <div className="quiz-container">
      <h1>Softmax Function Quiz</h1>
      
      {!quizComplete ? (
        <div className="quiz-content">
          <div className="quiz-progress">
            <div className="progress-text">
              Question {currentQuestionIndex + 1} of {questions.length}
            </div>
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${((currentQuestionIndex + 1) / questions.length) * 100}%` }}
              ></div>
            </div>
          </div>
          
          <div className="question-card">
            <h2 className="question-text">{currentQuestion.text}</h2>
            
            <div className="options-container">
              {currentQuestion.options.map((option, index) => (
                <button
                  key={index}
                  className={`option-button ${
                    selectedOption === index
                      ? index === currentQuestion.correctAnswer
                        ? 'correct'
                        : 'incorrect'
                      : ''
                  } ${selectedOption !== null && index === currentQuestion.correctAnswer ? 'correct-answer' : ''}`}
                  onClick={() => handleOptionSelect(index)}
                  disabled={selectedOption !== null}
                >
                  {option}
                </button>
              ))}
            </div>
            
            {showExplanation && (
              <div className="explanation">
                <h3>Explanation:</h3>
                <p>{currentQuestion.explanation}</p>
              </div>
            )}
            
            {selectedOption !== null && (
              <button 
                className="next-button"
                onClick={handleNextQuestion}
              >
                {currentQuestionIndex < questions.length - 1 ? 'Next Question' : 'Finish Quiz'}
              </button>
            )}
          </div>
        </div>
      ) : (
        <div className="quiz-results">
          <h2>Quiz Complete!</h2>
          <div className="score-display">
            <div className="score">
              {score} / {questions.length}
            </div>
            <div className="score-percentage">
              {Math.round((score / questions.length) * 100)}%
            </div>
          </div>
          
          <div className="score-message">
            {score === questions.length ? (
              <p>Perfect score! You have mastered the softmax function concepts.</p>
            ) : score >= questions.length * 0.8 ? (
              <p>Great job! You have a solid understanding of how softmax works in LLMs.</p>
            ) : score >= questions.length * 0.6 ? (
              <p>Good work! You understand the basics of the softmax function but might want to review some concepts.</p>
            ) : (
              <p>You might need more study. Consider reviewing the softmax tutorial to strengthen your understanding.</p>
            )}
          </div>
          
          <button className="restart-button" onClick={restartQuiz}>
            Restart Quiz
          </button>
          
          <a className="tutorial-link" href="/tutorials/softmax">
            Review Tutorial
          </a>
        </div>
      )}
    </div>
  );
};

export default SoftmaxQuiz; 