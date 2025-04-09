import React, { useState } from 'react';
import '../Quiz.css';

interface Question {
  id: number;
  text: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
}

const TransformerArchitectureQuiz: React.FC = () => {
  const questions: Question[] = [
    {
      id: 1,
      text: "What is the key innovation of the transformer architecture compared to RNNs?",
      options: [
        "Transformers process tokens sequentially, one at a time",
        "Transformers use attention mechanisms instead of recurrence",
        "Transformers don't use embeddings for tokens",
        "Transformers can only process a fixed number of tokens"
      ],
      correctAnswer: 1,
      explanation: "The key innovation of transformers is replacing recurrence with self-attention mechanisms, allowing parallel processing of sequences and better handling of long-range dependencies."
    },
    {
      id: 2,
      text: "Which of these is NOT a core component of the transformer architecture?",
      options: [
        "Multi-head attention",
        "Positional encoding",
        "Recurrent connections",
        "Feed-forward networks"
      ],
      correctAnswer: 2,
      explanation: "Recurrent connections are specifically what transformers replaced. The core components include self-attention mechanisms, positional encodings, feed-forward networks, and layer normalization."
    },
    {
      id: 3,
      text: "In a multi-head attention mechanism, what do Q, K, and V represent?",
      options: [
        "Quality, Knowledge, and Value",
        "Quantity, Key, and Vector",
        "Query, Key, and Value",
        "Quantum, Kinetic, and Velocity"
      ],
      correctAnswer: 2,
      explanation: "In attention mechanisms, Q stands for Query (what we're looking for), K stands for Key (what we match against), and V stands for Value (what we retrieve if there's a match)."
    },
    {
      id: 4,
      text: "Why are positional encodings necessary in transformer models?",
      options: [
        "To limit the sequence length",
        "To add information about token positions since transformers process tokens in parallel",
        "To store the model's parameters",
        "To convert tokens into vectors"
      ],
      correctAnswer: 1,
      explanation: "Since transformers process all tokens in parallel (not sequentially like RNNs), they need positional encodings to know the position of each token in the sequence."
    },
    {
      id: 5,
      text: "Which variant of the transformer architecture is used by models like GPT and Claude?",
      options: [
        "Encoder-only",
        "Encoder-decoder",
        "Decoder-only",
        "Hybrid transformer"
      ],
      correctAnswer: 2,
      explanation: "GPT, Claude, and similar large language models use the decoder-only variant of the transformer architecture, which generates text autoregressively."
    },
    {
      id: 6,
      text: "What does the formula Attention(Q, K, V) = softmax(QK^T/√d_k)V represent?",
      options: [
        "The positional encoding calculation",
        "The feed-forward network function",
        "The attention mechanism calculation",
        "The embedding lookup process"
      ],
      correctAnswer: 2,
      explanation: "This formula represents the scaled dot-product attention calculation, which is at the heart of transformer models. It computes attention weights between queries and keys, then applies these weights to values."
    },
    {
      id: 7,
      text: "What is the purpose of layer normalization in transformers?",
      options: [
        "To reduce the number of parameters",
        "To ensure that the output sequence has the right length",
        "To ensure inputs to each sublayer have consistent scale",
        "To prevent overfitting during training"
      ],
      correctAnswer: 2,
      explanation: "Layer normalization helps stabilize training by normalizing the inputs to each sublayer, ensuring that they have a consistent scale (mean of 0 and variance of 1)."
    },
    {
      id: 8,
      text: "How do residual connections benefit transformer models?",
      options: [
        "They reduce the vocabulary size needed",
        "They allow better gradient flow during training",
        "They compress the model size",
        "They enforce causality in the attention mechanism"
      ],
      correctAnswer: 1,
      explanation: "Residual connections (adding the input to the output of a sublayer) help with gradient flow during training, making it easier to train deeper networks by providing a direct path for gradients."
    },
    {
      id: 9,
      text: "Which transformer variant is naturally bidirectional and best suited for classification tasks?",
      options: [
        "Encoder-only (e.g., BERT)",
        "Decoder-only (e.g., GPT)",
        "Encoder-decoder (e.g., T5)",
        "None of the above"
      ],
      correctAnswer: 0,
      explanation: "Encoder-only models like BERT are bidirectional, meaning they can attend to all tokens in the input sequence. This makes them well-suited for classification and understanding tasks."
    },
    {
      id: 10,
      text: "What is the main reason transformers scale effectively with more parameters?",
      options: [
        "They use less memory than other model types",
        "They can be trained on smaller datasets",
        "Their parallel nature allows efficient utilization of computational resources",
        "They don't suffer from overfitting"
      ],
      correctAnswer: 2,
      explanation: "Transformers scale well partially because their parallel nature allows efficient utilization of computational resources like GPUs, enabling training of increasingly larger models with more parameters."
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
      <h1>Transformer Architecture Quiz</h1>
      
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
              <p>Perfect score! You've mastered the transformer architecture concepts.</p>
            ) : score >= questions.length * 0.8 ? (
              <p>Great job! You have a solid understanding of transformer architecture.</p>
            ) : score >= questions.length * 0.6 ? (
              <p>Good work! You understand the basics but might want to review some concepts.</p>
            ) : (
              <p>You might need more study. Consider reviewing the transformer architecture tutorial again.</p>
            )}
          </div>
          
          <button className="restart-button" onClick={restartQuiz}>
            Restart Quiz
          </button>
          
          <a className="tutorial-link" href="/tutorials/transformer-architecture">
            Review Tutorial
          </a>
        </div>
      )}
    </div>
  );
};

export default TransformerArchitectureQuiz; 