import React, { useState } from 'react';
import '../Quiz.css';

interface Question {
  id: number;
  text: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
}

const InferenceQuiz: React.FC = () => {
  const questions: Question[] = [
    {
      id: 1,
      text: "What is the fundamental approach to text generation in LLMs?",
      options: [
        "Generating all tokens simultaneously",
        "Generating text backwards from a conclusion",
        "Autoregressive generation, one token at a time",
        "Predicting the middle token first, then expanding outward"
      ],
      correctAnswer: 2,
      explanation: "LLM inference is fundamentally autoregressive: the model predicts one token at a time, with each new token depending on all previously generated tokens."
    },
    {
      id: 2,
      text: "In greedy decoding, how is the next token selected?",
      options: [
        "Randomly from the top 5 tokens",
        "The token with the highest probability",
        "The token that appears most frequently in the training data",
        "The token that makes the sequence most diverse"
      ],
      correctAnswer: 1,
      explanation: "Greedy decoding simply selects the token with the highest probability at each step. While this is deterministic and computationally efficient, it can lead to repetitive text and inability to recover from mistakes."
    },
    {
      id: 3,
      text: "What does temperature sampling control in text generation?",
      options: [
        "The length of the generated text",
        "The randomness vs. determinism of token selection",
        "The computational resources used during inference",
        "The time taken for generating each token"
      ],
      correctAnswer: 1,
      explanation: "Temperature sampling controls the randomness versus determinism of token selection. Lower values (e.g., 0.1) make the model more deterministic, while higher values (e.g., 1.5) make it more random and creative."
    },
    {
      id: 4,
      text: "What is Top-K sampling?",
      options: [
        "Selecting the K most probable tokens and sampling from only those",
        "Generating K different outputs and selecting the best one",
        "Using K different models and averaging their predictions",
        "Running the model K times for better precision"
      ],
      correctAnswer: 0,
      explanation: "Top-K sampling restricts the sampling pool to only the K tokens with the highest probabilities. This prevents the model from selecting extremely unlikely tokens while still maintaining some randomness."
    },
    {
      id: 5,
      text: "What makes Nucleus (Top-p) sampling different from Top-K?",
      options: [
        "It uses multiple GPUs for faster inference",
        "It considers the smallest set of tokens whose cumulative probability exceeds p",
        "It only works with certain model architectures",
        "It requires pre-processing the entire corpus first"
      ],
      correctAnswer: 1,
      explanation: "Nucleus (Top-p) sampling dynamically adjusts the number of tokens considered by sampling from the smallest set of tokens whose cumulative probability exceeds the threshold p (e.g., 0.9). This adapts to the confidence of the model in each context."
    },
    {
      id: 6,
      text: "What is a key advantage of beam search over simple sampling methods?",
      options: [
        "It's much faster than other methods",
        "It uses less memory",
        "It explores multiple promising paths simultaneously",
        "It works with any size vocabulary"
      ],
      correctAnswer: 2,
      explanation: "Beam search explores multiple promising paths (beams) simultaneously and keeps track of the most likely sequences overall. This can lead to more coherent outputs than simple sampling methods, especially for tasks with more deterministic answers."
    },
    {
      id: 7,
      text: "What is KV caching in LLM inference?",
      options: [
        "A method to compress the model for faster loading",
        "Storing and reusing key-value pairs from previous tokens to avoid redundant computation",
        "A technique to quantize the model to lower precision",
        "A way to cache the entire vocabulary for faster lookups"
      ],
      correctAnswer: 1,
      explanation: "KV caching is an optimization technique that stores and reuses the key-value pairs computed for previous tokens, avoiding redundant computation since these values don't change when generating new tokens."
    },
    {
      id: 8,
      text: "How does temperature affect the probability distribution in sampling?",
      options: [
        "It has no effect on the distribution",
        "Higher temperature makes high-probability tokens even more likely",
        "Lower temperature makes the distribution more uniform (random)",
        "Lower temperature makes high-probability tokens even more likely"
      ],
      correctAnswer: 3,
      explanation: "Lower temperature (T < 1) makes the probability distribution more 'peaky', emphasizing high-probability tokens. Higher temperature (T > 1) flattens the distribution, making it more uniform and increasing randomness."
    },
    {
      id: 9,
      text: "What is speculative sampling in the context of LLM inference?",
      options: [
        "Using a smaller model to draft multiple tokens for verification by the main model",
        "Generating text based on market speculation data",
        "Making speculative investments based on LLM outputs",
        "A random sampling method that sometimes produces incorrect output"
      ],
      correctAnswer: 0,
      explanation: "Speculative sampling (or decoding) uses a smaller, faster 'draft' model to suggest multiple tokens at once, which are then verified by the main model. This can significantly speed up inference by generating multiple tokens in a single forward pass of the large model."
    },
    {
      id: 10,
      text: "What is the main benefit of model quantization for inference?",
      options: [
        "It improves model accuracy",
        "It allows for longer context windows",
        "It reduces memory usage and increases inference speed",
        "It enables the model to generate more creative text"
      ],
      correctAnswer: 2,
      explanation: "Quantization reduces the precision of model weights (e.g., from FP32 to INT8 or INT4), decreasing memory usage and increasing inference speed, usually with minimal impact on quality when done properly."
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
      <h1>LLM Inference Techniques Quiz</h1>
      
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
              <p>Perfect score! You have mastered LLM inference techniques.</p>
            ) : score >= questions.length * 0.8 ? (
              <p>Great job! You have a solid understanding of how inference works in LLMs.</p>
            ) : score >= questions.length * 0.6 ? (
              <p>Good work! You understand the basics of LLM inference but might want to review some techniques.</p>
            ) : (
              <p>You might need more study. Consider reviewing the inference tutorial to strengthen your understanding.</p>
            )}
          </div>
          
          <button className="restart-button" onClick={restartQuiz}>
            Restart Quiz
          </button>
          
          <a className="tutorial-link" href="/tutorials/inference">
            Review Tutorial
          </a>
        </div>
      )}
    </div>
  );
};

export default InferenceQuiz; 