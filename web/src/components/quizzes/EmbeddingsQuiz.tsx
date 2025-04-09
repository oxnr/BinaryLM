import React, { useState } from 'react';
import '../Quiz.css';

interface Question {
  id: number;
  text: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
}

const EmbeddingsQuiz: React.FC = () => {
  const questions: Question[] = [
    {
      id: 1,
      text: "What is the primary purpose of embeddings in language models?",
      options: [
        "To compress the model size",
        "To convert discrete tokens into continuous vector representations",
        "To accelerate training speed",
        "To filter out rare tokens from the vocabulary"
      ],
      correctAnswer: 1,
      explanation: "Embeddings transform discrete tokens (like words or subwords) into continuous vector spaces where semantic relationships can be captured mathematically."
    },
    {
      id: 2,
      text: "What property allows word embeddings to capture semantic relationships?",
      options: [
        "Words with similar meanings have vectors that are close in the embedding space",
        "Embeddings are stored as integers rather than floating-point numbers",
        "Embeddings have exactly the same dimensionality as the vocabulary size",
        "Words are sorted alphabetically in the embedding space"
      ],
      correctAnswer: 0,
      explanation: "A key property of good embeddings is that semantically similar words have vectors that are close to each other in the embedding space, allowing the model to understand relationships between concepts."
    },
    {
      id: 3,
      text: "Which of these is NOT a type of embedding commonly used in transformer models?",
      options: [
        "Token embeddings",
        "Positional embeddings",
        "Recurrent embeddings",
        "Segment/type embeddings"
      ],
      correctAnswer: 2,
      explanation: "Transformers don't use recurrent embeddings, as they replaced recurrence with attention. Common embeddings in transformers include token embeddings, positional embeddings, and sometimes segment/type embeddings."
    },
    {
      id: 4,
      text: "What is the difference between static word embeddings and contextual embeddings?",
      options: [
        "Static embeddings use larger vectors than contextual embeddings",
        "Static embeddings are learned during inference, while contextual embeddings are learned during training",
        "Static embeddings give each word a single fixed vector regardless of context, while contextual embeddings depend on surrounding words",
        "Static embeddings are only used in older models, while all modern models use contextual embeddings exclusively"
      ],
      correctAnswer: 2,
      explanation: "In static word embeddings (like Word2Vec), each word has exactly one embedding regardless of context. In contextual embeddings (like those in BERT or GPT), the same word can have different vector representations depending on its context."
    },
    {
      id: 5,
      text: "What mathematical operation is typically used to combine token embeddings and positional embeddings?",
      options: [
        "Multiplication",
        "Concatenation",
        "Addition",
        "Subtraction"
      ],
      correctAnswer: 2,
      explanation: "Token embeddings and positional embeddings are typically added together: input_representation = token_embedding + positional_embedding. This allows the model to consider both the token identity and its position."
    },
    {
      id: 6,
      text: "What does the dimensionality of embeddings (e.g., 768 or 1024) affect in a language model?",
      options: [
        "The maximum sequence length the model can process",
        "The number of attention heads in the model",
        "The amount of information each embedding can encode",
        "The vocabulary size of the model"
      ],
      correctAnswer: 2,
      explanation: "The dimensionality of embeddings affects how much information each embedding can encode. Higher dimensions can represent more complex relationships but require more parameters and computation."
    },
    {
      id: 7,
      text: "What is an example of an analogical relationship that well-trained embeddings can capture?",
      options: [
        "Larger words have larger embedding vectors",
        "Nouns have positive values, verbs have negative values",
        "king - man + woman ≈ queen",
        "Words with the same number of letters have similar embeddings"
      ],
      correctAnswer: 2,
      explanation: "Well-trained embeddings can capture analogical relationships such as 'king - man + woman ≈ queen', showing that the model has learned meaningful semantic and syntactic relationships between concepts."
    },
    {
      id: 8,
      text: "How are embeddings typically used in a Retrieval-Augmented Generation (RAG) system?",
      options: [
        "Embeddings are used to rank different pre-trained language models",
        "Documents and queries are embedded in the same space to find relevant information via similarity",
        "Embeddings are used to compress the size of the retrieved documents",
        "Embedding layers are skipped in RAG to improve retrieval speed"
      ],
      correctAnswer: 1,
      explanation: "In RAG systems, both documents and queries are embedded into the same vector space, allowing the system to find relevant information by measuring the similarity between the query embedding and document embeddings."
    },
    {
      id: 9,
      text: "What makes positional embeddings necessary in transformer models?",
      options: [
        "Transformers process tokens sequentially and need to know the order",
        "Transformers process tokens in parallel and need positional information to understand token order",
        "Positional embeddings help reduce the size of the model",
        "Positional embeddings simplify the attention mechanism calculations"
      ],
      correctAnswer: 1,
      explanation: "Since transformers process all tokens in parallel rather than sequentially, they need positional embeddings to add information about where each token appears in the sequence."
    },
    {
      id: 10,
      text: "What technique is commonly used to visualize high-dimensional embeddings in 2D or 3D?",
      options: [
        "Attention map visualization",
        "Matrix multiplication",
        "Dimensionality reduction (like t-SNE or PCA)",
        "Vectorization"
      ],
      correctAnswer: 2,
      explanation: "To visualize high-dimensional embeddings (which typically have hundreds or thousands of dimensions), dimensionality reduction techniques like t-SNE (t-Distributed Stochastic Neighbor Embedding) or PCA (Principal Component Analysis) are used to project them to 2D or 3D."
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
      <h1>Embeddings in LLMs Quiz</h1>
      
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
              <p>Perfect score! You have an excellent understanding of embeddings in language models.</p>
            ) : score >= questions.length * 0.8 ? (
              <p>Great job! You have a solid grasp of how embeddings work in LLMs.</p>
            ) : score >= questions.length * 0.6 ? (
              <p>Good work! You understand the basics of embeddings but might want to review some concepts.</p>
            ) : (
              <p>You might need more study. Consider reviewing the embeddings tutorial again to strengthen your understanding.</p>
            )}
          </div>
          
          <button className="restart-button" onClick={restartQuiz}>
            Restart Quiz
          </button>
          
          <a className="tutorial-link" href="/tutorials/embeddings">
            Review Tutorial
          </a>
        </div>
      )}
    </div>
  );
};

export default EmbeddingsQuiz; 