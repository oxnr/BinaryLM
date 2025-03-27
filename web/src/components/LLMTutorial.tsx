import React, { useState, useEffect, useRef } from 'react';
import mermaid from 'mermaid';
import TutorialQuiz, { QuizQuestion } from './TutorialQuiz';
import './LLMTutorial.css';

// These should match the interfaces in TutorialsPage.tsx
interface TutorialStep {
  id: string;
  title: string;
  content: React.ReactNode;
  image?: string;
  mermaidDiagram?: string;
  codeExample?: string;
}

interface TutorialSection {
  id: string;
  title: string;
  description: string;
  steps: TutorialStep[];
  quiz?: QuizQuestion[];
}

interface Tutorial {
  id: string;
  title: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedTime: string;
  sections: TutorialSection[];
}

interface LLMTutorialProps {
  tutorial: Tutorial;
}

const LLMTutorial: React.FC<LLMTutorialProps> = ({ tutorial }) => {
  const [activeSection, setActiveSection] = useState<string>(tutorial.sections[0].id);
  const [activeStep, setActiveStep] = useState<string>(tutorial.sections[0].steps[0].id);
  const [showAllSteps, setShowAllSteps] = useState<boolean>(false);
  const [completedSteps, setCompletedSteps] = useState<Set<string>>(new Set());
  const [completedQuizzes, setCompletedQuizzes] = useState<Set<string>>(new Set());
  const [showingQuiz, setShowingQuiz] = useState<boolean>(false);
  const [quizScores, setQuizScores] = useState<Record<string, { score: number, total: number }>>({});
  
  const currentSection = tutorial.sections.find(section => section.id === activeSection);
  const currentStep = currentSection?.steps.find(step => step.id === activeStep);
  const mermaidRef = useRef<HTMLDivElement>(null);
  
  // Process Mermaid diagrams when step changes
  useEffect(() => {
    if (currentStep?.mermaidDiagram && mermaidRef.current) {
      try {
        mermaid.initialize({
          startOnLoad: true,
          theme: 'neutral',
          securityLevel: 'loose',
          fontFamily: 'sans-serif',
        });
        
        // Clear previous rendered diagrams
        const diagramContainer = mermaidRef.current;
        diagramContainer.innerHTML = '';
        
        // Create a unique ID for this diagram
        const id = `mermaid-${activeSection}-${activeStep}`;
        
        // Render the diagram
        mermaid.render(id, currentStep.mermaidDiagram)
          .then(({ svg }) => {
            if (diagramContainer) {
              diagramContainer.innerHTML = svg;
            }
          })
          .catch(err => {
            console.error('Mermaid rendering error:', err);
            diagramContainer.innerHTML = '<div class="mermaid-error">Diagram rendering error</div>';
          });
      } catch (error) {
        console.error('Error processing mermaid diagram:', error);
      }
    }
  }, [activeSection, activeStep, currentStep]);
  
  const markStepAsCompleted = (stepId: string) => {
    const newCompletedSteps = new Set(completedSteps);
    newCompletedSteps.add(stepId);
    setCompletedSteps(newCompletedSteps);
  };

  const markQuizAsCompleted = (sectionId: string, score: number, total: number) => {
    const newCompletedQuizzes = new Set(completedQuizzes);
    newCompletedQuizzes.add(sectionId);
    setCompletedQuizzes(newCompletedQuizzes);
    
    setQuizScores(prev => ({
      ...prev,
      [sectionId]: { score, total }
    }));
  };
  
  const handleQuizComplete = (score: number, totalQuestions: number) => {
    if (currentSection) {
      markQuizAsCompleted(currentSection.id, score, totalQuestions);
      
      // Automatically proceed to the next section if this isn't the last one
      const currentSectionIndex = tutorial.sections.findIndex(section => section.id === activeSection);
      if (currentSectionIndex < tutorial.sections.length - 1) {
        const nextSection = tutorial.sections[currentSectionIndex + 1];
        setActiveSection(nextSection.id);
        setActiveStep(nextSection.steps[0].id);
        setShowingQuiz(false);
      }
    }
  };
  
  const goToNextStep = () => {
    if (!currentSection) return;
    
    const currentStepIndex = currentSection.steps.findIndex(step => step.id === activeStep);
    
    if (currentStepIndex < currentSection.steps.length - 1) {
      // Move to next step in current section
      markStepAsCompleted(activeStep);
      setActiveStep(currentSection.steps[currentStepIndex + 1].id);
    } else {
      // Current section is complete, check if there's a quiz
      markStepAsCompleted(activeStep);
      
      if (currentSection.quiz && currentSection.quiz.length > 0 && !completedQuizzes.has(currentSection.id)) {
        // Show the quiz for this section
        setShowingQuiz(true);
      } else {
        // No quiz, or quiz already completed - find next section
        const currentSectionIndex = tutorial.sections.findIndex(section => section.id === activeSection);
        
        if (currentSectionIndex < tutorial.sections.length - 1) {
          // Move to first step of next section
          const nextSection = tutorial.sections[currentSectionIndex + 1];
          setActiveSection(nextSection.id);
          setActiveStep(nextSection.steps[0].id);
        } else {
          // Tutorial complete
          alert('Congratulations! You have completed this tutorial.');
        }
      }
    }
  };
  
  const goToPrevStep = () => {
    if (!currentSection) return;
    
    // If showing quiz, go back to the last step in the section
    if (showingQuiz) {
      setShowingQuiz(false);
      setActiveStep(currentSection.steps[currentSection.steps.length - 1].id);
      return;
    }
    
    const currentStepIndex = currentSection.steps.findIndex(step => step.id === activeStep);
    
    if (currentStepIndex > 0) {
      // Move to previous step in current section
      setActiveStep(currentSection.steps[currentStepIndex - 1].id);
    } else {
      // Go to last step of previous section
      const currentSectionIndex = tutorial.sections.findIndex(section => section.id === activeSection);
      
      if (currentSectionIndex > 0) {
        const prevSection = tutorial.sections[currentSectionIndex - 1];
        setActiveSection(prevSection.id);
        
        // Check if the previous section has a quiz and it's not completed
        if (prevSection.quiz && prevSection.quiz.length > 0 && !completedQuizzes.has(prevSection.id)) {
          setShowingQuiz(true);
        } else {
          // Go to the last step of the previous section
          setActiveStep(prevSection.steps[prevSection.steps.length - 1].id);
        }
      }
    }
  };
  
  const calculateProgress = () => {
    // Count all steps and quizzes
    const totalSteps = tutorial.sections.reduce((acc, section) => acc + section.steps.length, 0);
    const totalQuizzes = tutorial.sections.filter(section => section.quiz && section.quiz.length > 0).length;
    const totalItems = totalSteps + totalQuizzes;
    
    // Count completed steps and quizzes
    const completedItems = completedSteps.size + completedQuizzes.size;
    
    return Math.round((completedItems / totalItems) * 100);
  };
  
  return (
    <div className="llm-tutorial">
      <div className="tutorial-header">
        <h1>{tutorial.title}</h1>
        <p className="tutorial-description">{tutorial.description}</p>
        <div className="tutorial-meta">
          <div className="tutorial-difficulty">
            <span className="label">Difficulty:</span>
            <span className={`difficulty-badge ${tutorial.difficulty}`}>{tutorial.difficulty}</span>
          </div>
          <div className="tutorial-time">
            <span className="label">Estimated Time:</span>
            <span className="time-value">{tutorial.estimatedTime}</span>
          </div>
        </div>
        <div className="tutorial-progress">
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ width: `${calculateProgress()}%` }}
            ></div>
          </div>
          <div className="progress-label">{calculateProgress()}% Complete</div>
        </div>
      </div>
      
      <div className="tutorial-content">
        <div className="tutorial-sidebar">
          <div className="toc-toggle">
            <button 
              className="view-toggle"
              onClick={() => setShowAllSteps(!showAllSteps)}
            >
              {showAllSteps ? 'Show Sections Only' : 'Show All Steps'}
            </button>
          </div>
          
          <div className="tutorial-toc">
            {tutorial.sections.map(section => (
              <div key={section.id} className="toc-section">
                <div 
                  className={`section-title ${activeSection === section.id ? 'active' : ''} 
                    ${section.steps.every(step => completedSteps.has(step.id)) && completedQuizzes.has(section.id) ? 'completed' : ''}`}
                  onClick={() => {
                    setActiveSection(section.id);
                    setActiveStep(section.steps[0].id);
                    setShowingQuiz(false);
                  }}
                >
                  {section.title}
                  {completedQuizzes.has(section.id) && (
                    <span className="section-quiz-score">
                      {quizScores[section.id]?.score}/{quizScores[section.id]?.total}
                    </span>
                  )}
                </div>
                
                {(showAllSteps || activeSection === section.id) && (
                  <div className="section-steps">
                    {section.steps.map(step => (
                      <div 
                        key={step.id}
                        className={`step-link ${activeStep === step.id && activeSection === section.id ? 'active' : ''} 
                          ${completedSteps.has(step.id) ? 'completed' : ''}`}
                        onClick={() => {
                          setActiveSection(section.id);
                          setActiveStep(step.id);
                          setShowingQuiz(false);
                        }}
                      >
                        {step.title}
                      </div>
                    ))}
                    {section.quiz && section.quiz.length > 0 && (
                      <div 
                        className={`step-link ${activeSection === section.id && showingQuiz ? 'active' : ''} 
                          ${completedQuizzes.has(section.id) ? 'completed' : ''}`}
                        onClick={() => {
                          setActiveSection(section.id);
                          setShowingQuiz(true);
                        }}
                      >
                        Quiz
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
        
        <div className="tutorial-step-content">
          {showingQuiz && currentSection && currentSection.quiz ? (
            <div className="quiz-container">
              <h2>Quiz: {currentSection.title}</h2>
              <p className="quiz-description">
                Test your understanding of {currentSection.title} with these questions.
              </p>
              <TutorialQuiz 
                questions={currentSection.quiz} 
                onComplete={handleQuizComplete}
              />
            </div>
          ) : currentStep ? (
            <div className="step-container">
              <h2 className="step-title">{currentStep.title}</h2>
              <div className="step-content">
                {currentStep.content}
              </div>
              
              {currentStep.image && (
                <div className="step-image">
                  <img src={currentStep.image} alt={currentStep.title} />
                </div>
              )}
              
              {currentStep.mermaidDiagram && (
                <div className="step-mermaid" ref={mermaidRef}></div>
              )}
              
              {currentStep.codeExample && (
                <div className="step-code">
                  <div className="code-header">
                    <h3>Code Example</h3>
                    <button 
                      className="copy-button"
                      onClick={() => {
                        navigator.clipboard.writeText(currentStep.codeExample || '');
                      }}
                    >
                      Copy
                    </button>
                  </div>
                  <pre>
                    <code>{currentStep.codeExample}</code>
                  </pre>
                </div>
              )}
              
              <div className="step-navigation">
                <button 
                  className="prev-button"
                  onClick={goToPrevStep}
                  disabled={
                    activeSection === tutorial.sections[0].id && 
                    activeStep === tutorial.sections[0].steps[0].id
                  }
                >
                  Previous
                </button>
                <button 
                  className="next-button"
                  onClick={goToNextStep}
                >
                  {activeSection === tutorial.sections[tutorial.sections.length - 1].id && 
                  activeStep === currentSection?.steps[currentSection.steps.length - 1].id &&
                  (!currentSection.quiz || completedQuizzes.has(currentSection.id))
                    ? 'Complete' 
                    : activeStep === currentSection?.steps[currentSection.steps.length - 1].id && 
                      currentSection.quiz && !completedQuizzes.has(currentSection.id)
                      ? 'Take Quiz'
                      : 'Next'}
                </button>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
};

export default LLMTutorial; 