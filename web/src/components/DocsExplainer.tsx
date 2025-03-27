import React from 'react';
import './DocsExplainer.css';

interface DocsExplainerProps {
  term: string;
}

const DocsExplainer: React.FC<DocsExplainerProps> = ({ term }) => {
  const explanations: Record<string, { title: string, content: React.ReactNode }> = {
    'masking': {
      title: 'Masking in Transformers',
      content: (
        <>
          <p>
            <strong>Masking</strong> is a technique used in transformer models to control which tokens can attend to which other tokens. There are two main types of masking:
          </p>
          <h4>Padding Mask</h4>
          <p>
            Used to ignore padding tokens in variable-length sequences. This ensures the model doesn't pay attention to padding tokens, which don't contain actual information.
          </p>
          <h4>Causal Mask (Future Mask)</h4>
          <p>
            Used in autoregressive generation to prevent tokens from attending to future tokens. This is crucial for decoder models like GPT, which generate text one token at a time.
          </p>
          <p>
            For example, when predicting the 3rd word in a sequence, the model can only look at the 1st and 2nd words, not the 4th or 5th.
          </p>
          <h4>Implementation</h4>
          <p>
            Masking is implemented by adding large negative values (like -10000) to positions that should be masked before the softmax operation. This effectively sets the attention probability to zero for those positions.
          </p>
        </>
      )
    },
    'softmax': {
      title: 'Softmax Function',
      content: (
        <>
          <p>
            <strong>Softmax</strong> is a mathematical function that converts a vector of real numbers into a probability distribution. In language models, it's used in two critical places:
          </p>
          <h4>In Attention Mechanism</h4>
          <p>
            After computing attention scores (dot product of queries and keys), softmax normalizes these scores to create attention weights that sum to 1. This determines how much each value contributes to the output.
          </p>
          <h4>In Output Layer</h4>
          <p>
            The final layer of a language model produces logits for each token in the vocabulary. Softmax converts these logits into probabilities, allowing the model to predict the most likely next token.
          </p>
          <h4>Mathematical Definition</h4>
          <p>
            For a vector z, the softmax function is defined as:
          </p>
          <div className="math-formula">
            softmax(z)<sub>i</sub> = e<sup>z<sub>i</sub></sup> / Σ<sub>j</sub> e<sup>z<sub>j</sub></sup>
          </div>
          <p>
            This formula ensures all outputs are between 0 and 1 and sum to 1, creating a valid probability distribution.
          </p>
          <h4>Numerical Stability</h4>
          <p>
            In practice, softmax is often implemented with a "shift" to prevent numerical overflow:
          </p>
          <div className="math-formula">
            softmax(z)<sub>i</sub> = e<sup>(z<sub>i</sub> - max(z))</sup> / Σ<sub>j</sub> e<sup>(z<sub>j</sub> - max(z))</sup>
          </div>
        </>
      )
    }
  };
  
  const explanation = explanations[term.toLowerCase()];
  
  if (!explanation) {
    return (
      <div className="docs-explainer not-found">
        <p>No explanation found for term: <strong>{term}</strong></p>
      </div>
    );
  }
  
  return (
    <div className="docs-explainer">
      <h3 className="explainer-title">{explanation.title}</h3>
      <div className="explainer-content">
        {explanation.content}
      </div>
    </div>
  );
};

export default DocsExplainer; 