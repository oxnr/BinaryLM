import React from 'react';
import '../Tutorial.css';

const SoftmaxTutorial: React.FC = () => {
  return (
    <div className="tutorial-container">
      <h1>Understanding the Softmax Function in LLMs</h1>
      
      <section className="tutorial-section">
        <h2>What is Softmax?</h2>
        <p>
          The softmax function is a critical component in language models that converts a vector of raw scores (logits) 
          into a probability distribution. It's essential for both the attention mechanism and the final output layer of LLMs.
        </p>
        <div className="key-insight">
          <h3>Key Insight</h3>
          <p>
            Softmax ensures that all output values are between 0 and 1 and sum to exactly 1, creating a valid probability distribution
            that can be sampled from during text generation.
          </p>
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>The Mathematics of Softmax</h2>
        
        <div className="formula-box">
          <h3>Softmax Formula</h3>
          <p>
            For a vector <strong>z</strong> with <em>n</em> elements, the softmax function is defined as:
          </p>
          <div className="formula">
            softmax(z<sub>i</sub>) = e<sup>z<sub>i</sub></sup> / Σ<sub>j=1</sub><sup>n</sup> e<sup>z<sub>j</sub></sup>
          </div>
          <p>
            Where:
          </p>
          <ul>
            <li>z<sub>i</sub> is the input value (logit) for class i</li>
            <li>e<sup>z<sub>i</sub></sup> is the exponential function applied to z<sub>i</sub></li>
            <li>Σ<sub>j=1</sub><sup>n</sup> e<sup>z<sub>j</sub></sup> is the sum of exponentials for all elements in the vector</li>
          </ul>
        </div>
        
        <div className="example-box">
          <h3>Simple Example</h3>
          <p>
            Let's say we have logits [2.0, 1.0, 0.1] for three tokens:
          </p>
          <ol>
            <li>Calculate exponentials: [e<sup>2.0</sup>, e<sup>1.0</sup>, e<sup>0.1</sup>] = [7.39, 2.72, 1.11]</li>
            <li>Sum of exponentials: 7.39 + 2.72 + 1.11 = 11.22</li>
            <li>Softmax: [7.39/11.22, 2.72/11.22, 1.11/11.22] = [0.66, 0.24, 0.10]</li>
          </ol>
          <p>
            The resulting probability distribution [0.66, 0.24, 0.10] sums to 1, indicating that the first token
            has a 66% probability, the second token 24%, and the third token 10%.
          </p>
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>Properties of Softmax</h2>
        
        <div className="property-box">
          <h3>Normalization</h3>
          <p>
            Softmax outputs sum to 1, creating a proper probability distribution regardless of the input values' magnitude.
          </p>
        </div>
        
        <div className="property-box">
          <h3>Relative Scale Preservation</h3>
          <p>
            Larger input values result in higher probabilities, preserving the relative ordering of the raw scores.
          </p>
        </div>
        
        <div className="property-box">
          <h3>Non-linearity</h3>
          <p>
            The exponential function makes softmax highly non-linear, amplifying differences between scores.
          </p>
        </div>
        
        <div className="property-box">
          <h3>Translation Invariance</h3>
          <p>
            Adding a constant to all logits doesn't change the resulting probabilities, which helps with numerical stability.
          </p>
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>Softmax in Attention Mechanisms</h2>
        <p>
          In the self-attention mechanism, softmax is applied to the scaled dot products of queries and keys:
        </p>
        <div className="code-example">
          <pre>
{`Attention(Q, K, V) = softmax(QK^T / √d_k)V`}
          </pre>
        </div>
        <p>
          This converts raw attention scores into attention weights (a probability distribution) that determine how much focus 
          to place on different tokens when creating the contextual representation of a token.
        </p>
        <div className="example-visualization">
          <div className="attention-example">
            [Visualization of attention weights after softmax would be shown here]
          </div>
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>Softmax in the Output Layer</h2>
        <p>
          At the final layer of an LLM, softmax is applied to the logits to produce a probability distribution over the vocabulary:
        </p>
        <ol>
          <li>The model produces raw scores (logits) for each token in the vocabulary</li>
          <li>Softmax converts these logits into a probability distribution</li>
          <li>This distribution can be used to:
            <ul>
              <li>Select the most likely next token (greedy decoding)</li>
              <li>Sample from the distribution (temperature sampling)</li>
              <li>Use more complex sampling strategies (nucleus sampling, beam search, etc.)</li>
            </ul>
          </li>
        </ol>
      </section>
      
      <section className="tutorial-section">
        <h2>Softmax and Temperature</h2>
        <p>
          When generating text, the "temperature" parameter modifies how the softmax function distributes probability:
        </p>
        <div className="formula-box">
          <h3>Softmax with Temperature</h3>
          <div className="formula">
            softmax(z<sub>i</sub>/T) = e<sup>z<sub>i</sub>/T</sup> / Σ<sub>j=1</sub><sup>n</sup> e<sup>z<sub>j</sub>/T</sup>
          </div>
          <p>Where T is the temperature parameter.</p>
        </div>
        
        <div className="property-box">
          <h3>Effects of Temperature</h3>
          <ul>
            <li><strong>Low temperature (T &lt; 1):</strong> Makes distribution more "peaky" - highest probability tokens become even more likely</li>
            <li><strong>High temperature (T &gt; 1):</strong> Makes distribution more uniform - gives more weight to lower-probability tokens</li>
            <li><strong>T = 1:</strong> Standard softmax without modification</li>
            <li><strong>T → 0:</strong> Approaches argmax (deterministic selection of highest-scoring token)</li>
            <li><strong>T → ∞:</strong> Approaches uniform distribution (random selection)</li>
          </ul>
        </div>
        
        <div className="interactive-element">
          [Interactive visualization of temperature effects would be displayed here]
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>Implementation Considerations</h2>
        
        <div className="consideration-box">
          <h3>Numerical Stability</h3>
          <p>
            Computing softmax directly can lead to numerical overflow due to large exponentials. The standard solution is to 
            subtract the maximum value from all elements before applying the exponential:
          </p>
          <div className="code-example">
            <pre>
{`def stable_softmax(x):
    # Subtract max for numerical stability
    shifted_x = x - x.max(axis=-1, keepdims=True)
    # Calculate softmax with shifted values
    exp_x = np.exp(shifted_x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)`}
            </pre>
          </div>
        </div>
        
        <div className="consideration-box">
          <h3>Log Softmax</h3>
          <p>
            For better numerical stability when computing cross-entropy loss, log_softmax is often used:
          </p>
          <div className="formula">
            log_softmax(z<sub>i</sub>) = z<sub>i</sub> - log(Σ<sub>j=1</sub><sup>n</sup> e<sup>z<sub>j</sub></sup>)
          </div>
          <p>
            This directly computes the logarithm of the softmax, avoiding potential numerical issues.
          </p>
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>Alternatives to Softmax</h2>
        <p>
          While softmax is the standard approach for probability distributions in LLMs, there are alternatives:
        </p>
        <ul>
          <li>
            <strong>Sparsemax:</strong> Creates sparse probability distributions where some elements receive exactly zero probability.
          </li>
          <li>
            <strong>Gumbel-Softmax:</strong> Allows for sampling discrete categorical variables in a differentiable way.
          </li>
          <li>
            <strong>Spherical Softmax:</strong> Projects probabilities onto a hypersphere rather than the probability simplex.
          </li>
        </ul>
        <p>
          However, standard softmax remains the most widely used approach in current LLM architectures.
        </p>
      </section>
      
      <div className="tutorial-navigation">
        <button className="tutorial-nav-button">Previous: Attention Mechanisms</button>
        <button className="tutorial-nav-button">Next: Inference Techniques</button>
      </div>
    </div>
  );
};

export default SoftmaxTutorial; 