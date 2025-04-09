import React from 'react';
import '../Tutorial.css';

const InferenceTutorial: React.FC = () => {
  return (
    <div className="tutorial-container">
      <h1>LLM Inference: How Models Generate Text</h1>
      
      <section className="tutorial-section">
        <h2>Understanding LLM Inference</h2>
        <p>
          Inference is the process where a language model generates text based on input. While training focuses on 
          learning patterns from data, inference is about using the trained model to produce new content.
        </p>
        <div className="key-insight">
          <h3>Key Insight</h3>
          <p>
            LLM inference is fundamentally autoregressive: the model predicts one token at a time,
            with each new token depending on all previously generated tokens.
          </p>
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>The Inference Pipeline</h2>
        
        <div className="pipeline-stage">
          <h3>1. Input Processing</h3>
          <ul>
            <li>
              <strong>Tokenization:</strong> Convert input text into tokens using the model's tokenizer
            </li>
            <li>
              <strong>Token IDs:</strong> Map tokens to their corresponding IDs in the vocabulary
            </li>
            <li>
              <strong>Prompt formatting:</strong> Add any necessary special tokens, instruction templates, etc.
            </li>
          </ul>
        </div>
        
        <div className="pipeline-stage">
          <h3>2. Forward Pass</h3>
          <ul>
            <li>
              <strong>Embeddings:</strong> Convert token IDs to embeddings
            </li>
            <li>
              <strong>Layer processing:</strong> Pass embeddings through all transformer layers
            </li>
            <li>
              <strong>Logits generation:</strong> The model outputs raw scores (logits) for the next token
            </li>
          </ul>
        </div>
        
        <div className="pipeline-stage">
          <h3>3. Token Selection</h3>
          <ul>
            <li>
              <strong>Softmax application:</strong> Convert logits to probabilities
            </li>
            <li>
              <strong>Sampling or selection:</strong> Choose the next token based on a sampling strategy
            </li>
          </ul>
        </div>
        
        <div className="pipeline-stage">
          <h3>4. Text Generation Loop</h3>
          <ul>
            <li>
              <strong>Append token:</strong> Add the newly generated token to the sequence
            </li>
            <li>
              <strong>Repeat:</strong> Continue steps 2-4 until a stop condition is met
            </li>
          </ul>
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>Sampling Strategies</h2>
        
        <div className="strategy-box">
          <h3>Greedy Decoding</h3>
          <p>
            Simply select the token with the highest probability at each step.
          </p>
          <div className="code-example">
            <pre>
{`next_token_id = argmax(logits)`}
            </pre>
          </div>
          <div className="pros-cons">
            <div className="pros">
              <h4>Pros</h4>
              <ul>
                <li>Deterministic - same output every time</li>
                <li>Simple to implement</li>
                <li>Often works well for short, factual responses</li>
              </ul>
            </div>
            <div className="cons">
              <h4>Cons</h4>
              <ul>
                <li>Can get stuck in repetition loops</li>
                <li>Lacks creativity and diversity</li>
                <li>Can't recover from mistakes</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div className="strategy-box">
          <h3>Temperature Sampling</h3>
          <p>
            Sample from the probability distribution after applying temperature to control randomness.
          </p>
          <div className="code-example">
            <pre>
{`# Apply temperature to logits
logits_with_temp = logits / temperature
# Convert to probabilities
probs = softmax(logits_with_temp)
# Sample from distribution
next_token_id = random_sample(probs)`}
            </pre>
          </div>
          <div className="pros-cons">
            <div className="pros">
              <h4>Pros</h4>
              <ul>
                <li>Controllable randomness through temperature parameter</li>
                <li>More diverse and creative outputs</li>
                <li>Can explore different possible continuations</li>
              </ul>
            </div>
            <div className="cons">
              <h4>Cons</h4>
              <ul>
                <li>Non-deterministic - different output each run</li>
                <li>Can generate incoherent text at high temperatures</li>
                <li>May hallucinate or generate factual errors</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div className="strategy-box">
          <h3>Top-K Sampling</h3>
          <p>
            Restrict sampling to the K tokens with the highest probabilities.
          </p>
          <div className="code-example">
            <pre>
{`# Get top K tokens by probability
top_k_logits = keep_top_k(logits, k=50)
# Zero out all other logits
masked_logits = mask_except_top_k(logits, top_k_logits)
# Convert to probabilities
probs = softmax(masked_logits)
# Sample from distribution
next_token_id = random_sample(probs)`}
            </pre>
          </div>
          <div className="pros-cons">
            <div className="pros">
              <h4>Pros</h4>
              <ul>
                <li>Prevents sampling from low-probability tokens</li>
                <li>Balance of randomness and control</li>
              </ul>
            </div>
            <div className="cons">
              <h4>Cons</h4>
              <ul>
                <li>Fixed K may be too restrictive or too permissive depending on context</li>
                <li>Doesn't adapt to the shape of the probability distribution</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div className="strategy-box">
          <h3>Nucleus (Top-p) Sampling</h3>
          <p>
            Sample from the smallest set of tokens whose cumulative probability exceeds p.
          </p>
          <div className="code-example">
            <pre>
{`# Sort tokens by probability
sorted_logits = sort_by_prob(logits)
# Find minimal set where cumulative prob > p
nucleus = get_tokens_with_cumulative_prob(sorted_logits, p=0.9)
# Zero out all tokens outside the nucleus
masked_logits = mask_except_nucleus(logits, nucleus)
# Convert to probabilities
probs = softmax(masked_logits)
# Sample from distribution
next_token_id = random_sample(probs)`}
            </pre>
          </div>
          <div className="pros-cons">
            <div className="pros">
              <h4>Pros</h4>
              <ul>
                <li>Adapts to the confidence of the model in each context</li>
                <li>Good balance of diversity and quality</li>
                <li>Works well across different types of text</li>
              </ul>
            </div>
            <div className="cons">
              <h4>Cons</h4>
              <ul>
                <li>More complex to implement</li>
                <li>Requires careful tuning of p parameter</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div className="strategy-box">
          <h3>Beam Search</h3>
          <p>
            Maintain multiple candidate sequences (beams) and choose the highest probability sequence overall.
          </p>
          <div className="code-example">
            <pre>
{`# Initialize with top B (beam width) tokens for first position
beams = get_top_tokens(logits, beam_width=5)
# For each subsequent position
for position in range(1, max_length):
    # For each beam, get next token probabilities
    candidates = []
    for beam in beams:
        next_logits = model.forward(beam)
        # Add all possible continuations with their scores
        for token, score in zip(range(vocab_size), next_logits):
            candidates.append((beam + [token], beam_score * score))
    # Keep only the top B candidates
    beams = get_top_beams(candidates, beam_width=5)
# Return the highest scoring complete sequence
return beams[0]`}
            </pre>
          </div>
          <div className="pros-cons">
            <div className="pros">
              <h4>Pros</h4>
              <ul>
                <li>Explores multiple promising paths simultaneously</li>
                <li>Often produces more coherent text than simple sampling</li>
                <li>Good for tasks with a "correct" answer (e.g., translation)</li>
              </ul>
            </div>
            <div className="cons">
              <h4>Cons</h4>
              <ul>
                <li>Computationally expensive</li>
                <li>Can produce generic, high-probability but boring text</li>
                <li>Tends to prefer shorter sequences due to probability multiplication</li>
              </ul>
            </div>
          </div>
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>Advanced Inference Techniques</h2>
        
        <div className="technique-box">
          <h3>Contrastive Search</h3>
          <p>
            Balances likelihood with diversity by selecting tokens that are probable but different from previous context.
          </p>
        </div>
        
        <div className="technique-box">
          <h3>Speculative Sampling</h3>
          <p>
            Uses a smaller "draft" model to suggest multiple tokens which are then verified by the main model, speeding up generation.
          </p>
        </div>
        
        <div className="technique-box">
          <h3>Classifier-Free Guidance</h3>
          <p>
            Interpolates between conditional and unconditional generation to improve adherence to the prompt.
          </p>
        </div>
        
        <div className="technique-box">
          <h3>Self-Consistency</h3>
          <p>
            Generates multiple candidate responses and selects the most consistent one, often used for reasoning tasks.
          </p>
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>Prompt Engineering for Inference</h2>
        <p>
          How you format the input prompt can dramatically affect inference quality:
        </p>
        
        <div className="prompt-technique">
          <h3>Few-Shot Prompting</h3>
          <p>
            Include examples of the desired output format and style in the prompt to guide the model.
          </p>
          <div className="example">
            <pre>
{`Translate English to French:
English: The weather is nice today.
French: Le temps est beau aujourd'hui.
English: I love programming.
French: J'aime la programmation.
English: What time is the meeting?
French:`}
            </pre>
          </div>
        </div>
        
        <div className="prompt-technique">
          <h3>Chain-of-Thought Prompting</h3>
          <p>
            Encourage the model to show its reasoning step by step before giving a final answer.
          </p>
          <div className="example">
            <pre>
{`Question: If I have 3 apples and buy 2 more, then give 1 to my friend, how many apples do I have?
Let's think through this step by step:
1. Initially, I have 3 apples.
2. After buying 2 more, I have 3 + 2 = 5 apples.
3. After giving 1 to my friend, I have 5 - 1 = 4 apples.
Therefore, I have 4 apples.`}
            </pre>
          </div>
        </div>
        
        <div className="prompt-technique">
          <h3>Instruction Tuning Format</h3>
          <p>
            Use specific formats that match how the model was instruction-tuned.
          </p>
          <div className="example">
            <pre>
{`System: You are a helpful, harmless, and honest assistant.
User: Explain quantum computing in simple terms.
Assistant:`}
            </pre>
          </div>
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>Inference Optimization Techniques</h2>
        
        <div className="optimization-box">
          <h3>Quantization</h3>
          <p>
            Reduce precision of model weights (e.g., FP16, INT8, INT4) to decrease memory usage and increase inference speed.
          </p>
        </div>
        
        <div className="optimization-box">
          <h3>KV Caching</h3>
          <p>
            Store and reuse key-value pairs from previous tokens to avoid redundant computation.
          </p>
        </div>
        
        <div className="optimization-box">
          <h3>Batch Processing</h3>
          <p>
            Process multiple requests simultaneously to maximize hardware utilization.
          </p>
        </div>
        
        <div className="optimization-box">
          <h3>Continuous Batching</h3>
          <p>
            Dynamically manage batches to handle requests of varying lengths efficiently.
          </p>
        </div>
        
        <div className="optimization-box">
          <h3>Early Exit</h3>
          <p>
            Skip later layers for tokens where early layer outputs already provide high confidence predictions.
          </p>
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>Inference Visualization</h2>
        <p>
          Visualizing the inference process can provide insights into how the model makes decisions:
        </p>
        <div className="visualization-placeholder">
          [Interactive visualization of token probabilities during inference would be displayed here]
        </div>
        <p>
          In the visualization section, you can see the probability distribution for each token as it's generated,
          and how different sampling strategies would affect the output.
        </p>
      </section>
      
      <div className="tutorial-navigation">
        <button className="tutorial-nav-button">Previous: Softmax Function</button>
        <button className="tutorial-nav-button">Next: Fine-tuning Techniques</button>
      </div>
    </div>
  );
};

export default InferenceTutorial; 