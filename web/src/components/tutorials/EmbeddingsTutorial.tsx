import React from 'react';
import '../Tutorial.css';

const EmbeddingsTutorial: React.FC = () => {
  return (
    <div className="tutorial-container">
      <h1>Understanding Embeddings in LLMs</h1>
      
      <section className="tutorial-section">
        <h2>What Are Embeddings?</h2>
        <p>
          Embeddings are dense vector representations of discrete objects like words, tokens, or entire documents.
          In the context of LLMs, embeddings transform symbolic tokens (like words or subwords) into continuous vector spaces
          where semantic relationships can be captured mathematically.
        </p>
        <div className="key-insight">
          <h3>Key Insight</h3>
          <p>
            Embeddings map discrete tokens to continuous vector spaces where semantic similarity 
            can be measured by distance or angle between vectors.
          </p>
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>Types of Embeddings in LLMs</h2>
        
        <div className="component-explainer">
          <h3>1. Token Embeddings</h3>
          <p>Convert individual tokens (words or subwords) into vectors.</p>
          <ul>
            <li>Typically have dimensions between 512-4096 depending on model size</li>
            <li>Learned during pre-training to capture semantic meaning</li>
            <li>Similar words have similar vector representations</li>
          </ul>
        </div>
        
        <div className="component-explainer">
          <h3>2. Positional Embeddings</h3>
          <p>Encode the position of tokens in a sequence.</p>
          <ul>
            <li>Can be learned or use fixed sinusoidal functions</li>
            <li>Enable transformers to understand token order despite parallel processing</li>
            <li>Critical for capturing syntax and relationships between words</li>
          </ul>
        </div>
        
        <div className="component-explainer">
          <h3>3. Segment/Type Embeddings</h3>
          <p>Used in some models to distinguish between different parts of input.</p>
          <ul>
            <li>In BERT: Separate sentence A from sentence B</li>
            <li>In instruction-tuned models: Distinguish user queries from model responses</li>
          </ul>
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>How Embeddings Work</h2>
        
        <p>
          When input text is processed by an LLM, it first goes through tokenization, breaking it into
          discrete tokens. Each token is assigned an integer ID, which is then used to look up its 
          embedding vector from an embedding table (essentially a matrix).
        </p>
        
        <div className="code-example">
          <h3>Embedding Lookup Process:</h3>
          <pre>
{`// Tokenize input text
"Hello world" → ["Hello", "world"]

// Convert to token IDs
["Hello", "world"] → [15043, 2088]

// Look up embeddings from embedding matrix
token_id = 15043
embedding = embedding_matrix[token_id]  // Returns a vector of dimension d_model`}
          </pre>
        </div>
        
        <p>
          The resulting token embeddings are then combined with positional embeddings before
          being fed into the transformer layers:
        </p>
        <pre>
{`input_representation = token_embedding + positional_embedding`}
        </pre>
      </section>
      
      <section className="tutorial-section">
        <h2>Properties of Good Embeddings</h2>
        
        <ul>
          <li>
            <strong>Semantic similarity:</strong> Words with similar meanings have vectors that are close in the embedding space.
            For example, "dog" and "puppy" should be closer to each other than to "computer".
          </li>
          <li>
            <strong>Analogical relationships:</strong> Can capture relationships like "king" - "man" + "woman" ≈ "queen".
          </li>
          <li>
            <strong>Contextual awareness:</strong> In modern LLMs, embeddings capture the context in which words appear,
            allowing for disambiguation of words with multiple meanings.
          </li>
          <li>
            <strong>Clustering:</strong> Words from similar categories tend to cluster together in the embedding space.
          </li>
        </ul>
      </section>
      
      <section className="tutorial-section">
        <h2>Evolution of Embeddings</h2>
        
        <div className="timeline">
          <div className="timeline-item">
            <h3>Static Word Embeddings (2013-2017)</h3>
            <p>
              <strong>Examples:</strong> Word2Vec, GloVe, FastText
            </p>
            <p>
              <strong>Characteristics:</strong> Each word has exactly one embedding regardless of context.
              Good for capturing general word similarities but can't handle polysemy (words with multiple meanings).
            </p>
          </div>
          
          <div className="timeline-item">
            <h3>Contextual Embeddings (2018-Present)</h3>
            <p>
              <strong>Examples:</strong> BERT, GPT, T5 embeddings
            </p>
            <p>
              <strong>Characteristics:</strong> Words receive different embeddings based on their context.
              The word "bank" would have different representations in "river bank" versus "bank account".
              These are produced by the intermediate layers of transformer models.
            </p>
          </div>
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>Applications of Embeddings</h2>
        
        <div className="application-box">
          <h3>Semantic Search</h3>
          <p>
            Embeddings can convert queries and documents into vectors, allowing for retrieval based
            on semantic similarity rather than just keyword matching.
          </p>
        </div>
        
        <div className="application-box">
          <h3>Classification</h3>
          <p>
            Text can be embedded and then classified using the resulting vectors, which capture
            the semantic meaning of the content.
          </p>
        </div>
        
        <div className="application-box">
          <h3>Recommendation Systems</h3>
          <p>
            Items and user preferences can be embedded in the same space to find items similar
            to those a user has liked previously.
          </p>
        </div>
        
        <div className="application-box">
          <h3>Information Retrieval for LLMs</h3>
          <p>
            Embeddings enable Retrieval-Augmented Generation (RAG), where relevant information is 
            retrieved based on semantic similarity and then provided to the model.
          </p>
        </div>
      </section>
      
      <section className="tutorial-section">
        <h2>Visualizing Embeddings</h2>
        <p>
          Embeddings typically have hundreds or thousands of dimensions, making them difficult to visualize directly.
          Techniques like PCA (Principal Component Analysis) or t-SNE (t-Distributed Stochastic Neighbor Embedding)
          are used to reduce the dimensionality for visualization purposes.
        </p>
        <div className="visualization-placeholder">
          [Interactive embedding visualization would be displayed here]
        </div>
      </section>
      
      <div className="tutorial-navigation">
        <button className="tutorial-nav-button">Previous: Tokenization</button>
        <button className="tutorial-nav-button">Next: Attention Mechanisms</button>
      </div>
    </div>
  );
};

export default EmbeddingsTutorial; 