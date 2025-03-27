/**
 * API client for communicating with the backend server
 */

// API base URL (change in production)
const API_BASE_URL = 'http://localhost:5000';

/**
 * Token interface matching the backend Token model
 */
export interface Token {
  text: string;
  id: number;
  type: string;
  vector?: number[];
}

/**
 * TokenStep interface matching the backend TokenStep model
 */
export interface TokenStep {
  stage: string;
  tokens: Token[];
}

/**
 * TokenizeResponse interface matching the backend TokenizeResponse model
 */
export interface TokenizeResponse {
  steps: TokenStep[];
}

/**
 * TokenizerInfo interface for tokenizer metadata
 */
export interface TokenizerInfo {
  vocabulary_size: number;
  special_tokens: string[];
  algorithm: string;
  sample_tokens: string[];
}

/**
 * Generic API request helper with error handling
 */
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  try {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const defaultHeaders = {
      'Content-Type': 'application/json',
    };
    
    const response = await fetch(url, {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    });
    
    if (!response.ok) {
      // Try to parse error message from response
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `API error: ${response.status} ${response.statusText}`);
    }
    
    // For 204 No Content responses
    if (response.status === 204) {
      return {} as T;
    }
    
    return await response.json() as T;
  } catch (error) {
    console.error('API request failed:', error);
    throw error;
  }
}

/**
 * Tokenize text and get the tokenization steps
 */
export async function tokenizeText(
  text: string,
  showVectors: boolean = false
): Promise<TokenizeResponse> {
  return apiRequest<TokenizeResponse>('/api/tokenize', {
    method: 'POST',
    body: JSON.stringify({ text, show_vectors: showVectors }),
  });
}

/**
 * Get information about the tokenizer
 */
export async function getTokenizerInfo(): Promise<TokenizerInfo> {
  return apiRequest<TokenizerInfo>('/api/tokenizer/info');
}

/**
 * Check if the API is available
 */
export async function checkApiStatus(): Promise<{ message: string; version: string }> {
  return apiRequest<{ message: string; version: string }>('/');
}

export default {
  tokenizeText,
  getTokenizerInfo,
  checkApiStatus,
}; 