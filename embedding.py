import os
import numpy as np
from openai import OpenAI

# Set OpenAI API key securely
os.environ["OPENAI_API_KEY"] ="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"  # Replace with your actual key

# Initialize OpenAI client
client = OpenAI()

def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two embedding vectors.
    
    Parameters:
    embedding1 (np.ndarray): First embedding vector.
    embedding2 (np.ndarray): Second embedding vector.
    
    Returns:
    float: Cosine similarity score (ranges from -1 to 1, where 1 means identical meaning).
    """
    dot_product = np.dot(embedding1, embedding2)
    norm_product = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    return dot_product / norm_product if norm_product != 0 else 0.0

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> np.ndarray:
    """
    Fetches the embedding for a given text using the OpenAI API.
    
    Parameters:
    text (str): Input text to generate the embedding.
    model (str): OpenAI embedding model to use (default: "text-embedding-ada-002").
    
    Returns:
    np.ndarray: The embedding vector.
    """
    try:
        response = client.embeddings.create(
            model=model,
            input=text,
            encoding_format="float"
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Error fetching embedding: {e}")
        return np.array([])

if __name__ == "__main__":
    # Example usage
    text1 = "Air is sunny"
    text2 = "President Obama"
    
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    
    if embedding1.size > 0 and embedding2.size > 0:
        similarity_score = cosine_similarity(embedding1, embedding2)
        print(f"Cosine Similarity between '{text1}' and '{text2}': {similarity_score:.4f}")
    else:
        print("Failed to retrieve embeddings.")
