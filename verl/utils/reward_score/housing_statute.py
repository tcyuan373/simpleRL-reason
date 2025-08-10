import os
from pathlib import Path
from typing import Iterable, List
import torch
import openai
from openai import OpenAI             
import torch
from sentence_transformers import SentenceTransformer, util              

openai.api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI()   


embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

def compute_embedding_score(generated_text: str, reference_text: str) -> float:
    """
    Computes the cosine similarity score between two texts using a BGE model.
    """
    # Encode the sentences into embeddings
    embedding1 = embedding_model.encode(generated_text, convert_to_tensor=True)
    embedding2 = embedding_model.encode(reference_text, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_score = util.cos_sim(embedding1, embedding2)
    
    return cosine_score.item()

def compute_llm_judge_score(generated_text: str, reference_text: str, task_description: str) -> float:
    """
    Uses an LLM to judge the quality of generated_text against a reference.
    Returns a score from 0.0 to 10.0.
    """
    # The system prompt sets the context and rules for the LLM
    system_prompt = """
    You are an impartial judge. Your task is to evaluate a generated text based on a reference text and a task description.
    Please rate the quality of the "Generated Text" on a scale from 0 to 5 based on the following criteria:
    1.  **Relevance:** How relevant is the generated text to the task?
    2.  **Correctness:** Does the generated text correctly describe the action/scene?
    3.  **Fluency:** Is the generated text grammatically correct and natural?

    Provide ONLY a single floating-point number as your score. Do not provide any explanation or additional text.
    """
    # The user prompt provides the specific data for evaluation
    user_prompt = f"""
    **Reference Text:**
    {reference_text}

    **Generated Text:**
    {generated_text}

    **Score (0-5):**
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Or "gpt-3.5-turbo" for faster/cheaper evaluation
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0, # Low temperature for consistent, deterministic output
            max_tokens=5
        )
        score_str = response.choices[0].message.content.strip()
        score = float(score_str)
        # Normalize score to be between 0.0 and 1.0 for easier weighting
        return score / 10.0
    except (ValueError, IndexError, AttributeError) as e:
        print(f"Error parsing LLM judge score: {e}. Defaulting to 0.")
        return 0.0
            

def compute_final_score(
    generated_text: str,
    reference_text: str,
    weights: dict = None) -> dict:
    """
    Calculates a final, weighted score from original, embedding, and LLM-judge scores.
    """
        
    if weights is None:
        # Default weights. Tune these for your specific application!
        weights = {
            'embedding': 0.6,
            'llm_judge': 0.4
        }

    # Ensure weights sum to 1 for a normalized final score
    assert np.isclose(sum(weights.values()), 1.0), "Weights must sum to 1."

    # 1. Get embedding score
    embedding_score = compute_embedding_score(generated_text, reference_text)

    # 2. Get LLM-judge score
    # NOTE: API calls can be slow and costly. Consider caching results.
    llm_score = compute_llm_judge_score(generated_text, reference_text)

    # 3. Calculate the weighted final score
    final_score = (
        weights['embedding'] * embedding_score +
        weights['llm_judge'] * llm_score
    )

    # Return a dictionary for detailed logging and analysis
    return final_score


if __name__ == "__main__":
    episode_data = {
    'generated_text': "Robot got red thing.",
    'reference_text': "The robot arm successfully grasps the red cube from the table.",
}

    # Define weights that prioritize semantic and expert evaluation
    custom_weights = {
        'embedding': 0.6,  # High weight on semantic correctness
        'llm_judge': 0.4   # High weight on overall quality
    }

    final_evaluation = compute_final_score(
        generated_text=episode_data['generated_text'],
        reference_text=episode_data['reference_text'],
        weights=custom_weights
    )

    print("\n--- Final Evaluation ---")
    for key, value in final_evaluation.items():
        print(f"{key:<18}: {value:.4f}")
