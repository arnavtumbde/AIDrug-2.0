import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def generate_smiles_from_text(prompt: str) -> str:
    """
    Generate a SMILES string from a plain English description using Groq API.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing. Please set it to use this feature.")

    client = Groq(api_key=api_key)

    system_prompt = (
        "You are an expert chemoinformatics AI. Your ONLY task is to convert a natural "
        "language description of a molecule into a valid SMILES string. "
        "Output ONLY the SMILES string and nothing else. No markdown, no explanations, no wrapping quotes."
    )

    try:
        response = client.chat.completions.create(
            # Using a very reliable and fast model for short, strict output:
            model = "llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, # Deterministic setting
            max_tokens=150,
        )
        
        # Clean up the output string securely
        smiles = response.choices[0].message.content.strip()
        # Remove common markdown symbols just in case
        smiles = smiles.replace("`", "").replace("'", "").replace('"', "").strip()
        
        return smiles

    except Exception as e:
        raise RuntimeError(f"Groq API error: {e}")
