import os
import json
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

def analyze_analogs(smiles_list: list[str]) -> dict:
    """
    Analyzes a list of analog SMILES using Groq API and returns structured JSON output.
    Contains insights per molecule and an overall intelligence summary.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing.")

    client = Groq(api_key=api_key)

    system_prompt = (
        "You are an expert chemoinformatics AI. Your task is to analyze a list of SMILES strings representing generated analog molecules.\n"
        "You must return your analysis strictly as a JSON object, with no markdown formatting or extra text.\n\n"
        "The JSON object must have this exact structure:\n"
        "{\n"
        "  \"analogs\": [\n"
        "    {\n"
        "      \"smiles\": \"<the exact SMILES string from input>\",\n"
        "      \"status\": \"Known\" | \"Likely Known\" | \"Novel\" | \"Unknown\",\n"
        "      \"type\": \"<short structural classification>\",\n"
        "      \"use\": \"<drug-like | intermediate | unknown>\",\n"
        "      \"similar\": \"<similar known compounds or 'None'>\",\n"
        "      \"insight\": \"<1-2 line chemical reasoning>\"\n"
        "    }\n"
        "  ],\n"
        "  \"summary\": {\n"
        "    \"novel_count\": <integer>,\n"
        "    \"known_count\": <integer>,\n"
        "    \"common_pattern\": \"<most common structural pattern>\",\n"
        "    \"best_candidate_smiles\": \"<SMILES of the best candidate>\",\n"
        "    \"best_candidate_reasoning\": \"<Why this is the best candidate based on score and reasoning>\",\n"
        "    \"safety_observation\": \"<Brief observation on potential toxicity or safety>\",\n"
        "    \"recommendation\": \"<One-line recommendation>\"\n"
        "  }\n"
        "}\n\n"
        "Important constraints:\n"
        "- Do NOT hallucinate FDA claims. Use 'uncertain' if unsure.\n"
        "- Keep insights to 1-2 lines.\n"
        "- Base your analysis entirely on the provided SMILES structures.\n"
        "- Make sure to include all SMILES provided in the input list.\n"
    )

    prompt = f"Analyze the following {len(smiles_list)} SMILES strings:\n" + "\n".join(smiles_list)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        
        content = response.choices[0].message.content.strip()
        return json.loads(content)
        
    except Exception as e:
        raise RuntimeError(f"Groq API error or JSON parsing failed: {e}")
