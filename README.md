# 🔬 AI Drug Discovery Copilot (Tox21 GNN + Groq AI)

An AI-powered drug discovery assistant that combines **Graph Neural Networks (GNNs)** with **LLMs (Groq)** to predict molecular toxicity, analyze drug-likeness, and generate human-understandable insights.

This project transforms a traditional ML pipeline into an **AI Copilot for Drug Discovery** by adding natural language interaction, explainability, and decision support.

---

## 🚀 Features

### 🧪 Toxicity Prediction (Tox21 GNN)
- Multi-task prediction across 12 biological endpoints
- Built using PyTorch Geometric (GINEConv)
- Outputs probability scores for toxicity risks

### 🧠 Explainable AI (XAI)
- Atom-level importance visualization
- Highlights toxic regions in molecules
- Gradient-based explainability

### 🧬 Drug-Likeness Analysis
- Molecular Weight (MW)
- LogP (lipophilicity)
- TPSA (polarity)
- QED (drug-likeness score)
- SA Score (synthetic accessibility)
- Lipinski Rule evaluation + PAINS filtering

### 🔁 Analog Generation
- SELFIES-based mutation
- Generates valid molecular candidates
- Multi-objective scoring (toxicity + QED + SA)

---

## 🤖 Groq AI Integration

### 💬 Natural Language → SMILES
- Describe molecules in plain English
- AI converts into valid SMILES
- Preview before prediction

### 🧠 Explain Button
- Converts complex outputs into simple explanations
- Explains:
  - Toxicity tables
  - Drug metrics
  - Molecular structures

### 📊 Decision Support
- Helps interpret risk levels
- Suggests insights for molecule evaluation

---

## 🧩 Tech Stack

- **Frontend:** Streamlit  
- **ML Model:** PyTorch Geometric (GNN)  
- **Cheminformatics:** RDKit  
- **Molecule Generation:** SELFIES  
- **LLM Layer:** Groq API (LLaMA models)  

---

# Clone repo
git clone https://github.com/arnavtumbde/AIDrug-2.0.git
cd AIDrug-2.0

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# If using Mac/Linux, use this instead:
# source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Create .env file with Groq API key
echo GROQ_API_KEY=your_groq_api_key_here > .env

# Run the app
streamlit run only_app.py
