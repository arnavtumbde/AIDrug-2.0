
---

# 🔬 AI Drug Discovery Copilot

### *Tox21 GNN + Groq AI Powered Assistant*

An **AI-powered drug discovery copilot** that combines **Graph Neural Networks (GNNs)** with **Large Language Models (Groq)** to:

* Predict molecular toxicity
* Analyze drug-likeness
* Generate new molecular candidates
* Provide human-readable insights

This project upgrades a traditional ML pipeline into an **interactive AI assistant** with explainability and decision support.

---

## 🚀 Key Features

### 🧪 Toxicity Prediction (Tox21 GNN)

* Multi-task prediction across **12 biological endpoints**
* Built using **PyTorch Geometric (GINEConv)**
* Outputs **probability scores for toxicity risks**

---

### 🧠 Explainable AI (XAI)

* Atom-level importance visualization
* Highlights **toxic regions in molecules**
* Uses **gradient-based explainability**

---

### 🧬 Drug-Likeness Analysis

Evaluate molecules using key pharmaceutical metrics:

* Molecular Weight (MW)
* LogP (Lipophilicity)
* TPSA (Polarity)
* QED (Drug-likeness score)
* SA Score (Synthetic Accessibility)

✔ Lipinski Rule Evaluation
✔ PAINS Filtering

---

### 🔁 Analog Generation

* SELFIES-based mutation
* Generates **valid molecular candidates**
* Multi-objective optimization:

  * Toxicity ↓
  * QED ↑
  * SA Score ↑

---

## 🤖 Groq AI Integration

### 💬 Natural Language → SMILES

* Describe molecules in **plain English**
* AI converts them into **valid SMILES**
* Preview before running predictions

---

### 🧠 Explain Button

Simplifies complex outputs into human-friendly insights:

* Toxicity tables
* Drug-likeness metrics
* Molecular structures

---

### 📊 Decision Support

* Interprets **risk levels**
* Suggests actionable insights
* Helps in **molecule evaluation & selection**

---

## 🧩 Tech Stack

| Layer                  | Technology              |
| ---------------------- | ----------------------- |
| 🎨 Frontend            | Streamlit               |
| 🧠 ML Model            | PyTorch Geometric (GNN) |
| ⚗️ Cheminformatics     | RDKit                   |
| 🧬 Molecule Generation | SELFIES                 |
| 🤖 LLM Layer           | Groq API (LLaMA Models) |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/arnavtumbde/AIDrug-2.0.git
cd AIDrug-2.0
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

---

### 3️⃣ Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

### 4️⃣ Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

### 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 6️⃣ Set Up Environment Variables

Create a `.env` file:

```bash
echo GROQ_API_KEY=your_groq_api_key_here > .env
```

---

### 7️⃣ Run the Application

```bash
streamlit run only_app.py
```

---

## 🎯 Use Cases

* Early-stage **drug discovery screening**
* Toxicity risk assessment
* Molecule optimization
* AI-assisted medicinal chemistry

---

## 🌟 Future Improvements

* Docking simulation integration
* Protein-ligand interaction modeling
* Advanced multi-objective optimization
* Dataset expansion beyond Tox21

---

## 🤝 Contributing

Contributions are welcome. Feel free to fork the repo and submit pull requests.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 💡 Author

**Arnav Tumbde, Anshu Bagne, Aniruddha Moharir** 
AI + Drug Discovery Enthusiast

---

