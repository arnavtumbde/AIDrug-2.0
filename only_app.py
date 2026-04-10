import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import MolsToImage
from groq_service import generate_smiles_from_text

# Import functions and variables from our logic file
from mol_logic import (
    tox21_tasks, iupac_to_smiles, load_model, predict_smiles,
    explain_molecule, draw_png_molecule, compute_druglikeness,
    generate_selfies_analogs, score_candidate, SELFIES_AVAILABLE
)


def main():
    st.set_page_config(page_title="Tox21 GNN Explainer", layout="wide")
    st.title("🧪 Molecular Toxicity Predictor (Tox21 GNN + XAI)")

    # Initialize session state
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "analogs" not in st.session_state:
        st.session_state.analogs = None
    if "smiles" not in st.session_state:
        st.session_state.smiles = ""

    device = "cpu"
    model = load_model(device)

    # --- Sidebar/Inputs ---
    col_input, col_task = st.columns([2, 1])

    with col_input:
        input_type = st.radio("Input Type", ["SMILES", "IUPAC", "AI Prompt"], horizontal=True)
        
        if input_type == "AI Prompt":
            user_input = st.text_area("Describe the molecule in plain English", value="A simple painkiller like aspirin")
            
            if st.button("Generate SMILES"):
                if not user_input.strip():
                    st.error("Please enter a description.")
                else:
                    with st.spinner("Generating SMILES via AI Assistance..."):
                        try:
                            # Generate SMILES
                            generated_smiles = generate_smiles_from_text(user_input)
                            st.session_state.generated_smiles = generated_smiles
                            st.success(f"Generated SMILES: {generated_smiles}")
                            
                            # Preview Molecule
                            mol_preview = Chem.MolFromSmiles(generated_smiles)
                            if mol_preview:
                                st.image(MolsToImage([mol_preview]), caption="Generated Molecule Preview")
                            else:
                                st.error("Generated SMILES is invalid.")
                                st.session_state.generated_smiles = None
                        except Exception as e:
                            st.error(f"Failed to generate SMILES: {e}")
                            
            if st.session_state.get("generated_smiles"):
                st.info(f"Current generated SMILES: {st.session_state.generated_smiles}")
                st.session_state.ai_verified = st.checkbox("I confirm this molecule is correct")
            else:
                st.session_state.ai_verified = False

        else:
            user_input = st.text_input("Enter Molecule", value="CC(=O)OC1=CC=CC=C1C(=O)O")

    with col_task:
        task_name = st.selectbox("Explain endpoint", tox21_tasks)
        task_id = tox21_tasks.index(task_name)

    # --- Run Prediction ---
    if input_type == "AI Prompt":
        if not st.session_state.get("generated_smiles"):
            st.warning("Please generate a SMILES string first.")
        elif not st.session_state.get("ai_verified"):
            st.warning("Please check the verification box to enable prediction.")
            
    prediction_disabled = (input_type == "AI Prompt" and not st.session_state.get("ai_verified", False))

    if st.button("Run prediction", disabled=prediction_disabled):
        try:
            if input_type == "AI Prompt":
                smiles = st.session_state.get("generated_smiles", "")
                if not smiles:
                    raise ValueError("No SMILES generated yet.")
            else:
                smiles = iupac_to_smiles(user_input) if input_type == "IUPAC" else user_input
                
            if input_type == "IUPAC": st.info(f"Converted SMILES: {smiles}")

            mol, probs = predict_smiles(smiles, model, device)
            props = compute_druglikeness(mol)

            st.session_state.prediction = (mol, probs, props)
            st.session_state.smiles = smiles
        except Exception as e:
            st.error(f"Error: {e}")

    # --- Display Results ---
    if st.session_state.prediction:
        mol, probs, props = st.session_state.prediction

        st.subheader("Toxicity Predictions")
        st.table({
            "Endpoint": tox21_tasks,
            "Probability": [round(float(p), 3) for p in probs]
        })

        st.subheader("Drug-likeness & Developability")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Mol. Weight", f"{props['MW']:.1f}")
            st.metric("logP", f"{props['logP']:.2f}")
        with c2:
            st.metric("H-bond Donors", int(props["HBD"]))
            st.metric("QED Score", f"{props['QED']:.3f}")
        with c3:
            st.metric("Rings", int(props["Rings"]))
            if props["SA"]: st.metric("SA Score", f"{props['SA']:.2f}")

        # Explainability
        st.subheader(f"Explainability (XAI) for {task_name}")
        mol2, imp = explain_molecule(st.session_state.smiles, model, device, task_id)
        st.image(draw_png_molecule(mol2, imp), caption="Red highlights indicate toxicophore regions")

        # --- Analog Generation ---
        st.divider()
        st.subheader("🧬 AI-Suggested Analogs")
        if not SELFIES_AVAILABLE:
            st.warning("Install 'selfies' library to enable analog generation.")
        else:
            n = st.slider("Number of analogs", 0, 10, 0)
            if st.button("Generate & Score Analogs"):
                with st.spinner("Evolving molecules..."):
                    gens = generate_selfies_analogs(st.session_state.smiles, n_candidates=n)
                    scored = [score_candidate(s, model, device) for s in gens]
                    scored = [s for s in scored if s]
                    scored.sort(key=lambda x: x["score"], reverse=True)
                    st.session_state.analogs = scored

    # if st.session_state.analogs:
    #     st.subheader("Top Ranked Analog Molecules")
    #     st.table({
    #         "SMILES": [a["smiles"] for a in st.session_state.analogs],
    #         "Score": [round(a["score"], 3) for a in st.session_state.analogs],
    #         "Mean Tox": [round(a["tox"], 3) for a in st.session_state.analogs],
    #         "QED": [round(a["QED"], 3) for a in st.session_state.analogs]
    #     })
    #     mols = [a["mol"] for a in st.session_state.analogs]
    #     st.image(MolsToImage(mols, molsPerRow=5), caption="Visualizing top analogs")

        # Display analogs if they exist
        if st.session_state.analogs:
            st.subheader("Top Ranked Analog Molecules")

            # We build the data dictionary for the table
            df_data = {
                "IUPAC Name": [a["iupac"] for a in st.session_state.analogs],  # 🔥 NEW COLUMN
                "SMILES": [a["smiles"] for a in st.session_state.analogs],
                "Total Score": [round(a["score"], 3) for a in st.session_state.analogs],
                "Mean Tox": [round(a["tox"], 3) for a in st.session_state.analogs],
                "QED": [round(a["QED"], 3) for a in st.session_state.analogs]
            }

            # Displaying with st.dataframe allows for easy scrolling if the names are long
            st.dataframe(df_data, use_container_width=True)

            mols = [a["mol"] for a in st.session_state.analogs]
            st.image(MolsToImage(mols, molsPerRow=5), caption="Visualizing structure of top analogs")


if __name__ == "__main__":
    main()