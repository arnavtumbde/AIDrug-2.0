import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem.Draw import MolDraw2DCairo
from rdkit.Chem.Draw import MolsToImage
from PIL import Image
import numpy as np
import io
import random
import subprocess

# SELFIES
try:
    import selfies as sf
    SELFIES_AVAILABLE = True
except:
    SELFIES_AVAILABLE = False

# RDKit props
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

try:
    from rdkit.Chem.QSAR import SA_Score
    def calc_sa_score(mol):
        return SA_Score.sascorer.calculateScore(mol)
except:
    def calc_sa_score(mol):
        return None

# Build PAINS catalog
pains_params = FilterCatalogParams()
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
pains_catalog = FilterCatalog(pains_params)

# -----------------------------
# Model + Featurization
# -----------------------------

tox21_tasks = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]
# -----------------------------
# OPSIN (IUPAC → SMILES via JAR)
# -----------------------------
def iupac_to_smiles(name):
    try:
        result = subprocess.run(
            ["java", "-jar", "opsin.jar"],
            input=name + "\n",
            capture_output=True,
            text=True
        )

        smiles = result.stdout.strip()

        if not smiles:
            raise ValueError("OPSIN failed to convert IUPAC name")

        return smiles

    except Exception as e:
        raise RuntimeError(f"OPSIN error: {str(e)}")

def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic()),
        atom.GetImplicitValence(),
        int(atom.GetHybridization()),
    ], dtype=torch.float)

def bond_features(bond):
    return torch.tensor([
        bond.GetBondTypeAsDouble(),
        int(bond.GetIsConjugated()),
        int(bond.GetStereo()),
        int(bond.IsInRing()),
    ], dtype=torch.float)

def mol_to_graph(mol):
    if mol is None:
        return None

    x = torch.stack([atom_features(a) for a in mol.GetAtoms()])

    edge_index, edge_attr = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        edge_index.append([i, j]); edge_attr.append(f)
        edge_index.append([j, i]); edge_attr.append(f)

    if len(edge_index) == 0:
        return None

    return Data(
        x=x,
        edge_index=torch.tensor(edge_index).t().contiguous(),
        edge_attr=torch.stack(edge_attr)
    )

class GIN_edge(nn.Module):
    def __init__(self, num_tasks=12):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(7, 64), nn.ReLU(), nn.Linear(64, 64))
        self.conv1 = GINEConv(nn1, edge_dim=4)

        nn2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
        self.conv2 = GINEConv(nn2, edge_dim=4)

        self.fc = nn.Linear(64, num_tasks)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.relu(self.conv2(x, data.edge_index, data.edge_attr))
        x = global_mean_pool(x, data.batch)
        return self.fc(x)

@st.cache_resource
def load_model(device="cpu"):
    model = GIN_edge().to(device)
    model.load_state_dict(torch.load("tox21_gnn.pt", map_location=device))
    model.eval()
    return model

def predict_smiles(smiles, model, device="cpu"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    g = mol_to_graph(mol)
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([g]).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return mol, probs

def explain_molecule(smiles, model, device, task_id):
    mol = Chem.MolFromSmiles(smiles)
    g = mol_to_graph(mol)
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([g]).to(device)
    batch.x.requires_grad_(True)

    out = model(batch)
    logit = out[0, task_id]
    model.zero_grad()
    logit.backward()

    grad = batch.x.grad.detach().cpu().abs().sum(dim=1).numpy()
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    return mol, grad

def draw_png_molecule(mol, importance):
    atom_colors = {i: (float(w), 0, 0) for i, w in enumerate(importance)}
    drawer = MolDraw2DCairo(400, 300)
    drawer.drawOptions().useBWAtomPalette()
    drawer.DrawMolecule(mol, highlightAtoms=list(atom_colors), highlightAtomColors=atom_colors)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))

# Drug-likeness
def compute_druglikeness(mol):
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    qed = QED.qed(mol)
    sa = calc_sa_score(mol)

    violations = sum([
        mw > 500, logp > 5, hbd > 5, hba > 10
    ])

    pains = [m.GetDescription() for m in pains_catalog.GetMatches(mol)]

    return {
        "MW": mw, "logP": logp, "TPSA": tpsa,
        "HBD": hbd, "HBA": hba, "Rotatable": rot,
        "Rings": rings, "QED": qed, "SA": sa,
        "Lipinski": violations, "PAINS": pains
    }

# -----------------------------
# SELFIES ANALOG GENERATION
# -----------------------------

def _init_selfies_alphabet():
    if not SELFIES_AVAILABLE: return None
    return list(sf.get_semantic_robust_alphabet())

SELFIES_ALPHABET = _init_selfies_alphabet()

def generate_selfies_analogs(seed_smiles, n_candidates=40, max_mutations=2):
    if not SELFIES_AVAILABLE:
        raise RuntimeError("SELFIES not installed")

    seed_sf = sf.encoder(seed_smiles)
    seed_tokens = list(sf.split_selfies(seed_sf))

    results = set()
    tries = 0

    while len(results) < n_candidates and tries < n_candidates * 20:
        tries += 1
        toks = seed_tokens.copy()

        for _ in range(random.randint(1, max_mutations)):
            idx = random.randrange(len(toks))
            toks[idx] = random.choice(SELFIES_ALPHABET)

        smi = sf.decoder("".join(toks))
        if not smi: continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue

        results.add(Chem.MolToSmiles(mol))

    return list(results)

def score_candidate(smiles, model, device="cpu"):
    try:
        mol, tox = predict_smiles(smiles, model, device)
    except:
        return None

    props = compute_druglikeness(mol)
    mean_tox = float(np.mean(tox))
    qed = props["QED"]

    sa = props["SA"]
    if sa is None:
        sa_term = 0.5
    else:
        sa_term = 1 - min(max((sa - 1)/9, 0), 1)

    score = (
        0.4*qed +
        0.4*(1-mean_tox) +
        0.15*sa_term +
        0.05*(1 - props["Lipinski"]/2)
    )
    if props["PAINS"]: score -= 0.5

    return {
        "smiles": smiles,
        "mol": mol,
        "score": score,
        "tox": mean_tox,
        "QED": qed,
        "Lip": props["Lipinski"],
        "PAINS": props["PAINS"],
    }

# -----------------------------
# STREAMLIT UI (FIXED)
# -----------------------------

def main():
    st.set_page_config(page_title="Tox21 GNN Explainer", layout="wide")
    st.title("🧪 Molecular Toxicity Predictor (Tox21 GNN + XAI)")

    # Init session state
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "analogs" not in st.session_state:
        st.session_state.analogs = None

    device = "cpu"
    model = load_model(device)

    col1, col2 = st.columns([2,1])

    with col1:
        input_type = st.radio("Input Type", ["SMILES", "IUPAC"], horizontal=True)
        user_input = st.text_input(
            "Enter SMILES or IUPAC name",
            value="CC(=O)OC1=CC=CC=C1C(=O)O"
        )

    with col2:
        task_name = st.selectbox("Explain endpoint", tox21_tasks)
        task_id = tox21_tasks.index(task_name)

    # --- Predict Button ---
    if st.button("Run prediction"):
        try:
            if input_type == "IUPAC":
                smiles = iupac_to_smiles(user_input)
                st.info(f"Converted SMILES: {smiles}")
            else:
                smiles = user_input

            mol, probs = predict_smiles(smiles, model, device)
            props = compute_druglikeness(mol)
            st.session_state.prediction = (mol, probs, props)
            st.session_state.smiles = smiles
        except Exception as e:
            st.error(str(e))

    # If prediction exists
    if st.session_state.prediction:
        mol, probs, props = st.session_state.prediction

        st.subheader("Toxicity Predictions")
        st.table({
            "Endpoint": tox21_tasks,
            "Probability": [round(float(p),3) for p in probs]
        })

       # --- NEW: Drug-likeness & developability metrics ---
        st.subheader("Drug-likeness & Developability")

        props = compute_druglikeness(mol)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Molecular Weight (Da)", f"{props['MW']:.1f}")
            st.metric("logP", f"{props['logP']:.2f}")
            st.metric("TPSA", f"{props['TPSA']:.1f}")
        with col2:
            st.metric("H-bond Donors", int(props["HBD"]))
            st.metric("H-bond Acceptors", int(props["HBA"]))
            st.metric("Rotatable Bonds", int(props["Rotatable"]))
        with col3:
            st.metric("Rings", int(props["Rings"]))
            st.metric("QED (0–1, higher=more drug-like)", f"{props['QED']:.3f}")
            sa_val = props["SA"]
            if sa_val is not None:
                st.metric("SA Score (1 good – 10 bad)", f"{sa_val:.2f}")
            else:
                st.write("SA Score: N/A (module not available)")

        st.write(
            f"**Lipinski Rule-of-5 violations**: {props['Lipinski']} "
            "(0 is ideal; >1 may reduce oral bioavailability)."
        )

        pains = props["PAINS"]
        if pains:
            st.warning(
                f"⚠ PAINS structural alerts detected ({len(pains)}): "
                + "; ".join(pains[:5])
                + (" ..." if len(pains) > 5 else "")
            )
        else:
            st.success("No PAINS structural alerts detected.")

        st.subheader(f"Explainability for {task_name}")
        mol2, imp = explain_molecule(st.session_state.smiles, model, device, task_id)
        st.image(draw_png_molecule(mol2, imp))

        # -----------------------------
        # ANALOG GENERATION SECTION
        # -----------------------------

        st.markdown("---")
        st.subheader("🧬 AI-Suggested Analogs")

        if not SELFIES_AVAILABLE:
            st.warning("Install selfies: pip install selfies")
        else:
            n = st.slider("How many analogs?", 10,10, 0, 1)

            if st.button("Generate AI analogs"):
                with st.spinner("Generating analogs..."):
                    gens = generate_selfies_analogs(st.session_state.smiles, n_candidates=n)
                    scored = [score_candidate(s, model, device) for s in gens]
                    scored = [s for s in scored if s]
                    scored.sort(key=lambda x: x["score"], reverse=True)
                    st.session_state.analogs = scored[:10]

    if st.session_state.analogs:
        st.subheader("Top Analog Molecules")
        table = {
            "SMILES": [a["smiles"] for a in st.session_state.analogs],
            "Score": [round(a["score"],3) for a in st.session_state.analogs],
            "Mean Tox": [round(a["tox"],3) for a in st.session_state.analogs],
            "QED": [round(a["QED"],3) for a in st.session_state.analogs],
            "Lipinski": [a["Lip"] for a in st.session_state.analogs],
            "PAINS": ["Yes" if a["PAINS"] else "No" for a in st.session_state.analogs],
        }
        st.table(table)

        mols = [a["mol"] for a in st.session_state.analogs]
        st.image(MolsToImage(mols, molsPerRow=5))

if __name__ == "__main__":
    main()
