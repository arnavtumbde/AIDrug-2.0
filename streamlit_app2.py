import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D, MolDraw2DCairo
from rdkit.Chem.Draw import MolsToImage
from PIL import Image
import numpy as np
import io



# --- NEW: RDKit property imports ---
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# Try to import SA_Score if available
try:
    from rdkit.Chem.QSAR import SA_Score
    def calc_sa_score(mol):
        return SA_Score.sascorer.calculateScore(mol)
except Exception:
    def calc_sa_score(mol):
        return None  # gracefully fall back if not available

# Build PAINS catalog once (NEW)
pains_params = FilterCatalogParams()
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
pains_catalog = FilterCatalog(pains_params)

# -----------------------------
# 1. Model + featurization
# -----------------------------

tox21_tasks = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

def atom_features(atom):
    """Must match what you used during training (7 dims)."""
    return torch.tensor([
        atom.GetAtomicNum(),               # 1
        atom.GetTotalDegree(),             # 2
        atom.GetFormalCharge(),            # 3
        atom.GetTotalNumHs(),              # 4
        int(atom.GetIsAromatic()),         # 5
        atom.GetImplicitValence(),         # 6
        int(atom.GetHybridization()),      # 7
    ], dtype=torch.float)

def bond_features(bond):
    """4-dim edge features (must match training)."""
    return torch.tensor([
        bond.GetBondTypeAsDouble(),        # 1
        int(bond.GetIsConjugated()),       # 2
        int(bond.GetStereo()),             # 3
        int(bond.IsInRing()),              # 4
    ], dtype=torch.float)

def mol_to_graph(mol):
    if mol is None:
        return None

    x = torch.stack([atom_features(a) for a in mol.GetAtoms()])

    edge_index = []
    edge_attr = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        f = bond_features(b)

        edge_index.append([i, j])
        edge_attr.append(f)

        edge_index.append([j, i])
        edge_attr.append(f)

    if len(edge_index) == 0:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class GIN_edge(torch.nn.Module):
    def __init__(self, num_tasks=12):
        super().__init__()

        nn1 = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.conv1 = GINEConv(nn1, edge_dim=4)

        nn2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.conv2 = GINEConv(nn2, edge_dim=4)

        self.fc = nn.Linear(64, num_tasks)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
        return self.fc(x)


@st.cache_resource
def load_model(device="cpu"):
    model = GIN_edge().to(device)
    state_dict = torch.load("tox21_gnn.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_smiles(smiles, model, device="cpu"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    g = mol_to_graph(mol)
    if g is None:
        raise ValueError("Molecule has no bonds / graph could not be built.")

    # Create a batch of size 1
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([g]).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return mol, probs


def explain_molecule(smiles, model, device="cpu", task_id=0):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    g = mol_to_graph(mol)
    if g is None:
        raise ValueError("Molecule has no bonds / graph could not be built.")

    from torch_geometric.data import Batch
    batch = Batch.from_data_list([g]).to(device)

    # We need gradients w.r.t. node features
    batch.x.requires_grad_(True)

    out = model(batch)           # shape [1, 12]
    logit = out[0, task_id]      # scalar for this task
    model.zero_grad()
    logit.backward()

    # importance per atom = gradient magnitude
    grad = batch.x.grad.detach().cpu()
    importance = grad.abs().sum(dim=1).numpy()    # [num_atoms]

    # normalize 0-1
    importance = (importance - importance.min()) / (importance.ptp() + 1e-8)
    return mol, importance


def draw_png_molecule(mol, importance):
    """Return a PNG image with atoms colored by importance."""
    atom_colors = {
        i: (float(w), 0.0, 0.0)   # Red intensity
        for i, w in enumerate(importance)
    }

    drawer = MolDraw2DCairo(400, 300)
    drawer.drawOptions().useBWAtomPalette()
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors
    )
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()

    return Image.open(io.BytesIO(png_bytes))


# --- NEW: compute drug-likeness & developability metrics ---
def compute_druglikeness(mol):
    """Compute basic drug-likeness / developability properties."""
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    qed_score = float(QED.qed(mol))

    sa = calc_sa_score(mol)

    # Lipinski rule-of-5 violations
    lipinski_violations = 0
    if mw > 500:
        lipinski_violations += 1
    if logp > 5:
        lipinski_violations += 1
    if hbd > 5:
        lipinski_violations += 1
    if hba > 10:
        lipinski_violations += 1

    # PAINS alerts
    pains_matches = pains_catalog.GetMatches(mol)
    pains_alerts = [m.GetDescription() for m in pains_matches]

    return {
        "MW": mw,
        "logP": logp,
        "TPSA": tpsa,
        "HBD": hbd,
        "HBA": hba,
        "RotatableBonds": rot_bonds,
        "Rings": rings,
        "QED": qed_score,
        "SA": sa,
        "LipinskiViolations": lipinski_violations,
        "PAINSAlerts": pains_alerts,
    }

# -----------------------------
# 2. Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="Tox21 GNN Explainer", layout="wide")
    st.title("🧪 Molecular Toxicity Predictor (Tox21 GNN + XAI)")

    st.markdown(
        """
        Enter a SMILES string to predict **12 Tox21 toxicity endpoints** and see
        which atoms contribute most (red = higher contribution).
        """
    )

    device = "cpu"  # Vercel / CPU-only
    model = load_model(device=device)

    col_input, col_options = st.columns([2, 1])

    with col_input:
        smiles = st.text_input(
            "SMILES",
            value="CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            help="Example: CC(=O)OC1=CC=CC=C1C(=O)O",
        )

    with col_options:
        explain_task_name = st.selectbox(
            "Endpoint to explain (for atom heatmap):",
            tox21_tasks,
            index=0,
        )
        task_id = tox21_tasks.index(explain_task_name)

    if st.button("Run prediction"):
        if not smiles.strip():
            st.error("Please enter a SMILES string.")
            return

        try:
            mol, probs = predict_smiles(smiles, model, device=device)
        except Exception as e:
            st.error(f"Error: {e}")
            return

        # --- Toxicity predictions ---
        st.subheader("Tox21 Toxicity Predictions")
        st.write(
            "Probability that the molecule is **active** in each assay "
            "(1 = toxic / active, 0 = inactive)."
        )

        pred_table = {
            "Endpoint": tox21_tasks,
            "Probability": [float(f"{p:.3f}") for p in probs],
        }
        st.table(pred_table)

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
            st.metric("Rotatable Bonds", int(props["RotatableBonds"]))
        with col3:
            st.metric("Rings", int(props["Rings"]))
            st.metric("QED (0–1, higher=more drug-like)", f"{props['QED']:.3f}")
            sa_val = props["SA"]
            if sa_val is not None:
                st.metric("SA Score (1 good – 10 bad)", f"{sa_val:.2f}")
            else:
                st.write("SA Score: N/A (module not available)")

        st.write(
            f"**Lipinski Rule-of-5 violations**: {props['LipinskiViolations']} "
            "(0 is ideal; >1 may reduce oral bioavailability)."
        )

        pains = props["PAINSAlerts"]
        if pains:
            st.warning(
                f"⚠ PAINS structural alerts detected ({len(pains)}): "
                + "; ".join(pains[:5])
                + (" ..." if len(pains) > 5 else "")
            )
        else:
            st.success("No PAINS structural alerts detected.")

        # --- XAI heatmap ---
        st.subheader(f"Atom-level explanation for: **{explain_task_name}**")
        try:
            mol_ex, importance = explain_molecule(
                smiles, model, device=device, task_id=task_id
            )
            img = draw_png_molecule(mol_ex, importance)
            st.image(img, caption="Atom importance heatmap")

        except Exception as e:
            st.warning(f"Could not generate explanation: {e}")

if __name__ == "__main__":
    main()
