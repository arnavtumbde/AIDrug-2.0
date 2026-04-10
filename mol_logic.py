import urllib

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem.Draw import MolDraw2DCairo
from PIL import Image
import numpy as np
import io
import random
import subprocess
import streamlit as st  # Needed for @st.cache_resource

# RDKit props
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# SELFIES
try:
    import selfies as sf
    SELFIES_AVAILABLE = True
except ImportError:
    SELFIES_AVAILABLE = False

# Try loading SA_Score
try:
    from rdkit.Chem.QSAR import SA_Score
    def calc_sa_score(mol):
        return SA_Score.sascorer.calculateScore(mol)
except ImportError:
    def calc_sa_score(mol):
        return None

# --- Constants & Global Objects ---
tox21_tasks = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

pains_params = FilterCatalogParams()
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
pains_catalog = FilterCatalog(pains_params)

# --- IUPAC Logic ---
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

# --- Model Architecture ---
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

# --- Featurization ---
def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(), atom.GetTotalDegree(), atom.GetFormalCharge(),
        atom.GetTotalNumHs(), int(atom.GetIsAromatic()),
        atom.GetImplicitValence(), int(atom.GetHybridization()),
    ], dtype=torch.float)

def bond_features(bond):
    return torch.tensor([
        bond.GetBondTypeAsDouble(), int(bond.GetIsConjugated()),
        int(bond.GetStereo()), int(bond.IsInRing()),
    ], dtype=torch.float)

def mol_to_graph(mol):
    if mol is None: return None
    x = torch.stack([atom_features(a) for a in mol.GetAtoms()])
    edge_index, edge_attr = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        edge_index.append([i, j]); edge_attr.append(f)
        edge_index.append([j, i]); edge_attr.append(f)
    if len(edge_index) == 0: return None
    return Data(x=x, edge_index=torch.tensor(edge_index).t().contiguous(),
                edge_attr=torch.stack(edge_attr))

# --- Model Actions ---
@st.cache_resource
def load_model(device="cpu"):
    model = GIN_edge().to(device)
    model.load_state_dict(torch.load("tox21_gnn.pt", map_location=device))
    model.eval()
    return model

def predict_smiles(smiles, model, device="cpu"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError("Invalid SMILES")
    g = mol_to_graph(mol)
    batch = Batch.from_data_list([g]).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return mol, probs

def explain_molecule(smiles, model, device, task_id):
    mol = Chem.MolFromSmiles(smiles)
    g = mol_to_graph(mol)
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

# --- Chemistry Metrics ---
def compute_druglikeness(mol):
    props = {
        "MW": Descriptors.MolWt(mol),
        "logP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "Rotatable": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Rings": rdMolDescriptors.CalcNumRings(mol),
        "QED": QED.qed(mol),
        "SA": calc_sa_score(mol)
    }
    props["Lipinski"] = sum([props["MW"] > 500, props["logP"] > 5, props["HBD"] > 5, props["HBA"] > 10])
    props["PAINS"] = [m.GetDescription() for m in pains_catalog.GetMatches(mol)]
    return props

# --- Analog Generation ---
def generate_selfies_analogs(seed_smiles, n_candidates=40, max_mutations=2):
    if not SELFIES_AVAILABLE: return []
    seed_sf = sf.encoder(seed_smiles)
    seed_tokens = list(sf.split_selfies(seed_sf))
    alphabet = list(sf.get_semantic_robust_alphabet())
    results = set()
    tries = 0
    while len(results) < n_candidates and tries < n_candidates * 20:
        tries += 1
        toks = seed_tokens.copy()
        for _ in range(random.randint(1, max_mutations)):
            idx = random.randrange(len(toks))
            toks[idx] = random.choice(alphabet)
        smi = sf.decoder("".join(toks))
        if smi and Chem.MolFromSmiles(smi): results.add(Chem.MolToSmiles(Chem.MolFromSmiles(smi)))
    return list(results)

# def score_candidate(smiles, model, device="cpu"):
#     try:
#         mol, tox = predict_smiles(smiles, model, device)
#     except: return None
#     props = compute_druglikeness(mol)
#     mean_tox = float(np.mean(tox))
#     sa_term = (1 - min(max((props["SA"] - 1)/9, 0), 1)) if props["SA"] else 0.5
#     score = (0.4*props["QED"] + 0.4*(1-mean_tox) + 0.15*sa_term + 0.05*(1 - props["Lipinski"]/2))
#     if props["PAINS"]: score -= 0.5
#     return {"smiles": smiles, "mol": mol, "score": score, "tox": mean_tox, "QED": props["QED"], "Lip": props["Lipinski"], "PAINS": props["PAINS"]}

def get_iupac_from_cactus(smiles):
    """Helper to fetch IUPAC names from NIH Cactus service"""
    try:
        encoded_smiles = urllib.parse.quote(smiles)
        url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_smiles}/iupac_name"
        # We add a 2-second timeout so one slow request doesn't freeze the whole app
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            return response.text.strip()
        return "Unknown Structure"
    except:
        return "N/A (Timeout/Error)"


def score_candidate(smiles, model, device="cpu"):
    try:
        mol, tox = predict_smiles(smiles, model, device)
    except:
        return None

    props = compute_druglikeness(mol)
    mean_tox = float(np.mean(tox))

    # 🔥 NEW: Fetch IUPAC name for the analog
    iupac_name = get_iupac_from_cactus(smiles)

    sa_term = (1 - min(max((props["SA"] - 1) / 9, 0), 1)) if props["SA"] else 0.5
    score = (0.4 * props["QED"] + 0.4 * (1 - mean_tox) + 0.15 * sa_term + 0.05 * (1 - props["Lipinski"] / 2))

    if props["PAINS"]:
        score -= 0.5

    return {
        "smiles": smiles,
        "iupac": iupac_name,  # 🔥 Add to dictionary
        "mol": mol,
        "score": score,
        "tox": mean_tox,
        "QED": props["QED"],
        "Lip": props["Lipinski"],
        "PAINS": props["PAINS"]
    }