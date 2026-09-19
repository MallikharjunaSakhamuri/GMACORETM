"""Conversion of SMILES strings into PyTorch Geometric molecular graphs.

Node features are partitioned into two groups, consistent with Section 4.1.1 of
the manuscript:

    x_cat   categorical atomic descriptors (atom identity, chirality tag)
    x_phys  continuous physicochemical and structural descriptors
            (atomic-weight contribution, logP contribution, formal charge,
             hybridization, aromaticity, hydrogen count, valence, degree)

The two groups are concatenated prior to the learnable transformation and are
encoded jointly by a single shared MLP node encoder. The separation is a
notational device only and does not imply two separate neural encoders.

The default featurization is two dimensional. Three-dimensional conformer
generation is available behind an explicit flag but is disabled by default,
because the reported experiments do not use geometric information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Data

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, rdMolTransforms

    RDLogger.DisableLog("rdApp.*")
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "RDKit is required for molecular featurization. "
        "Install it with `conda install -c conda-forge rdkit`."
    ) from exc

logger = logging.getLogger(__name__)

NUM_NODE_CATEGORICAL = 2
NUM_NODE_PHYSICOCHEMICAL = 8
NUM_EDGE_FEATURES = 5

ATOM_LIST = list(range(1, 119))

CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]

BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

BOND_DIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


@dataclass
class FeaturizerConfig:
    """Options controlling molecular graph construction."""

    add_explicit_hydrogens: bool = False
    generate_conformer: bool = False
    max_heavy_atoms: Optional[int] = None
    min_heavy_atoms: int = 2


class MolecularFeaturizer:
    """Builds PyG `Data` objects from SMILES strings."""

    def __init__(self, config: Optional[FeaturizerConfig] = None) -> None:
        self.config = config or FeaturizerConfig()
        self._atom_index = {z: i for i, z in enumerate(ATOM_LIST)}
        self._chirality_index = {c: i for i, c in enumerate(CHIRALITY_LIST)}
        self._bond_index = {b: i for i, b in enumerate(BOND_LIST)}
        self._bond_dir_index = {d: i for i, d in enumerate(BOND_DIR_LIST)}
        self._element_contribution_cache: dict = {}

    # ------------------------------------------------------------------
    # Atom level features
    # ------------------------------------------------------------------
    def _element_contributions(self, symbol: str) -> Tuple[float, float]:
        """Atomic weight and logP contribution for an isolated element.

        Results are cached because the underlying RDKit calls are relatively
        expensive and depend only on the element symbol.
        """
        if symbol in self._element_contribution_cache:
            return self._element_contribution_cache[symbol]

        weight, logp = 0.0, 0.0
        probe = Chem.MolFromSmiles(f"[{symbol}]")
        if probe is not None:
            try:
                weight = float(Descriptors.ExactMolWt(probe))
                logp = float(Descriptors.MolLogP(probe))
            except Exception:  # noqa: BLE001 - RDKit raises broad exceptions
                logger.debug("Descriptor computation failed for element %s", symbol)

        self._element_contribution_cache[symbol] = (weight, logp)
        return weight, logp

    def _atom_features(self, atom: Chem.Atom) -> Tuple[List[int], List[float]]:
        categorical = [
            self._atom_index.get(atom.GetAtomicNum(), 0),
            self._chirality_index.get(atom.GetChiralTag(), 0),
        ]

        weight, logp = self._element_contributions(atom.GetSymbol())
        physicochemical = [
            weight,
            logp,
            float(atom.GetFormalCharge()),
            float(int(atom.GetHybridization())),
            float(atom.GetIsAromatic()),
            float(atom.GetTotalNumHs()),
            float(atom.GetTotalValence()),
            float(atom.GetDegree()),
        ]
        return categorical, physicochemical

    def _node_features(self, mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
        categorical, physicochemical = [], []
        for atom in mol.GetAtoms():
            cat, phys = self._atom_features(atom)
            categorical.append(cat)
            physicochemical.append(phys)

        x_cat = torch.tensor(categorical, dtype=torch.long)
        x_phys = torch.tensor(physicochemical, dtype=torch.float)
        return x_cat, x_phys

    # ------------------------------------------------------------------
    # Bond level features
    # ------------------------------------------------------------------
    @staticmethod
    def _is_rotatable(bond: Chem.Bond) -> bool:
        return (
            bond.GetBondType() == Chem.rdchem.BondType.SINGLE
            and not bond.IsInRing()
            and bond.GetBeginAtom().GetDegree() > 1
            and bond.GetEndAtom().GetDegree() > 1
        )

    @staticmethod
    def _bond_length(mol: Chem.Mol, start: int, end: int) -> float:
        if mol.GetNumConformers() == 0:
            return 0.0
        conformer = mol.GetConformer()
        if not conformer.Is3D():
            return 0.0
        try:
            return float(rdMolTransforms.GetBondLength(conformer, start, end))
        except Exception:  # noqa: BLE001
            return 0.0

    def _edge_features(self, mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
        rows: List[int] = []
        cols: List[int] = []
        features: List[List[float]] = []

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feature = [
                float(self._bond_index.get(bond.GetBondType(), 0)),
                float(self._bond_dir_index.get(bond.GetBondDir(), 0)),
                float(bond.GetIsConjugated()),
                float(self._is_rotatable(bond)),
                self._bond_length(mol, start, end),
            ]
            # Each bond is stored in both directions.
            rows += [start, end]
            cols += [end, start]
            features += [feature, feature]

        if not rows:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, NUM_EDGE_FEATURES), dtype=torch.float)
            return edge_index, edge_attr

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(features, dtype=torch.float)
        return edge_index, edge_attr

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def _prepare_mol(self, smiles: str) -> Optional[Chem.Mol]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if self.config.add_explicit_hydrogens:
            mol = Chem.AddHs(mol)

        try:
            Chem.SanitizeMol(mol)
        except Exception:  # noqa: BLE001
            return None

        if self.config.generate_conformer:
            if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
                return None
            if AllChem.MMFFOptimizeMolecule(mol) != 0:
                AllChem.UFFOptimizeMolecule(mol)

        return mol

    def _passes_size_filter(self, mol: Chem.Mol) -> bool:
        heavy = mol.GetNumHeavyAtoms()
        if heavy < self.config.min_heavy_atoms:
            return False
        if self.config.max_heavy_atoms is not None and heavy > self.config.max_heavy_atoms:
            return False
        return True

    def __call__(self, smiles: str, label: Optional[Sequence[float]] = None) -> Optional[Data]:
        """Return a molecular graph, or `None` if the SMILES cannot be used."""
        mol = self._prepare_mol(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        if not self._passes_size_filter(mol):
            return None

        x_cat, x_phys = self._node_features(mol)
        edge_index, edge_attr = self._edge_features(mol)

        # Isolated atoms carry no contrastive signal under message passing.
        if edge_index.size(1) == 0:
            return None

        data = Data(
            x_cat=x_cat,
            x_phys=x_phys,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=x_cat.size(0),
        )
        data.smiles = smiles

        if label is not None:
            data.y = torch.tensor([label], dtype=torch.float)

        return data

    def featurize_many(
        self,
        smiles_list: Sequence[str],
        labels: Optional[Sequence[Sequence[float]]] = None,
        progress: bool = True,
    ) -> Tuple[List[Data], List[str]]:
        """Featurize a list of SMILES, returning graphs and the failed strings."""
        iterator = range(len(smiles_list))
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Featurizing molecules")

        graphs: List[Data] = []
        failed: List[str] = []
        for index in iterator:
            smiles = smiles_list[index]
            label = labels[index] if labels is not None else None
            graph = self(smiles, label)
            if graph is None:
                failed.append(smiles)
            else:
                graphs.append(graph)

        if failed:
            logger.info("Skipped %d of %d SMILES strings", len(failed), len(smiles_list))
        return graphs, failed


def feature_dimensions() -> Tuple[int, int]:
    """Return the node and edge feature dimensionalities produced above."""
    return NUM_NODE_CATEGORICAL + NUM_NODE_PHYSICOCHEMICAL, NUM_EDGE_FEATURES
