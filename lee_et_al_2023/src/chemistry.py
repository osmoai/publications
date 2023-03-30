import functools
from typing import Callable, Optional, Iterable, Iterator
import numpy as np

from rdkit import Chem, DataStructs
import rdkit.Chem.rdMolDescriptors

# rdkit's DataStructs.ExplicitBitVect is more efficient for rdkit-internal use.
get_morgan_fp: Callable[[Chem.Mol], DataStructs.ExplicitBitVect] = functools.partial(
    Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect, radius=2, nBits=2048
)

def tanimoto_sim(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """Compute Tanimoto similarity for just two molecules."""
    return DataStructs.FingerprintSimilarity(
        get_morgan_fp(mol1), get_morgan_fp(mol2), metric=DataStructs.TanimotoSimilarity
    )


def _bulk_similarity(
    mols1: Iterable[Chem.Mol], mols2: Optional[Iterable[Chem.Mol]] = None
) -> Iterator[np.ndarray]:
    if mols2 is None:
        mols2 = mols1
    mol1_fps = map(get_morgan_fp, mols1)
    mol2_fps = tuple(map(get_morgan_fp, mols2))
    for fp in mol1_fps:
        yield DataStructs.BulkTanimotoSimilarity(fp, mol2_fps)


def canonical_smiles(smiles: str, kekulize: bool = False) -> str:
    """Use rdkit to convert the `smiles` string to canonical form"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:  # If a mol object was successfully create (i.e. not `None`)
        if kekulize:
            Chem.Kekulize(mol)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    else:  # No mol object means the `smiles` string was invalid
        smiles = ""
    return smiles
