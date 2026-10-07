"""Geração de atributos moleculares para a base SIDER.

Para cada SMILES:
  1. padroniza a molécula e mantém o maior fragmento (remove contra-íons de sais);
  2. calcula os descritores 2D do RDKit (físico-químicos, topológicos, contagens);
  3. calcula o fingerprint de Morgan (ECFP4, raio 2, 1024 bits);
  4. calcula as chaves MACCS (167 bits).

Saída: data/features.npz com X_desc, X_morgan, X_maccs, Y, nomes de atributos e rótulos.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]


def largest_fragment(mol):
    chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
    return chooser.choose(mol)


def main():
    df = pd.read_csv(ROOT / "data" / "sider.csv")
    label_names = list(df.columns[1:])
    Y = df[label_names].values.astype(np.int8)

    desc_names = [name for name, _ in Descriptors.descList]
    X_desc, X_morgan, X_maccs, n_frag = [], [], [], []
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        n_frag.append(len(Chem.GetMolFrags(mol)))
        mol = largest_fragment(mol)
        row = []
        for _, fn in Descriptors.descList:
            try:
                row.append(float(fn(mol)))
            except Exception:
                row.append(np.nan)
        X_desc.append(row)
        X_morgan.append(np.array(gen.GetFingerprintAsNumPy(mol), dtype=np.uint8))
        X_maccs.append(np.array(list(MACCSkeys.GenMACCSKeys(mol)), dtype=np.uint8))

    X_desc = np.array(X_desc, dtype=np.float64)
    X_desc[~np.isfinite(X_desc)] = np.nan
    # remove descritores constantes ou com muitos faltantes
    keep = (np.nanstd(X_desc, axis=0) > 0) & (np.isnan(X_desc).mean(axis=0) < 0.05)
    X_desc = X_desc[:, keep]
    desc_names = [n for n, k in zip(desc_names, keep) if k]
    # imputação pela mediana (é feita antes da divisão em folds, mas não usa rótulos)
    med = np.nanmedian(X_desc, axis=0)
    idx = np.where(np.isnan(X_desc))
    X_desc[idx] = np.take(med, idx[1])
    # transformação log-assimétrica para domar caudas pesadas (ex.: MolWt de peptídeos)
    X_desc = np.sign(X_desc) * np.log1p(np.abs(X_desc))

    X_morgan = np.vstack(X_morgan)
    X_maccs = np.vstack(X_maccs)
    print(f"descritores: {X_desc.shape}, morgan: {X_morgan.shape}, maccs: {X_maccs.shape}")
    print(f"moléculas com mais de um fragmento (sais/misturas): {sum(n > 1 for n in n_frag)}")

    np.savez_compressed(
        ROOT / "data" / "features.npz",
        X_desc=X_desc, X_morgan=X_morgan, X_maccs=X_maccs, Y=Y,
        desc_names=np.array(desc_names), label_names=np.array(label_names),
        n_frag=np.array(n_frag),
    )


if __name__ == "__main__":
    main()
