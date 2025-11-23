import os
import csv
import torch
import multiprocessing as mp
from rdkit import Chem
from openbabel import pybel
from tqdm import tqdm


SAVE_DIR = "coords"
os.makedirs(SAVE_DIR, exist_ok=True)


def generate_3d_openbabel(smiles):
    try:
        mol = pybel.readstring("smi", smiles)
        mol.make3D()
        coords = [list(a.coords) for a in mol.atoms]
        return coords
    except Exception:
        return None


def worker_process(task):
    idx, smiles = task
    coords = generate_3d_openbabel(smiles)

    if coords is None:
        return idx, False

    save_path = os.path.join(SAVE_DIR, f"{idx:08d}.pt")
    torch.save(coords, save_path)

    return idx, True


if __name__ == "__main__":
    with open("data/pubchem-10m-clean.txt") as f:
        smiles_list = [row[-1] for row in csv.reader(f)]

    N = len(smiles_list)
    print("Total SMILES:", N)

    num_workers = mp.cpu_count()
    print(f"Using {num_workers} workers with multiprocessing")

    pool = mp.Pool(num_workers)

    valid_indices = []
    invalid_indices = []

    pbar = tqdm(total=N, desc="generate 3d", dynamic_ncols=True)

    for idx, flag in pool.imap_unordered(worker_process, enumerate(smiles_list), chunksize=200):
        if flag:
            valid_indices.append(idx)
        else:
            invalid_indices.append(idx)

        pbar.set_postfix({
            "valid": len(valid_indices),
            "invalid": len(invalid_indices),
            "valid_%": f"{100 * len(valid_indices) / (len(valid_indices)+len(invalid_indices)):.2f}%"
        })
        pbar.update()

    pbar.close()
    pool.close()
    pool.join()

    torch.save(invalid_indices, "valid_idx.pt")
    with open("invalid_idx.txt", "w") as f:
        for i in invalid_indices:
            f.write(str(i) + "\n")