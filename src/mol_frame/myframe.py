import numpy as np
import scipy
import scipy.spatial
import string
import os,re
from os.path import exists
import hydra
import torch
from omegaconf import OmegaConf
from openbabel import openbabel

from src.mol_frame.mychemical import initialize_chemdata, load_pdb_ideal_sdf_strings
from src.mol_frame.mychemical import ChemicalData as ChemData
from src.mol_frame.mykinematics import get_chirals
from src.mol_frame.myutil import get_bond_feats, get_nxgraph, get_atom_frames, get_automorphs


def clean_sdffile(filename):
    # lowercase the 2nd letter of the element name (e.g. FE->Fe) so openbabel can parse it correctly
    lines2 = []
    with open(filename) as f:
        lines = f.readlines()
        num_atoms = int(lines[3][:3])
        for i in range(len(lines)):
            if i>=4 and i<4+num_atoms:
                lines2.append(lines[i][:32]+lines[i][32].lower()+lines[i][33:])
            else:
                lines2.append(lines[i])
    molstring = ''.join(lines2)

    return molstring


def parse_mol(filename, filetype="mol2", string=False, remove_H=True, find_automorphs=True, generate_conformer: bool = False):
    # examples/small_molecule/XG4.sdf sdf False True True True
    # Stop openbabel warnings
    openbabel.obErrorLog.StopLogging()
    
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat(filetype)
    obmol = openbabel.OBMol()
    if string:
        obConversion.ReadString(obmol,filename)
    elif filetype=='sdf':
        molstring = clean_sdffile(filename)
        obConversion.ReadString(obmol,molstring)
    else:
        obConversion.ReadFile(obmol,filename)
    if generate_conformer:
        builder = openbabel.OBBuilder()
        builder.Build(obmol)
        ff = openbabel.OBForceField.FindForceField("mmff94")
        did_setup = ff.Setup(obmol)
        if did_setup:
            ff.FastRotorSearch()
            ff.GetCoordinates(obmol)
        else:
            raise ValueError(f"Failed to generate 3D coordinates for molecule {filename}.")
    if remove_H:
        obmol.DeleteHydrogens()
        # the above sometimes fails to get all the hydrogens
        i = 1
        while i < obmol.NumAtoms()+1:
            if obmol.GetAtom(i).GetAtomicNum()==1:
                obmol.DeleteAtom(obmol.GetAtom(i))
            else:
                i += 1
    atomtypes = [ChemData().atomnum2atomtype.get(obmol.GetAtom(i).GetAtomicNum(), 'ATM') 
                 for i in range(1, obmol.NumAtoms()+1)]
    msa = torch.tensor([ChemData().aa2num[x] for x in atomtypes])
    ins = torch.zeros_like(msa)

    atom_coords = torch.tensor([[obmol.GetAtom(i).x(),obmol.GetAtom(i).y(), obmol.GetAtom(i).z()] 
                                for i in range(1, obmol.NumAtoms()+1)]).unsqueeze(0) # (1, natoms, 3)
    mask = torch.full(atom_coords.shape[:-1], True) # (1, natoms,)

    if find_automorphs:
        atom_coords, mask = get_automorphs(obmol, atom_coords[0], mask[0])

    return obmol, msa, ins, atom_coords, mask

def compute_features_from_obmol(obmol, msa, xyz):
    L = msa.shape[0]
    ins = torch.zeros_like(msa)
    bond_feats = get_bond_feats(obmol)
    chirals = get_chirals(obmol, xyz[0])
    G = get_nxgraph(obmol)
    atom_frames = get_atom_frames(msa, G) # (num-atoms, 3, 2)
    msa, ins = msa[None], ins[None]
    return [msa, ins, bond_feats, chirals, atom_frames]

def croods_to_frames(xyz, atom_frames): # atom-frames -- (1, num-atoms, 3, 2)
    
    atom_crds = xyz.clone() # (1, num-atoms, 1, 3)
    B, atom_L = atom_crds.shape[:2] # 1, num-atoms
    xyz_frame = torch.zeros((B, atom_L, 3, 3))
    frames_reindex = torch.zeros(atom_frames.shape[:-1]) # (1, num-atoms, 3)
    for i in range(atom_L):
        frames_reindex[:, i, :] = (i+atom_frames[..., i, :, 0]) # get the absolute index
    frames_reindex = frames_reindex.long()
    xyz_frame[:, :, :3] = atom_crds.reshape(atom_L, 3)[frames_reindex]
    return xyz_frame

def calc_rotate_imat(N, CA, C):
    p1 = N - CA
    epsilon = 1e-10

    norm_p1 = np.linalg.norm(p1, axis=-1, keepdims=True)
    norm_p1 = np.where(norm_p1 == 0, epsilon, norm_p1)
    x = p1 / norm_p1

    p2 = C - N

    inner_1 = np.matmul(np.expand_dims(p1, axis=1), np.expand_dims(p1, axis=2))[:, :, 0]
    inner_2 = np.matmul(np.expand_dims(-p1, axis=1), np.expand_dims(p2, axis=2))[:, :, 0]

    inner_2_safe = np.where(inner_2 == 0, epsilon, inner_2)
    alpha = inner_1 / inner_2_safe

    y = alpha * p2 + p1

    norm_y = np.linalg.norm(y, axis=-1, keepdims=True)
    norm_y = np.where(norm_y == 0, epsilon, norm_y)
    y = y / norm_y

    z = np.cross(x, y)

    mat = np.concatenate([x, y, z], axis=-1)
    mat = mat.reshape(*mat.shape[:-1], 3, 3)
    assert not np.isnan(np.sum(mat)), "the calculated rotation matrix contains NaN."
    
    return mat

class chem_config:
    use_phospate_frames_for_NA = True 
    use_cif_ordering_for_trp = True
    

def process_sdf(input_file):
    obmol, msa, ins, xyz, mask = parse_mol(input_file,'sdf',False,True,False,True) # msa -- (1, num-atoms); xyz -- (1, num-atoms, 3)
    for bond in openbabel.OBMolBondIter(obmol):
        obmol.DeleteBond(bond)
    msa, ins, bond_feats, chirals, atom_frames = compute_features_from_obmol(obmol, msa, xyz) # atom-frames -- (num-atoms, 3, 2)
    T = xyz
    xyz = croods_to_frames(xyz.unsqueeze(-2), atom_frames.unsqueeze(0)) # xyz -- (1, num-atoms/num-frames, 3 , 3)
    IR = torch.tensor(calc_rotate_imat(xyz[0,:,0,:].numpy(),xyz[0,:,1,:].numpy(),xyz[0,:,2,:].numpy())).unsqueeze(0) # (1, num-atoms, 3, 3)
    return msa[0].numpy(), IR[0].numpy(), T[0].numpy(), atom_frames.numpy(), obmol
    
if __name__ == "__main__":
    # process_sdf()
    pass
