import copy
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from mol_graph import array_rep_from_smiles
import xml.etree.ElementTree as ET
from IPython.display import SVG, display
from matplotlib import colors
import rdkit.Chem as Chem
import json
from rdkit import DataStructs
from rdkit import Geometry
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdMolDescriptors as rdMD

class Mol_Tokenizer():
    def __init__(self,tokens_id_file):
        self.vocab = json.load(open(r'{}'.format(tokens_id_file),'r'))
        self.MST_MAX_WEIGHT = 100
        self.get_vocab_size = len(self.vocab.keys())
        self.id_to_token = {value:key for key,value in self.vocab.items()}
    def tokenize(self,smiles):
        mol = Chem.MolFromSmiles(r'{}'.format(smiles))
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        ids,edge = self.tree_decomp(mol) 
        motif_list = []
        for id_ in ids:
            _,token_mols = self.get_clique_mol(mol,id_)
            token_id = self.vocab.get(token_mols)
            if token_id!=None:
                motif_list.append(token_id)
            else: 
                motif_list.append(self.vocab.get('<unk>'))
        return motif_list,edge,ids
    def sanitize(self,mol):
        try:
            smiles = self.get_smiles(mol)
            mol = self.get_mol(smiles)
        except Exception as e:
            return None
        return mol
    def get_mol(self,smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.Kekulize(mol)
        return mol
    def get_smiles(self,mol):
        return Chem.MolToSmiles(mol, kekuleSmiles=True)
    def get_clique_mol(self,mol,atoms_ids):
    # get the fragment of clique
        smiles = Chem.MolFragmentToSmiles(mol, atoms_ids, kekuleSmiles=False) 
        new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        new_mol = self.copy_edit_mol(new_mol).GetMol()
        new_mol = self.sanitize(new_mol)  # We assume this is not None
        return new_mol,smiles
    def copy_atom(self,atom):
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        return new_atom
    def copy_edit_mol(self,mol):
        new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
        for atom in mol.GetAtoms():
            new_atom = self.copy_atom(atom)
            new_mol.AddAtom(new_atom)
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            new_mol.AddBond(a1, a2, bt)
        return new_mol
    def tree_decomp(self,mol):
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                cliques.append([a1, a2])

        # get rings
        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)

        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        # Merge Rings with intersection > 2 atoms
        for i in range(len(cliques)):
            if len(cliques[i]) <= 2: continue
            for atom in cliques[i]:
                for j in nei_list[atom]:
                    if i >= j or len(cliques[j]) <= 2: continue
                    inter = set(cliques[i]) & set(cliques[j])
                    if len(inter) > 2:
                        cliques[i].extend(cliques[j])
                        cliques[i] = list(set(cliques[i]))
                        cliques[j] = []

        cliques = [c for c in cliques if len(c) > 0]
        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        # Build edges and add singleton cliques
        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1:
                continue
            cnei = nei_list[atom]
            bonds = [c for c in cnei if len(cliques[c]) == 2]
            rings = [c for c in cnei if len(cliques[c]) > 4]
            if len(bonds) > 2 or (len(bonds) == 2 and len(
                    cnei) > 2):  # In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = 1
            elif len(rings) > 2:  # Multiple (n>2) complex rings
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = self.MST_MAX_WEIGHT - 1
            else:
                for i in range(len(cnei)):
                    for j in range(i + 1, len(cnei)):
                        c1, c2 = cnei[i], cnei[j]
                        inter = set(cliques[c1]) & set(cliques[c2])
                        if edges[(c1, c2)] < len(inter):
                            edges[(c1, c2)] = len(inter)  # cnei[i] < cnei[j] by construction

        edges = [u + (self.MST_MAX_WEIGHT - v,) for u, v in edges.items()]
        if len(edges) == 0:
            return cliques, edges

        # Compute Maximum Spanning Tree
        row, col, data = zip(*edges)
        n_clique = len(cliques)
        clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
        junc_tree = minimum_spanning_tree(clique_graph)
        row, col = junc_tree.nonzero()
        edges = [(row[i], col[i]) for i in range(len(row))]
        return (cliques, edges)

class GroundTruthSub(): ### To visualize substructures described in literature
    def __init__(self,smiles,atoms_to_highlight,colors = '#c44240'):
        self.smiles = smiles 
        self.atoms_to_highlight = atoms_to_highlight
        self.colors = colors
        self.bond_diaplay = 'fill:{};fill-rule:evenodd;stroke:{};stroke-width:3px;stroke-linejoin:miter;stroke-opacity:1.0'
        self.atom_display = 'fill:{};fill-rule:evenodd;stroke:{};stroke-width:0.1px;stroke-linejoin:miter;stroke-opacity:1.0'
        self.svg_init,self.mol = self.init_svg() 
        self.bonds_to_highlight = self.init_bonds_to_highlight() 
    def init_svg(self):
        mol = Chem.MolFromSmiles(self.smiles)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        d = rdMolDraw2D.MolDraw2DSVG(250, 250)
        d.SetFontSize(0.68)
        tm = rdMolDraw2D.PrepareMolForDrawing(mol)
        hcolor = colors.to_rgb('white')
        d.drawOptions().setHighlightColour(hcolor)
        d.drawOptions().updateAtomPalette({i: (0, 0, 0, 1) for i in range(100)}) 
        d.drawOptions().highlightRadius=.10
        d.DrawMolecule(tm,highlightAtoms=self.atoms_to_highlight)
        d.FinishDrawing()
        svg = d.GetDrawingText()
        svg = svg.replace('svg:','')
        return svg,mol 
    def init_bonds_to_highlight(self):
        all_bond = set()
        for i in self.atoms_to_highlight:
            for j in self.atoms_to_highlight:
                if i!=j:
                    b = self.mol.GetBondBetweenAtoms(i,j)
                    if b!=None:
                        all_bond.add(b.GetIdx())
        return list(all_bond)
    def visualize(self):
        parsed = ET.fromstring(self.svg_init)
        new_parsed = copy.deepcopy(parsed)
        for i,j in enumerate(parsed):
            if j.tag == '{http://www.w3.org/2000/svg}path':
                if j.attrib.get('class')!=None:
                    bond = j.attrib['class'].split('-')
                    if int(bond[-1]) in self.bonds_to_highlight and 'bond' in bond[0]:
                        j.attrib['style']=self.bond_diaplay.format(self.colors,self.colors)
                    elif int(bond[-1]) in self.atoms_to_highlight and 'atom' in bond[0]:
                        j.attrib['style']=self.atom_display.format(self.colors,self.colors)
            
            # elif j.tag == '{http://www.w3.org/2000/svg}ellipse':
            #     j.attrib['style']=node_display 
            new_parsed[i]=j 
        ET.register_namespace('', "http://www.w3.org/2000/svg")
        new_svg = ET.tostring(new_parsed,encoding='iso-8859-1',method='xml').decode('iso-8859-1')
        display(SVG(new_svg))

def MapAtomFromWeights(mol,weights,draw2d,contourLines=0,sigma=None,gridResolution=0.06): 
    # A modified version from rdkit SimilarityMaps.GetSimilarityMapFromWeights
    """
    Generates the similarity map for a molecule given the atomic weights.

    Parameters:
      mol -- the molecule of interest
      colorMap -- the matplotlib color map scheme, default is custom PiWG color map
      scale -- the scaling: scale < 0 -> the absolute maximum weight is used as maximum scale
                            scale = double -> this is the maximum scale
      contourLines -- if integer number N: N contour lines are drawn
                      if list(numbers): contour lines at these numbers are drawn
    """
    if mol.GetNumAtoms() < 2:
        raise ValueError("too few atoms")
    mol = rdMolDraw2D.PrepareMolForDrawing(mol,addChiralHs=False)
    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
    if sigma is None:
        if mol.GetNumBonds() > 0:
            bond = mol.GetBondWithIdx(0)
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            sigma = 0.3 * (mol.GetConformer().GetAtomPosition(idx1)-mol.GetConformer().GetAtomPosition(idx2)).Length()
        else:
            sigma = 0.3 * (mol.GetConformer().GetAtomPosition(0)-mol.GetConformer().GetAtomPosition(1)).Length()
        sigma = round(sigma, 2)
    sigmas = [sigma]*mol.GetNumAtoms()
    locs=[]
    for i in range(mol.GetNumAtoms()):
        p = mol.GetConformer().GetAtomPosition(i)
        locs.append(Geometry.Point2D(p.x,p.y))
    draw2d.ClearDrawing()
    ps = Draw.ContourParams()
    ps.fillGrid=True
    ps.gridResolution=gridResolution
    ps.extraGridPadding = 1
    Draw.ContourAndDrawGaussians(draw2d,locs,weights,sigmas,nContours=contourLines,params=ps)
    draw2d.drawOptions().clearBackground = False
    draw2d.DrawMolecule(mol)
    return draw2d

def get_numed_mol(smiles,addH = False,img_size=400):
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if addH:
        mol = Chem.AddHs(mol) 
    for i ,atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i)
    drawer = Chem.Draw.MolDraw2DSVG(img_size, img_size)
    mol = rdMolDraw2D.PrepareMolForDrawing(mol,addChiralHs=False)
    drawer.ClearDrawing()
    drawer.drawOptions().clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:','')
    return SVG(svg)

degrees = [0,1,2,3,4,5]
def atoms_by_order(rdkit_ix,max_atom):
    order_matrix = np.zeros((max_atom,max_atom),'float32')
    for i,j in enumerate(rdkit_ix):
        order_matrix[j,i] = 1
    return order_matrix
def connectivity_to_Matrix(array_rep, total_num_features,degree):#
    total_num = []
    mat = np.zeros((total_num_features, total_num_features),'float32') 
    if degree == 0:   
        for i,x in enumerate(array_rep[('atom_neighbors',degree)].astype('int32')):
            mat[i,x] = 1        
        return mat
    else:
        for i in range(degree):
            atom_neighbors_list = array_rep[('atom_neighbors',i)].astype('int32')
            total_num.append(len(atom_neighbors_list))
        total_num = sum(total_num)
        for i,x in enumerate(array_rep[('atom_neighbors',degree)].astype('int32')):
            mat[total_num + i,x] = 1
        return mat
def bond_features_by_degree(total_atoms,summed_degrees,degree):
    mat = np.zeros((total_atoms,10),'float32')
    total_num = []
    if degree == 0:
        for i,x in enumerate(summed_degrees[0]):
            mat[i] = x
        return mat
    else:
        for i in range(degree):
            total_num.append(len(summed_degrees[i]))
        total_num = sum(total_num)
        for i,x in enumerate(summed_degrees[degree]):
            mat[total_num + i] = x
        return mat
def extract_bondfeatures_of_neighbors_by_degree(array_rep):
    """
    Sums up all bond features that connect to the atoms (sorted by degree)
    
    Returns:
    ----------
    
    list with elements of shape: [(num_atoms_degree_0, 6), (num_atoms_degree_1, 6), (num_atoms_degree_2, 6), etc....]
    
    e.g.:
    
    >> print [x.shape for x in extract_bondfeatures_of_neighbors_by_degree(array_rep)]
    
    [(0,), (269, 6), (524, 6), (297, 6), (25, 6), (0,)]  
    
    """
    bond_features_by_atom_by_degree = []
    for degree in degrees:
        bond_features = array_rep['bond_features']
        bond_neighbors_list = array_rep[('bond_neighbors', degree)]
        summed_bond_neighbors = bond_features[bond_neighbors_list].sum(axis=1)
        bond_features_by_atom_by_degree.append(summed_bond_neighbors)
    return bond_features_by_atom_by_degree
if __name__ == "__main__":
    mol_token = Mol_Tokenizer('token_id.json')
    token_ids,edges = mol_token.tokenize('CC1CCC2=CC(F)=CC3=C2N1C=C(C(O)=O)C3=O')
    print(token_ids)