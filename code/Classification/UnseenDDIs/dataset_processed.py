from cProfile import label
import pandas as pd
import numpy as np
import tensorflow as tf
import networkx as nx
from rdkit import Chem
from utils import *

def get_adj_matrix(num_list,edges):
    adjoin_matrix = np.eye(len(num_list))
    for edge in edges:
        u = edge[0]
        v = edge[1]
        adjoin_matrix[u,v] = 1.0
        adjoin_matrix[v,u] = 1.0
    return adjoin_matrix

def get_dist_matrix(num_list,edges):
    make_graph = nx.Graph()
    make_graph.add_edges_from(edges)
    dist_matrix = np.zeros((len(num_list),len(num_list)))
    dist_matrix.fill(1e9)
    row, col = np.diag_indices_from(dist_matrix)
    dist_matrix[row,col] = 0
    graph_nodes = sorted(make_graph.nodes.keys())
    all_distance = dict(nx.all_pairs_shortest_path_length(make_graph))
    for dist in graph_nodes:
        node_relative_distance = dict(sorted(all_distance[dist].items(),key = lambda x:x[0]))
        temp_node_dist_dict = {i:node_relative_distance.get(i) if \
        node_relative_distance.get(i)!= None else 1e9 for i in graph_nodes} ### Refer to rdkit Chem.GetDistanceMatrix(mol)
        temp_node_dist_list = list(temp_node_dist_dict.values()) 
        dist_matrix[dist][graph_nodes] =  temp_node_dist_list
    return dist_matrix.astype(np.float32)

def molgraph_rep(smi,cliques):
    def atom_to_motif_match(atom_order,cliques):
        atom_order = atom_order.tolist()
        temp_matrix = np.zeros((len(cliques),len(atom_order)))
        for th,cli in enumerate(cliques):
            for i in cli:
                temp_matrix[th,atom_order.index(i)] = 1
        return temp_matrix
    def get_adj_dist_matrix(mol_graph,smi):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        num_atoms = mol.GetNumAtoms() 
        adjoin_matrix_temp = np.eye(num_atoms)
        adj_matrix = Chem.GetAdjacencyMatrix(mol)
        adj_matrix = (adjoin_matrix_temp + adj_matrix)[:,mol_graph['rdkit_ix']][mol_graph['rdkit_ix']]
        dist_matrix = Chem.GetDistanceMatrix(mol)[:,mol_graph['rdkit_ix']][mol_graph['rdkit_ix']]
        return adj_matrix,dist_matrix 
    single_dict = {'input_atom_features':[],
            'atom_match_matrix':[],
            'sum_atoms':[],
            'adj_matrix':[],
            'dist_matrix':[]
            }
    array_rep = array_rep_from_smiles(smi)
    summed_degrees = extract_bondfeatures_of_neighbors_by_degree(array_rep)  
    atom_features = array_rep['atom_features'] 
    all_bond_features = []
    for degree in degrees:
        atom_neighbors_list = array_rep[('atom_neighbors', degree)].astype('int32')
        if len(atom_neighbors_list)==0:
                true_summed_degree = np.zeros((atom_features.shape[0], 10),'float32')
        else:
                true_summed_degree = bond_features_by_degree(atom_features.shape[0],summed_degrees,degree) 
        all_bond_features.append(true_summed_degree) 
    single_dict['atom_match_matrix'] = atom_to_motif_match(array_rep['rdkit_ix'],cliques)
    single_dict['sum_atoms'] = np.reshape(np.sum(single_dict['atom_match_matrix'],axis=1),(-1,1))
    out_bond_features = 0
    for arr in all_bond_features:
        out_bond_features = out_bond_features + arr
    single_dict['input_atom_features'] = np.concatenate([atom_features,out_bond_features],axis=1) 
    adj_matrix,dist_matrix = get_adj_dist_matrix(array_rep,smi)
    single_dict['adj_matrix'] = adj_matrix
    single_dict['dist_matrix'] = dist_matrix 
    single_dict = {key:np.array(value,dtype='float32') for key,value in single_dict.items()} 
    return single_dict

class Graph_Bert_Dataset_fine_tune(object):
    def __init__(self,train_set,val_set,tst_set,tokenizer,batch_size,map_dict,label_field='DDI'):
        
        self.train_set  = train_set
        self.val_set = val_set
        self.tst_set = tst_set
        self.train_set[label_field] = self.train_set[label_field].map(int)
        self.val_set[label_field] = self.val_set[label_field].map(int)
        self.tst_set[label_field] = self.tst_set[label_field].map(int)
        self.label_field = label_field
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.pad_value = self.tokenizer.vocab['<pad>'] 
        self.map_dict = map_dict

    def get_data(self):

        self.dataset1 = tf.data.Dataset.from_tensor_slices((self.train_set['drug_A'], self.train_set['drug_B'],self.train_set[self.label_field]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_seq).padded_batch(self.batch_size, padding_values= (tf.constant(0,dtype = tf.int64),tf.constant(0,dtype = tf.int64),\
            tf.constant(0,dtype = tf.float32),tf.constant(0,dtype = tf.float32),tf.constant(-1e9,dtype = tf.float32),\
                tf.constant(-1e9,dtype = tf.float32),tf.constant(0,dtype = tf.float32),tf.constant(0,dtype = tf.float32),\
                    tf.constant(0,dtype = tf.float32),tf.constant(0,dtype = tf.float32),tf.constant(-1e9,dtype = tf.float32),\
                tf.constant(-1e9,dtype = tf.float32),tf.constant(0,dtype = tf.float32),tf.constant(0,dtype = tf.float32),\
                    tf.constant(1,dtype = tf.float32),tf.constant(1,dtype = tf.float32),tf.constant(0,dtype = tf.int64)),
        padded_shapes=(
            tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),\
                tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),\
                tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),\
                tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([1]))).prefetch(20)
        self.dataset2 = tf.data.Dataset.from_tensor_slices((self.val_set['drug_A'], self.val_set['drug_B'],self.val_set[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_seq).padded_batch(self.batch_size, padding_values= (tf.constant(0,dtype = tf.int64),tf.constant(0,dtype = tf.int64),\
            tf.constant(0,dtype = tf.float32),tf.constant(0,dtype = tf.float32),tf.constant(-1e9,dtype = tf.float32),\
                tf.constant(-1e9,dtype = tf.float32),tf.constant(0,dtype = tf.float32),tf.constant(0,dtype = tf.float32),\
                    tf.constant(0,dtype = tf.float32),tf.constant(0,dtype = tf.float32),tf.constant(-1e9,dtype = tf.float32),\
                tf.constant(-1e9,dtype = tf.float32),tf.constant(0,dtype = tf.float32),tf.constant(0,dtype = tf.float32),\
                    tf.constant(1,dtype = tf.float32),tf.constant(1,dtype = tf.float32),tf.constant(0,dtype = tf.int64)),
        padded_shapes=(
            tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),\
                tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),\
                tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),\
                tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([1]))).prefetch(20)
        self.dataset3 = tf.data.Dataset.from_tensor_slices((self.tst_set['drug_A'], self.tst_set['drug_B'],self.tst_set[self.label_field]))
        self.dataset3 = self.dataset3.map(self.tf_numerical_seq).padded_batch(self.batch_size, padding_values= (tf.constant(0,dtype = tf.int64),tf.constant(0,dtype = tf.int64),\
            tf.constant(0,dtype = tf.float32),tf.constant(0,dtype = tf.float32),tf.constant(-1e9,dtype = tf.float32),\
                tf.constant(-1e9,dtype = tf.float32),tf.constant(0,dtype = tf.float32),tf.constant(0,dtype = tf.float32),\
                    tf.constant(0,dtype = tf.float32),tf.constant(0,dtype = tf.float32),tf.constant(-1e9,dtype = tf.float32),\
                tf.constant(-1e9,dtype = tf.float32),tf.constant(0,dtype = tf.float32),tf.constant(0,dtype = tf.float32),\
                    tf.constant(1,dtype = tf.float32),tf.constant(1,dtype = tf.float32),tf.constant(0,dtype = tf.int64)),
        padded_shapes=(
            tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),\
                tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),\
                tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),\
                tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([None,None]),tf.TensorShape([1]))).prefetch(20)
        return self.dataset1, self.dataset2,self.dataset3

    def numerical_seq(self,smiles1,smiles2,labels):
        smiles1 = smiles1.numpy().decode()
        smiles2 = smiles2.numpy().decode()
        ### Process molecule 1
        nums_list1 = self.map_dict[smiles1]['nums_list'] 
        dist_matrix1 = self.map_dict[smiles1]['dist_matrix'] 
        adjoin_matrix1 = self.map_dict[smiles1]['adj_matrix']
        single_dict1_atom = self.map_dict[smiles1]['single_dict']
        nums_list1 = [self.tokenizer.vocab['<global>']] + nums_list1
        temp1 = np.ones((len(nums_list1),len(nums_list1)))
        temp1[1:,1:] = adjoin_matrix1
        adjoin_matrix1 = (1 - temp1) * (-1e9)
        temp1_dist = np.ones((len(nums_list1),len(nums_list1)))
        temp1_dist[0][0] = 0
        temp1_dist[1:,1:] = dist_matrix1
        dist_matrix1 = temp1_dist
        ### drug1 atom_level
        atom_features1 = single_dict1_atom['input_atom_features']
        dist_matrix1_atom = single_dict1_atom['dist_matrix']
        adjoin_matrix1_atom = single_dict1_atom['adj_matrix']
        adjoin_matrix1_atom = (1 - adjoin_matrix1_atom) * (-1e9) 
        atom_match_matrix1 = single_dict1_atom['atom_match_matrix']
        sum_atoms1 = single_dict1_atom['sum_atoms']
        ### Process molecule 2
        nums_list2 = self.map_dict[smiles2]['nums_list'] 
        dist_matrix2 = self.map_dict[smiles2]['dist_matrix'] 
        adjoin_matrix2 = self.map_dict[smiles2]['adj_matrix'] 
        single_dict2_atom = self.map_dict[smiles2]['single_dict']
        nums_list2 = [self.tokenizer.vocab['<global>']] + nums_list2 
        temp2 = np.ones((len(nums_list2),len(nums_list2)))
        temp2[1:,1:] = adjoin_matrix2
        adjoin_matrix2 = (1 - temp2) * (-1e9)
        temp2_dist = np.ones((len(nums_list2),len(nums_list2)))
        temp2_dist[0][0] = 0
        temp2_dist[1:,1:] = dist_matrix2
        dist_matrix2 = temp2_dist
        ### drug2 atom_level
        atom_features2 = single_dict2_atom['input_atom_features']
        dist_matrix2_atom = single_dict2_atom['dist_matrix']
        adjoin_matrix2_atom = single_dict2_atom['adj_matrix']
        adjoin_matrix2_atom = (1 - adjoin_matrix2_atom) * (-1e9) 
        atom_match_matrix2 = single_dict2_atom['atom_match_matrix']
        sum_atoms2 = single_dict2_atom['sum_atoms']

        y = np.array([labels]).astype('int64')
        x1 = np.array(nums_list1).astype('int64')
        x2 = np.array(nums_list2).astype('int64')
        return x1,x2,adjoin_matrix1,adjoin_matrix2,dist_matrix1,dist_matrix2,\
            atom_features1,atom_features2,adjoin_matrix1_atom,adjoin_matrix2_atom,dist_matrix1_atom,\
                dist_matrix2_atom,atom_match_matrix1,atom_match_matrix2,sum_atoms1,sum_atoms2,y 

    def tf_numerical_seq(self,smiles1,smiles2,labels):

        x1,x2,adjoin_matrix1,adjoin_matrix2,dist_matrix1,dist_matrix2,\
            atom_features1,atom_features2,adjoin_matrix1_atom,adjoin_matrix2_atom,dist_matrix1_atom,\
                dist_matrix2_atom,atom_match_matrix1,atom_match_matrix2,sum_atoms1,sum_atoms2,y = tf.py_function(self.numerical_seq, [smiles1,smiles2,labels],
                                                    [tf.int64, tf.int64,tf.float32,tf.float32,\
                                                        tf.float32,tf.float32,tf.float32,tf.float32,\
                                                        tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,\
                                                        tf.float32,tf.float32,tf.float32,tf.int64])
        x1.set_shape([None])
        x2.set_shape([None])
        adjoin_matrix1.set_shape([None,None])
        adjoin_matrix2.set_shape([None,None])
        dist_matrix1.set_shape([None,None])
        dist_matrix2.set_shape([None,None])
        atom_features1.set_shape([None,None])
        atom_features2.set_shape([None,None])
        adjoin_matrix1_atom.set_shape([None,None])
        adjoin_matrix2_atom.set_shape([None,None])
        dist_matrix1_atom.set_shape([None,None])
        dist_matrix2_atom.set_shape([None,None])
        atom_match_matrix1.set_shape([None,None])
        atom_match_matrix2.set_shape([None,None])
        sum_atoms1.set_shape([None,None])
        sum_atoms2.set_shape([None,None])
        y.set_shape([None])
        return x1,x2,adjoin_matrix1,adjoin_matrix2,dist_matrix1,dist_matrix2,\
            atom_features1,atom_features2,adjoin_matrix1_atom,adjoin_matrix2_atom,dist_matrix1_atom,\
                dist_matrix2_atom,atom_match_matrix1,atom_match_matrix2,sum_atoms1,sum_atoms2,y