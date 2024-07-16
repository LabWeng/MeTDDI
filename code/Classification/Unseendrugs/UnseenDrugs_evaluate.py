# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from utils import *
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from dataset_processed import Graph_Bert_Dataset_fine_tune
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# %%
from model_motif_level import * 
from model_atom_level import *


# %%
def input_solver1(sample,sample1,sample2,sample3,sample4,sample5,\
    sample6,sample7,sample8,sample9,sample10,sample11,sample12,sample13,sample14,sample15,sample16):
    return {'molecule_sequence1': sample,'molecule_sequence2': sample1, 'adj_matrix1': sample2,
           'adj_matrix2': sample3,'dist_matrix1': sample4,'dist_matrix2': sample5,
           'atom_features1':sample6,'atom_features2':sample7,'adjoin_matrix1_atom':sample8,
           'adjoin_matrix2_atom':sample9,'dist_matrix1_atom':sample10,'dist_matrix2_atom':sample11,
           'atom_match_matrix1':sample12,'atom_match_matrix2':sample13,'sum_atoms1':sample14,'sum_atoms2':sample15}, sample16

# %%
dataFolder = '/data/Classification/Unseendrugs'
tr_dataset = pd.read_csv(dataFolder + '/tr_dataset.csv')
val_unseenone = pd.read_csv(dataFolder + '/val_dataset_unseen_onedrug.csv') 
val_unseentwo = pd.read_csv(dataFolder + '/val_dataset_unseen_twodrugs.csv')

# %%
tokenizer = Mol_Tokenizer('/code/Classification/Unseendrugs/token_id.json')
map_dict = np.load('/code/Classification/Unseendrugs/preprocessed_drug_info.npy',allow_pickle=True).item()

# %%
train_dataset,validation_unseenone, validation_unseentwo = Graph_Bert_Dataset_fine_tune(tr_dataset,val_unseenone,val_unseentwo,label_field='DDI',tokenizer=tokenizer,map_dict=map_dict,batch_size = 64).get_data()
train_dataset = train_dataset.map(input_solver1)
validation_unseenone = validation_unseenone.map(input_solver1) 
validation_unseentwo = validation_unseentwo.map(input_solver1)

# %%
param = {'name': 'Small', 'num_layers': 4, 'num_heads': 8, 'd_model': 256}
arch = param   ## small 3 4 128   medium: 6 6  256     large:  12 8 516
num_layers = arch['num_layers']
num_heads =  arch['num_heads']
d_model =  arch['d_model']*2 
dff = d_model 
input_vocab_size = tokenizer.get_vocab_size
dropout_rate = 0.1
training = False 

# %%
## motif_level inputs
motif_inputs = Input(shape=(None,), name = "molecule_sequence")
motif_adj_inputs = Input(shape=(None,None), name= "adj_matrix") 
motif_dist_inputs = Input(shape=(None,None), name= "dist_matrix")
### atom_level inputs
atom_inputs = Input(shape=(None,61), name = "atom_features")
atom_adj_inputs = Input(shape=(None,None), name= "atom_adj_matrix") 
atom_dist_inputs = Input(shape=(None,None), name= "atom_dist_matrix")
atom_match_matrix = Input(shape=(None,None), name= "atom_match_matrix")
sum_atoms = Input(shape=(None,None), name= "sum_atoms")

# %%
## build atom_level model
Outseq_atom,*_,encoder_padding_mask_atom = EncoderModel_atom(num_layers = 2,d_model = arch['d_model'],dff = dff,\
                                                             num_heads = num_heads)(atom_inputs,adjoin_matrix = atom_adj_inputs,\
    dist_matrix = atom_dist_inputs,atom_match_matrix = atom_match_matrix,sum_atoms = sum_atoms,training = training) 
## build motif_level model
Outseq_motif,*_,encoder_padding_mask_motif = EncoderModel_motif(num_layers = num_layers,d_model = d_model,dff = dff,\
                                            num_heads = num_heads,input_vocab_size = input_vocab_size)(motif_inputs,adjoin_matrix = motif_adj_inputs,\
    dist_matrix = motif_dist_inputs,atom_level_features = Outseq_atom,training = training)
model_motif = Model(inputs = [atom_inputs,atom_adj_inputs,atom_dist_inputs,
            atom_match_matrix,sum_atoms,motif_inputs,motif_adj_inputs,motif_dist_inputs], outputs = [Outseq_motif,encoder_padding_mask_motif,Outseq_atom])

# %%
### Build dual inputs
### motif-level inputs
motif_inputs1 = Input(shape=(None,), name= "molecule_sequence1")
motif_inputs2 = Input(shape=(None,), name= "molecule_sequence2")
# mask_inputs1 = create_padding_mask(inputs1)
# mask_inputs2 = create_padding_mask(inputs2)
motif_adj_inputs1 = Input(shape=(None,None), name= "adj_matrix1") 
motif_adj_inputs2 = Input(shape=(None,None), name= "adj_matrix2")
motif_dist_inputs1 = Input(shape=(None,None), name= "dist_matrix1")
motif_dist_inputs2 = Input(shape=(None,None), name= "dist_matrix2")
### atom level inputs
atom_inputs1 = Input(shape=(None,61), name = "atom_features1")
atom_inputs2 = Input(shape=(None,61), name = "atom_features2")
atom_adj_inputs1 = Input(shape=(None,None), name= "adjoin_matrix1_atom") 
atom_adj_inputs2 = Input(shape=(None,None), name= "adjoin_matrix2_atom") 
atom_dist_inputs1 = Input(shape=(None,None), name= "dist_matrix1_atom")
atom_dist_inputs2 = Input(shape=(None,None), name= "dist_matrix2_atom")
atom_match_matrix1 = Input(shape=(None,None), name= "atom_match_matrix1")
atom_match_matrix2 = Input(shape=(None,None), name= "atom_match_matrix2")
sum_atoms1 = Input(shape=(None,None), name= "sum_atoms1")
sum_atoms2 = Input(shape=(None,None), name= "sum_atoms2")

# %%
# build weight sharing model
druga_trans,encoder_padding_mask_a,Outseq_atoma = model_motif([atom_inputs1,atom_adj_inputs1,atom_dist_inputs1\
    ,atom_match_matrix1,sum_atoms1,motif_inputs1,motif_adj_inputs1,motif_dist_inputs1]) 
drugb_trans,encoder_padding_mask_b,Outseq_atomb = model_motif([atom_inputs2,atom_adj_inputs2,atom_dist_inputs2\
    ,atom_match_matrix2,sum_atoms2,motif_inputs2,motif_adj_inputs2,motif_dist_inputs2]) 

# %%
# build co-attention layers and Fcls
Co_attention_layers = Co_Attention_Layer(d_model,k = 128,name = 'Co_attention_layer')
fc1 = tf.keras.layers.Dense(d_model/2, activation='relu') 
dropout1 = tf.keras.layers.Dropout(dropout_rate) 
fc2 = tf.keras.layers.Dense(d_model/4, activation='relu') 
dropout2 = tf.keras.layers.Dropout(dropout_rate) 
fc3 = tf.keras.layers.Dense(4,activation='softmax')

# %%
### To avoid high similarity scores
Wa = tf.keras.layers.Dense(d_model) 
Wb = tf.keras.layers.Dense(d_model)

# %%
druga_trans_,drugb_trans_,*_ = Co_attention_layers([Wa(druga_trans),Wb(drugb_trans)])
output1_2 = tf.keras.layers.Concatenate()([druga_trans_,drugb_trans_])
output1_2 = fc1(output1_2)
output1_2 = dropout1(output1_2,training=training) 
output1_2 = fc2(output1_2)
output1_2 = dropout2(output1_2,training=training) 
output1_2 = fc3(output1_2)

# %%
# build MeTDDI
models = Model(inputs=[atom_inputs1,atom_adj_inputs1,atom_dist_inputs1\
    ,atom_match_matrix1,sum_atoms1,motif_inputs1,motif_adj_inputs1,motif_dist_inputs1,
    atom_inputs2,atom_adj_inputs2,atom_dist_inputs2\
    ,atom_match_matrix2,sum_atoms2,motif_inputs2,motif_adj_inputs2,motif_dist_inputs2],outputs =[output1_2])

# %%
### To evaluate unseen one drug

# %%
models.load_weights('/code/Classification/Unseendrugs/saved_weights/Unseen_onedrug.h5') 

# %%
from sklearn import metrics

# %%
def evaluation(preds,truths,average = 'micro',ndigits = 3):
    pred_res_to_labels = np.argmax(preds,axis=1)
    label_to_onehot = np.eye(4)[truths]
    acc = round(metrics.accuracy_score(truths,pred_res_to_labels),ndigits)
    auc = round(metrics.roc_auc_score(label_to_onehot,preds,average = average),ndigits)
    aupr = round(metrics.average_precision_score(label_to_onehot,preds,average = average),ndigits) 
    print('ACC:',acc,'AUROC:',auc,'AUPR:',aupr) 

# %%
pred_res = models.predict(validation_unseenone,verbose=False) 
labels = val_unseenone['DDI'].tolist() 

# %%
print('Unseen onedrug evaluation results:')
evaluation(pred_res,labels)

# %%
### To evaluate unseen two drugs
models.load_weights('/code/Classification/Unseendrugs/saved_weights/Unseen_twodrugs.h5') 

# %%
pred_res = models.predict(validation_unseentwo,verbose=False) 
labels = val_unseentwo['DDI'].tolist() 

# %%
print('Unseen two drugs evaluation results:')
evaluation(pred_res,labels)


