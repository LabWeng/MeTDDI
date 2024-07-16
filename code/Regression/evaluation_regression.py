# %%
import numpy as np
import pandas as pd
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import *
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from dataset_processed import Graph_Bert_Dataset_fine_tune
from tensorflow.keras.callbacks import ModelCheckpoint
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# %%
from model_motif_level import *
from model_atom_level import * 

# %%
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# %%
def input_solver1(sample,sample1,sample2,sample3,sample4,sample5,\
    sample6,sample7,sample8,sample9,sample10,sample11,sample12,sample13,sample14,sample15,sample16):
    return {'molecule_sequence1': sample,'molecule_sequence2': sample1, 'adj_matrix1': sample2,
           'adj_matrix2': sample3,'dist_matrix1': sample4,'dist_matrix2': sample5,
           'atom_features1':sample6,'atom_features2':sample7,'adjoin_matrix1_atom':sample8,
           'adjoin_matrix2_atom':sample9,'dist_matrix1_atom':sample10,'dist_matrix2_atom':sample11,
           'atom_match_matrix1':sample12,'atom_match_matrix2':sample13,'sum_atoms1':sample14,'sum_atoms2':sample15}, sample16

# %%
tr_dataset = pd.read_csv('/data/Regression/tr_datset_regression_fold1.csv') ## Fold one
val_dataset = pd.read_csv('/data/Regression/val_datset_regression_fold1.csv')
tst_dataset = pd.read_csv('/data/Regression/PKDDI_info_2023_FDA_external.csv')

# %%
## Transform AUC FC to log2
tr_dataset['AUC FC'] = tr_dataset['AUC FC'].apply(lambda x:np.log2(x))
val_dataset['AUC FC'] = val_dataset['AUC FC'].apply(lambda x:np.log2(x)) 
tst_dataset['AUC FC'] = tst_dataset['AUC FC'].apply(lambda x:np.log2(x)) 

# %%
tokenizer = Mol_Tokenizer('/code/Regression/token_id.json')

train_dataset_,validation_dataset, test_dataset_ = Graph_Bert_Dataset_fine_tune(tr_dataset,val_dataset,tst_dataset,label_field='AUC FC',tokenizer=tokenizer,batch_size = 128).get_data()
train_dataset = train_dataset_.map(input_solver1)
validation_dataset = validation_dataset.map(input_solver1) 
test_dataset = test_dataset_.map(input_solver1)

# %%
### param setting
small = {'name': 'Small', 'num_layers': 2, 'num_heads': 4, 'd_model': 256} 
arch = small  
num_layers = arch['num_layers']
num_heads =  arch['num_heads']
d_model =  arch['d_model']
dff = d_model 
input_vocab_size = tokenizer.get_vocab_size
dropout_rate = 0.1
training = False

# %%
# motif inputs
motif_inputs = Input(shape=(None,), name = "molecule_sequence")
# mask_inputs_motif = create_padding_mask(motif_inputs)
motif_adj_inputs = Input(shape=(None,None), name= "adj_matrix") 
motif_dist_inputs = Input(shape=(None,None), name= "dist_matrix")
# atom_level inputs
atom_inputs = Input(shape=(None,61), name = "atom_features") 
atom_adj_inputs = Input(shape=(None,None), name= "atom_adj_matrix") 
atom_dist_inputs = Input(shape=(None,None), name= "atom_dist_matrix")
atom_match_matrix = Input(shape=(None,None), name= "atom_match_matrix")
sum_atoms = Input(shape=(None,None), name= "sum_atoms")

# %%
# dual inputs setting
### motif-level inputs
motif_inputs1 = Input(shape=(None,), name= "molecule_sequence1") 
motif_inputs2 = Input(shape=(None,), name= "molecule_sequence2")
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
# build atom_level model
Outseq_atom,*_,encoder_padding_mask_atom = EncoderModel_atom(num_layers=2,d_model=128,dff=dff,num_heads=4)(atom_inputs,adjoin_matrix = atom_adj_inputs,\
    dist_matrix = atom_dist_inputs,atom_match_matrix = atom_match_matrix,sum_atoms = sum_atoms,training = training) 
Outseq_motif,*_,encoder_padding_mask_motif = EncoderModel_motif(num_layers=num_layers,d_model=d_model,dff=dff,num_heads=num_heads,input_vocab_size=input_vocab_size)(motif_inputs,adjoin_matrix = motif_adj_inputs,\
dist_matrix = motif_dist_inputs,atom_level_features=Outseq_atom,training = training)
# build motif_level model
model_motif = Model(inputs=[atom_inputs,atom_adj_inputs,atom_dist_inputs,
        atom_match_matrix,sum_atoms,motif_inputs,motif_adj_inputs,motif_dist_inputs], outputs=[Outseq_motif,encoder_padding_mask_motif,Outseq_atom])

# %%
# weight sharing 
druga_trans,encoder_padding_mask_a,Outseq_atoma = model_motif([atom_inputs1,atom_adj_inputs1,atom_dist_inputs1\
,atom_match_matrix1,sum_atoms1,motif_inputs1,motif_adj_inputs1,motif_dist_inputs1]) 
drugb_trans,encoder_padding_mask_b,Outseq_atomb = model_motif([atom_inputs2,atom_adj_inputs2,atom_dist_inputs2\
    ,atom_match_matrix2,sum_atoms2,motif_inputs2,motif_adj_inputs2,motif_dist_inputs2]) 
Co_attention_layers = Co_Attention_Layer(d_model,k = 128,name = 'Co_attention_layer')
fc1 = tf.keras.layers.Dense(d_model/2, activation='relu') 
dropout1 = tf.keras.layers.Dropout(0.1) 
fc2 = tf.keras.layers.Dense(d_model/4, activation='relu') 
dropout2 = tf.keras.layers.Dropout(0.1) 
fc3 = tf.keras.layers.Dense(1)

# %%
### To avoid high similarity scores
Wa = tf.keras.layers.Dense(d_model) 
Wb = tf.keras.layers.Dense(d_model)

# %%
## Co-attention and inference
druga_trans_,drugb_trans_,*_ = Co_attention_layers([Wa(druga_trans),Wb(drugb_trans)])
output1_2 = tf.keras.layers.Concatenate()([druga_trans_,drugb_trans_])
output1_2 = fc1(output1_2)
output1_2 = dropout1(output1_2,training=training) 
output1_2 = fc2(output1_2)
output1_2 = dropout2(output1_2,training=training) 
output1_2 = fc3(output1_2)

# %%
# Build regression model
models = Model(inputs=[atom_inputs1,atom_adj_inputs1,atom_dist_inputs1\
,atom_match_matrix1,sum_atoms1,motif_inputs1,motif_adj_inputs1,motif_dist_inputs1,
atom_inputs2,atom_adj_inputs2,atom_dist_inputs2\
,atom_match_matrix2,sum_atoms2,motif_inputs2,motif_adj_inputs2,motif_dist_inputs2],outputs =[output1_2])

# %%
models.load_weights('/code/Regression/saved_weights/regression_fold.h5') 

# %%
### evaluation 
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr 

# %%
predicts = models.predict(validation_dataset,verbose = 0) 
predicts = predicts.sum(axis=1) 

# %%
print('Evaluation results of validation set:','RMSE:',round(mean_squared_error(val_dataset['AUC FC'],predicts,squared=False),3),\
      'Pearsonr:',round(pearsonr(val_dataset['AUC FC'],predicts)[0],3)) 

# %%
predicts = models.predict(test_dataset,verbose = 0) 
predicts = predicts.sum(axis=1) 

# %%
print('Evaluation results of external data from FDA:','RMSE:',round(mean_squared_error(tst_dataset['AUC FC'],predicts,squared=False),3),\
      'Pearsonr:',round(pearsonr(tst_dataset['AUC FC'],predicts)[0],3))
