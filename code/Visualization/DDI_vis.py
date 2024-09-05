import numpy as np
from utils import *
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from dataset_inference import Inference_Dataset
from model_motif_level import *
from model_atom_level import *
from IPython.display import SVG, display
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tokenizer = Mol_Tokenizer('token_id.json')
small = {'name': 'Small', 'num_layers': 4, 'num_heads': 8, 'd_model': 256}
arch = small
num_layers = arch['num_layers']
num_heads =  arch['num_heads']
d_model =  arch['d_model']*2 
dff = d_model 
input_vocab_size = tokenizer.get_vocab_size 
dropout_rate = 0.1
#### motif inputs
motif_inputs = Input(shape=(None,), name = "molecule_sequence")
# mask_inputs_motif = create_padding_mask(motif_inputs)
motif_adj_inputs = Input(shape=(None,None), name= "adj_matrix") 
motif_dist_inputs = Input(shape=(None,None), name= "dist_matrix")
#### atom_level inputs
atom_inputs = Input(shape=(None,61), name = "atom_features")
atom_adj_inputs = Input(shape=(None,None), name= "atom_adj_matrix") 
atom_dist_inputs = Input(shape=(None,None), name= "atom_dist_matrix")
atom_match_matrix = Input(shape=(None,None), name= "atom_match_matrix")
sum_atoms = Input(shape=(None,None), name= "sum_atoms")
training = False
class DDI_Interpretation(object):
    def __init__(self,weight_path = r'weights.02-0.18.h5'):
        ### build atom_level model
        Outseq_atom,*_,encoder_padding_mask_atom = EncoderModel_atom(num_layers=2,d_model=256,dff=dff,num_heads=4)(atom_inputs,adjoin_matrix = atom_adj_inputs,\
            dist_matrix = atom_dist_inputs,atom_match_matrix = atom_match_matrix,sum_atoms = sum_atoms,training = training) 
        Outseq_motif,*_,encoder_padding_mask_motif = EncoderModel_motif(num_layers=num_layers,d_model=d_model,dff=dff,num_heads=num_heads,input_vocab_size=input_vocab_size)(motif_inputs,adjoin_matrix = motif_adj_inputs,\
        dist_matrix = motif_dist_inputs,atom_level_features=Outseq_atom,training = training)
        model_motif = Model(inputs=[atom_inputs,atom_adj_inputs,atom_dist_inputs,
                    atom_match_matrix,sum_atoms,motif_inputs,motif_adj_inputs,motif_dist_inputs], outputs=[Outseq_motif,encoder_padding_mask_motif,Outseq_atom])
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
        druga_trans,encoder_padding_mask_a,Outseq_atoma = model_motif([atom_inputs1,atom_adj_inputs1,atom_dist_inputs1\
        ,atom_match_matrix1,sum_atoms1,motif_inputs1,motif_adj_inputs1,motif_dist_inputs1]) 
        drugb_trans,encoder_padding_mask_b,Outseq_atomb = model_motif([atom_inputs2,atom_adj_inputs2,atom_dist_inputs2\
            ,atom_match_matrix2,sum_atoms2,motif_inputs2,motif_adj_inputs2,motif_dist_inputs2]) 
        Co_attention_layers = Co_Attention_Layer(d_model,k = 128,name = 'Co_attention_layer')
        fc1 = tf.keras.layers.Dense(d_model/2, activation='relu') 
        dropout1 = tf.keras.layers.Dropout(0.1) 
        fc2 = tf.keras.layers.Dense(d_model/4, activation='relu') 
        dropout2 = tf.keras.layers.Dropout(0.1) 
        fc3 = tf.keras.layers.Dense(4,activation='softmax')
        ### To avoid high similarity scores
        Wa = tf.keras.layers.Dense(d_model) 
        Wb = tf.keras.layers.Dense(d_model)
        druga_trans_,drugb_trans_,*_ = Co_attention_layers([Wa(druga_trans),Wb(drugb_trans)])
        output1_2 = tf.keras.layers.Concatenate()([druga_trans_,drugb_trans_])
        output1_2 = fc1(output1_2)
        output1_2 = dropout1(output1_2,training=training) 
        output1_2 = fc2(output1_2)
        output1_2 = dropout2(output1_2,training=training) 
        output1_2 = fc3(output1_2)
        self.models = Model(inputs=[atom_inputs1,atom_adj_inputs1,atom_dist_inputs1\
            ,atom_match_matrix1,sum_atoms1,motif_inputs1,motif_adj_inputs1,motif_dist_inputs1,
            atom_inputs2,atom_adj_inputs2,atom_dist_inputs2\
            ,atom_match_matrix2,sum_atoms2,motif_inputs2,motif_adj_inputs2,motif_dist_inputs2],outputs =[output1_2])
        self.models.load_weights(weight_path)
        self.attention_weight_output_model = Model(inputs=self.models.input,outputs=self.models.get_layer('Co_attention_layer').output)
        self.weights_dict_,self.predict_prob_ = '',''
        self.pair_dict = ''
        self.all_trans_attention = ''
        self.max_weights_atom_ids = ''
    
    def input_solver_predict(self,sample,sample1,sample2,sample3,sample4,sample5,\
            sample6,sample7,sample8,sample9,sample10,sample11,sample12,sample13,sample14,sample15):
        return {'molecule_sequence1': sample,'molecule_sequence2': sample1, 'adj_matrix1': sample2,
            'adj_matrix2': sample3,'dist_matrix1': sample4,'dist_matrix2': sample5,
            'atom_features1':sample6,'atom_features2':sample7,'adjoin_matrix1_atom':sample8,
            'adjoin_matrix2_atom':sample9,'dist_matrix1_atom':sample10,'dist_matrix2_atom':sample11,
            'atom_match_matrix1':sample12,'atom_match_matrix2':sample13,'sum_atoms1':sample14,'sum_atoms2':sample15}
    def predict(self,pair_dict):
        inference_dataset = Inference_Dataset(pair_dict,tokenizer).get_data()
        inference_dataset = inference_dataset.map(self.input_solver_predict)
        *_,weight_A,weight_B = self.attention_weight_output_model.predict(inference_dataset)
        predict_prob = self.models.predict(inference_dataset)
        weights_dict = {'drug_A':weight_A,'drug_B':weight_B}
        self.weights_dict_,self.predict_prob_ = weights_dict,predict_prob
        self.pair_dict = pair_dict
    def visualize(self,smiles,weights,img_size = 250,if_vis = True,topK=1,svg = False,if_atom_weights = False):
        if self.pair_dict == '':
            raise 'Please use the predict method first'
        mol = Chem.MolFromSmiles(smiles) 
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol)) 
        index = tokenizer.tokenize(smiles) 
        atom_weights = self.motif_to_atom_weights(mol,weights,index,topK=topK) 
        if if_vis:
            if svg: 
                drawer = rdMolDraw2D.MolDraw2DSVG(img_size, img_size)
                atom_weights = [float(i) for i in atom_weights]
                drawer = MapAtomFromWeights(mol,atom_weights,drawer)
                drawer.FinishDrawing()
                svg = drawer.GetDrawingText() 
                svg = svg.replace('svg:','')
                return SVG(svg)
            else:
                drawer = Chem.Draw.MolDraw2DCairo(img_size, img_size)
                atom_weights = [float(i) for i in atom_weights] 
                fig = SimilarityMaps.GetSimilarityMapFromWeights(
                mol,
                atom_weights,
                contourLines=0,alpha = 0.01,
                draw2d=drawer)   
                drawer.FinishDrawing() 
                img = self.show_cairo(drawer.GetDrawingText())
                return img 
        if if_atom_weights:
            return atom_weights
    def motif_to_atom_weights(self,mol,weights,index,topK=1):
        def get_topkweight_id(weights, topK): 
            # weights = weights#[0][0]
            if topK==1:
                return np.where(weights==weights.max())[0]
            topk_score = sorted(weights,reverse=True)[topK]
            return np.where(weights>topk_score)[0] 
        temp_atom_weights = [0 for i in range(mol.GetNumAtoms())]
        max_weight_motif = get_topkweight_id(weights,topK=topK) 
        max_weights_atom_ids = [] 
        id_atom_map = index[2] 
        assert len(index[0])==len(index[2]) 
        for th,i in enumerate(index[0]):
            token_atom_list = id_atom_map[th] 
            if th in max_weight_motif: 
                max_weights_atom_ids.append([i,token_atom_list])
                token_weights = weights[th]
            else:
                token_weights = weights[th]
            for j in token_atom_list:
                if token_weights >= temp_atom_weights[j]:
                    temp_atom_weights[j] = token_weights
        self.max_weights_atom_ids = max_weights_atom_ids 
        return np.array(temp_atom_weights).astype(np.float32).tolist()
    def show_cairo(self,data):
        import io
        from PIL import Image
        bio = io.BytesIO(data)
        img = Image.open(bio)
        return img
