from tensorflow.keras.models import Model
from custom.graphlayer import GCNLayer
from tensorflow.keras.layers import Input, Maximum, Dense, Embedding, Reshape, GRU, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, RepeatVector, Permute, TimeDistributed, dot
from tensorflow.keras.backend import tile, repeat, repeat_elements
from tensorflow.keras.optimizers import RMSprop, Adamax
import keras
import keras.utils
import tensorflow as tf

class AstGnnGRUModel:
    def __init__(self, config):
        
        self.config = config
        
        self.smlvocabsize = config['smlvocabsize']
        self.smllen = config['smllen']
        self.config['maxastnodes'] = 400
        self.embdims = 100
        self.smldims = 100
        self.recdims = 100
        self.findims = 100

        self.config['batch_maker'] = 'gazeout_gnn'
        self.config['num_input'] = 1
        self.config['num_output'] = 1
        
        self.config['asthops'] = 2

    def create_model(self):
        
        sml_input = Input(shape=(self.smllen,))
        edge_input = Input(shape=(self.smllen, self.smllen))
        focalpoint_input = Input(shape=(1,))
        
        se = Embedding(output_dim=self.smldims, input_dim=self.smlvocabsize, mask_zero=False)
        seinp = se(sml_input)
        
        for i in range(self.config['asthops']):
            seinp = GCNLayer(self.smldims)([seinp, edge_input])

        se_enc =GRU(self.recdims, return_state=True, return_sequences=True)
        seout, state_sml = se_enc(seinp)
        
        fp_enc = se(focalpoint_input)
        
        context = concatenate([seout, fp_enc], axis=1)
        
        #context = TimeDistributed(Dense(self.recdims, activation="tanh"))(context)
        
        out = Flatten()(context)
        out = Dense(1)(out)
        
        model = Model(inputs=[sml_input, edge_input, focalpoint_input], outputs=out)

        if self.config['multigpu']:
            model = keras.utils.multi_gpu_model(model, gpus=2)
        
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(loss='mse', optimizer=opt, metrics=['mse'])
        return self.config, model
