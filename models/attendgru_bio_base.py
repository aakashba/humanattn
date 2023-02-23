from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, GRU, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, RepeatVector, Permute, TimeDistributed, dot, Lambda
from tensorflow.keras.optimizers import RMSprop, Adamax
import tensorflow.keras
import tensorflow.keras.utils
import tensorflow as tf
from tensorflow.keras import metrics
import tensorflow.keras.backend as K

class AttentionGRUBioBaseModel:
    def __init__(self, config):
        
        config['tdatlen'] = 400
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.datlen = config['tdatlen']
        self.config['maxastnodes'] = self.datlen
        self.comlen = config['comlen']
        
        self.embdims = 100
        self.recdims = 100

        self.config['batch_config'] = [ ['smlnode', 'com'], ['comout'] ]

    def create_model(self):
        
        dat_input = Input(shape=(self.datlen,))
        bio_input = Input(shape=(self.datlen,))
        com_input = Input(shape=(self.comlen,))
        
        ee = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)(dat_input)
        enc = GRU(self.recdims, return_state=True, return_sequences=True)
        encout, state_h = enc(ee)

        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(com_input)
        dec = GRU(self.recdims, return_sequences=True)
        decout = dec(de, initial_state=state_h)

        attn = dot([decout, encout], axes=[2, 2])
        attn = Activation('softmax')(attn)
        context = dot([attn, encout], axes=[2,1])

        context = concatenate([context, decout])
        
        out = TimeDistributed(Dense(self.recdims, activation="tanh"))(context)

        out = Flatten()(out)
        out = concatenate([out, state_h])
        out = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[dat_input, com_input], outputs=out)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.config, model
