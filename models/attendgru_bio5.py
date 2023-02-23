from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, GRU, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, RepeatVector, Permute, TimeDistributed, dot, Lambda, average
from tensorflow.keras.optimizers import RMSprop, Adamax
import tensorflow.keras
import tensorflow.keras.utils
import tensorflow as tf
from tensorflow.keras import metrics
import tensorflow.keras.backend as K

class AttentionGRUBio5Model:
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

        self.config['batch_config'] = [ ['smlnode', 'bio', 'com'], ['comout'] ]

    def create_model(self):
        
        dat_input = Input(shape=(self.datlen,))
        bio_input = Input(shape=(self.datlen,))
        com_input = Input(shape=(self.comlen,))
        
        ee = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)(dat_input)
        enc = GRU(self.recdims, return_state=True, return_sequences=True)
        encout, state_h = enc(ee)

        #humanattn = tf.tile(bio_input, (1, 100))
        humanattn = bio_input
        #humanattn = Activation('softmax')(humanattn)
        humanattn = Reshape((self.datlen, 1))(humanattn)
        humanattn = Lambda(tensorflow.keras.backend.tile, arguments={'n':(1, 1, 100)})(humanattn)
        #ee_stop_grad = Lambda(lambda x: K.stop_gradient(x))(ee)
        #he = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)(dat_input)
        humancontext = multiply([ee, humanattn])
        humancontext, human_h = GRU(self.recdims, return_state=True, return_sequences=True)(humancontext, initial_state=state_h)
        
        #encout = multiply([encout, humanattn])

        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(com_input)
        dec = GRU(self.recdims, return_sequences=True)
        decout = dec(de, initial_state=state_h)

        humancontext = multiply([humancontext, humanattn])
        #decoutng = Lambda(lambda x: K.stop_gradient(x))(decout)
        
        humanattn = dot([decout, humancontext], axes=[2, 2])
        humanattn = Activation('softmax')(humanattn)
        humancontext = dot([humanattn, humancontext], axes=[2,1])

        #humancontext = concatenate([humancontext, decout])
        #outh = TimeDistributed(Dense(int((self.recdims*2)/2), activation="tanh"))(humancontext)
        #outh = Flatten()(outh)
        #outh = humancontext
        #outh = Dense(self.comvocabsize, activation="softmax")(outh)

        attn = dot([decout, encout], axes=[2, 2])
        attn = Activation('softmax')(attn)
        context = dot([attn, encout], axes=[2,1])

        context = concatenate([context, decout, humancontext])
        #context = concatenate([context, decout])
        
        out = TimeDistributed(Dense(int((self.recdims*3)/2), activation="tanh"))(context)

        out = Flatten()(out)
        #outb = concatenate([out, human_h])#humancontext])
        
        out = Dense(self.comvocabsize, activation="softmax")(out)
        #outb = Dense(self.comvocabsize, activation="softmax")(outb)
        
        #outc = average([out, outb])
        #outc = Lambda(lambda x: K.stop_gradient(x))(outc)
        
        model = Model(inputs=[dat_input, bio_input, com_input], outputs=[out])#, outb, outc])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.config, model
