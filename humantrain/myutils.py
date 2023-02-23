import sys
import javalang
from timeit import default_timer as timer
import tensorflow.keras
import numpy as np
import tensorflow as tf
import networkx as nx
import random

# do NOT import keras in this header area, it will break predict.py
# instead, import keras as needed in each function

# TODO refactor this so it imports in the necessary functions
dataprep = '/nfs/projects/funcom_bio/data/eyesum'
sys.path.append(dataprep)
import tokenizer

start = 0
end = 0

def prep(msg):
    global start
    statusout(msg)
    start = timer()

def statusout(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()

def drop():
    global start
    global end
    end = timer()
    sys.stdout.write('done, %s seconds.\n' % (round(end - start, 2)))
    sys.stdout.flush()

def index2word(tok):
	i2w = {}
	for word, index in tok.w2i.items():
		i2w[index] = word

	return i2w

def seq2sent(seq, tokenizer):
    sent = []
    check = index2word(tokenizer)
    for i in seq:
        sent.append(check[i])

    return(' '.join(sent))
            
class batch_gen(tensorflow.keras.utils.Sequence):
    def __init__(self, dataset, condat, config, training=True):
        self.batch_size = config['batch_size']
        self.dataset = dataset
        self.condat = condat
        self.allsids = list(dataset.keys())
        self.num_inputs = config['num_input']
        self.config = config
        self.training = training
        
    def __getitem__(self, idx):
        start = (idx*self.batch_size)
        end = self.batch_size*(idx+1)
        batchsids = self.allsids[start:end]
        return self.make_batch(batchsids)

    def make_batch(self, batchsids):
        if self.config['batch_maker'] == 'gazeout':
            return self.divideseqs_gazeout(batchsids)
        elif self.config['batch_maker'] == 'gazeout_gnn':
            return self.divideseqs_gazeout_gnn(batchsids)
        else:
            return None

    def __len__(self):
        return int(np.ceil(len(self.allsids)/self.batch_size))

    def on_epoch_end(self):
        random.shuffle(self.allsids)

    def divideseqs_gazeout(self, batchsids):
        import tensorflow.keras.utils
        
        smlseqs = list()
        focalpoints = list()
        comouts = list()
        
        fiddat = dict()

        for sid in batchsids:

            #  fid, subject, whole_token_in_original_code, word_position, gaze_time_on_word, subject_total_gaze_time
            (fid, sub, wword, wp, gtms, stgt) = self.dataset[sid]
            
            wsmlseq = self.condat['nodes'][fid] # whole function for context
            wfp = self.condat['nodes'][fid][wp] # single word being viewed

            #print(sid, self.dataset[sid], len(wsmlseq))

            # crop to maximum size
            wsmlseq = wsmlseq[:self.config['smllen']]
            
            # pad to minimum size
            for i in range(0, self.config['smllen'] - len(wsmlseq)):
                wsmlseq = np.append(wsmlseq, [0])

            if not self.training:
                fiddat[sid] = [wsmlseq, wfp]
            else:
                smlseqs.append(wsmlseq)
                focalpoints.append(wfp)
                #comout = gtms
                comout = round(gtms / stgt, 5) # percent of time reading word
                comouts.append(np.asarray(comout))

        smlseqs = np.asarray(smlseqs)
        focalpoints = np.asarray(focalpoints)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            return ([smlseqs, focalpoints], comouts)

    def divideseqs_gazeout_gnn(self, batchsids):
        import tensorflow.keras.utils
        
        smlseqs = list()
        smledges = list()
        focalpoints = list()
        comouts = list()
        
        fiddat = dict()

        for sid in batchsids:

            #  fid, subject, whole_token_in_original_code, word_position, gaze_time_on_word, subject_total_gaze_time
            (fid, sub, wword, wp, gtms, stgt) = self.dataset[sid]
            
            wsmlseq = self.condat['nodes'][fid] # whole function for context
            wsmledges = self.condat['edges'][fid] # AST edges
            wfp = self.condat['nodes'][fid][wp] # single word being viewed

            #print(sid, self.dataset[sid], len(wsmlseq))

            # crop to maximum size
            wsmlseq = wsmlseq[:self.config['smllen']]
            
            # pad to minimum size
            for i in range(0, self.config['smllen'] - len(wsmlseq)):
                wsmlseq = np.append(wsmlseq, [0])

            # crop/expand ast adjacency matrix to dense
            wsmledges = np.asarray(wsmledges.todense())
            wsmledges = wsmledges[:self.config['smllen'], :self.config['smllen']]
            tmp_1 = np.zeros((self.config['smllen'], self.config['smllen']), dtype='int32')
            tmp_1[:wsmledges.shape[0], :wsmledges.shape[1]] = wsmledges
            wsmledges = np.int32(tmp_1)

            if not self.training:
                fiddat[sid] = [wsmlseq, wsmledges, wfp]
            else:
                smlseqs.append(wsmlseq)
                smledges.append(wsmledges)
                focalpoints.append(wfp)
                #comout = gtms
                comout = round(gtms / stgt, 5) # percent of time reading word
                comouts.append(np.asarray(comout))

        smlseqs = np.asarray(smlseqs)
        smledges = np.asarray(smledges)
        focalpoints = np.asarray(focalpoints)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            return ([smlseqs, smledges, focalpoints], comouts)
