import pickle
import sys
import os
import math
import traceback
import argparse
import signal
import atexit
import time

import random
import tensorflow as tf
import numpy as np

seed = 1337
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

import tensorflow.keras
import tensorflow.keras.utils
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, Callback
import tensorflow.keras.backend as K
from model import create_model
from myutils import prep, drop, batch_gen, seq2sent
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from scipy.stats import pearsonr

class HistoryCallback(Callback):
    
    def setCatchExit(self, outdir, modeltype, timestart, mdlconfig):
        self.outdir = outdir
        self.modeltype = modeltype
        self.history = {}
        self.timestart = timestart
        self.mdlconfig = mdlconfig
        
        atexit.register(self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)
        signal.signal(signal.SIGINT, self.handle_exit)
    
    def handle_exit(self, *args):
        if len(self.history.keys()) > 0:
            try:
                fn = outdir+'/histories/'+self.modeltype+'_hist_'+str(self.timestart)+'.pkl'
                histoutfd = open(fn, 'wb')
                pickle.dump(self.history, histoutfd)
                print('saved history to: ' + fn)
                
                fn = outdir+'/histories/'+self.modeltype+'_conf_'+str(self.timestart)+'.pkl'
                confoutfd = open(fn, 'wb')
                pickle.dump(self.mdlconfig, confoutfd)
                print('saved config to: ' + fn)
            except Exception as ex:
                print(ex)
                traceback.print_exc(file=sys.stdout)
        sys.exit()
    
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


if __name__ == '__main__':

    timestart = int(round(time.time()))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, help='0 or 1', default='0')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=200)
    parser.add_argument('--trainper', dest='trainper', type=int, default=80)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--model-type', dest='modeltype', type=str, default='vanilla')
    parser.add_argument('--with-multigpu', dest='multigpu', action='store_true', default=False)
    parser.add_argument('--zero-dats', dest='zerodats', type=str, default='no')
    parser.add_argument('--with-no-savemodel', dest='nosavemodel', action='store_true', default=False)
    parser.add_argument('--data', dest='dataprep', type=str, default='/nfs/projects/humanattn/data/eyesum')
    parser.add_argument('--outdir', dest='outdir', type=str, default='outdir')
    parser.add_argument('--dtype', dest='dtype', type=str, default='float32')
    parser.add_argument('--tf-loglevel', dest='tf_loglevel', type=str, default='3')
    parser.add_argument('--with-calctest', dest='calctest', action='store_true', default=False)
    parser.add_argument('--load-model', dest='loadmodel', type=str, default='none')
    parser.add_argument('--test-method', dest='testmethod', type=int, default=0)
    parser.add_argument('--test-subject', dest='testsubject', type=str, default='KGT008')
    parser.add_argument('--with-calcviz', dest='calcviz', action='store_true', default=False)
    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    gpu = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    modeltype = args.modeltype
    multigpu = args.multigpu
    trainper = args.trainper
    calctest = args.calctest
    calcviz = args.calcviz
    loadmodel = args.loadmodel
    nosavemodel = args.nosavemodel
    testmethod = args.testmethod
    testsubject = args.testsubject

    K.set_floatx(args.dtype)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_loglevel
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
    sys.path.append(dataprep)
    import tokenizer

    prep('loading tokenizers... ')
    smltok = pickle.load(open('%s/smls.tok' % (dataprep), 'rb'), encoding='UTF-8')
    drop()

    prep('loading dataset... ')
    dataset = pickle.load(open('%s/dataset.pkl' % (dataprep), 'rb'))
    drop()

    # need to convert samples to dict format for batch_gen
    # sample id (sid) is in the first position of the sample
    #samples = dict()
    #for sample in dataset['samples']:
        #samples[sample[0]] = sample[1:]
    #allsids = list(samples.keys())
    
    # we are so data-limited we use testing as validation for now
    #trainlen = int((trainper/100) * len(dataset['samples']))
    #trainset = dict()
    #testset = dict()

    #for sid in allsids[:trainlen]:
        #trainset[sid] = samples[sid]
        
    #for sid in allsids[trainlen:]:
        #testset[sid] = samples[sid]

    #  sid, fid, subject, whole_token_in_original_code, word_position, gaze_time_on_word, subject_total_gaze_time

    testsubjects = [testsubject]
    testmethods = [testmethod]

    trainset = dict()
    testset = dict()
    valset = dict()
    for sample in dataset['samples']:
        
        if sample[1] in testmethods and sample[2] in testsubjects:
            testset[sample[0]] = sample[1:]
        #elif sample[1]==19 and sample[2] == 'KGT001':
            #valset[sample[0]] = sample[1:]
        else:
            if not sample[2] in testsubjects:
                trainset[sample[0]] = sample[1:]
        
        #if sample[2] in testsubjects: # split by subject
        #if sample[1] in testmethods: # split by method
        #if sample[2] in testsubjects and sample[1] in testmethods:
        #if sample[2] in testsubjects or sample[1] in testmethods:
            #testset[sample[0]] = sample[1:]
        #elif sample[2] in testsubjects:
        #else:
        #if sample[2] in testsubjects and not sample[1] in testmethods:
            #trainset[sample[0]] = sample[1:]
    trainlen = len(trainset)
    
    steps = int(len(trainset)/batch_size)+1
    valsteps = int(len(valset)/batch_size)+1
    
    smlvocabsize = smltok.vocab_size

    print('smlvocabsize %s' % (smlvocabsize))
    print('batch size {}'.format(batch_size))
    print('steps {}'.format(steps))
    print('training data size {}'.format(trainlen))
    print('vaidation data size {}'.format(len(dataset['samples']) - trainlen))
    print('------------------------------------------')

    config = dict()
    
    config['smlvocabsize'] = smlvocabsize
    config['smllen'] = 400
    config['multigpu'] = multigpu
    config['batch_size'] = batch_size
    print(valset)
    if loadmodel != 'none':
        prep('loading config... ')
        (modeltype, mid, timestart) = loadmodel.split('_')
        (timestart, ext) = timestart.split('.')
        modeltype = modeltype.split('/')[-1]
        config = pickle.load(open(outdir+'/histories/'+modeltype+'_conf_'+timestart+'.pkl', 'rb'))
        config['maxastnodes'] = 400
        num_inputs = config['num_input']
        drop()

        prep('loading model... ')
        model = tensorflow.keras.models.load_model(loadmodel)
        print(model.summary())
        drop()
    else:
        prep('creating model... ')
        config, model = create_model(modeltype, config)
        drop()

        print(model.summary())

        gen = batch_gen(trainset, dataset, config)
        checkpoint = ModelCheckpoint(outdir+'/models/'+modeltype+'_E{epoch:02d}_'+str(timestart)+'.h5')
        savehist = HistoryCallback()
        savehist.setCatchExit(outdir, modeltype, timestart, config)
        
        valgen = batch_gen(valset, dataset, config)

        if nosavemodel:
            callbacks = [ ]
        else:
            callbacks = [ checkpoint, savehist ]

        try:
            history = model.fit(gen, steps_per_epoch=steps, epochs=epochs, verbose=0, max_queue_size=8, workers=1, use_multiprocessing=False, callbacks=callbacks, validation_data=valgen, validation_steps=valsteps)
        except Exception as ex:
            print(ex)
            traceback.print_exc(file=sys.stdout)

    if calctest:
        preds = list()
        actus = list()
        
        for sid in testset.keys():
            (fid, sub, wword, wp, gtms, stgt) = testset[sid]
            
            wsmlseq = dataset['nodes'][fid] # whole function for context
            wfp = dataset['nodes'][fid][wp] # single word being viewed
            wsmledges = dataset['edges'][fid]
            # crop to maximum size
            wsmlseq = wsmlseq[:config['smllen']]
            
            # pad to minimum size
            for i in range(0, config['smllen'] - len(wsmlseq)):
                wsmlseq = np.append(wsmlseq, [0])
            
            wsmlseq = np.asarray([wsmlseq])
            wfp = np.asarray([wfp])
            #print(wsmlseq)
            #print(wsmledges)
            #print(wfp)

            if 'gnn' in modeltype:
                # crop/expand ast adjacency matrix to dense
                wsmledges = np.asarray(wsmledges.todense())
                wsmledges = wsmledges[:config['maxastnodes'], :config['maxastnodes']]
                tmp = np.zeros((config['maxastnodes'],config['maxastnodes']), dtype='int32')
                tmp[:wsmledges.shape[0], :wsmledges.shape[1]] = wsmledges
                wsmledges = np.int32(tmp)
                wsmledges = np.asarray([wsmledges])
                pred = model.predict([wsmlseq, wsmledges, wfp], batch_size=1)
            else:
                pred = model.predict([wsmlseq, wfp], batch_size=1)
                        
            actu = round(gtms / stgt, 5)
            #actu = gtms
            pred = round(pred[0][0], 5)
            diff = round(((pred-actu) / actu)*100, 2)
            
            preds.append(pred)
            actus.append(actu)
            
            #msg = '%d\t%s\t%d\t%d\t%d\t%f\t%f\t%f\t%s' % (fid, sub, wp, gtms, stgt, pred, actu, diff, wword)

            #print(msg)
        pickle.dump(preds,open("pavg/pred_%s%d.pkl" % (testsubject,testmethod),"wb"))
        pickle.dump(actus, open("pavg/actu_%s%d.pkl" % (testsubject,testmethod),"wb"))
        #corr, _ = pearsonr(preds, actus)
        #print('%s %d pearsons correlation: %.3f' % (testsubject, testmethod, corr))
        #pickle.dump(preds,open("predselect"+testsubject+str(testmethod)+".pkl","wb"))
    if calcviz:
        preds = list()
        fid = testmethod
        for wfp in dataset['nodes'][fid]:
            wfp = np.asarray([wfp])
            wsmlseq = dataset['nodes'][fid] # whole function for context
            wsmledges = dataset['edges'][fid]
            # crop to maximum size
            wsmlseq = wsmlseq[:config['smllen']]

            # pad to minimum size
            for i in range(0, config['smllen'] - len(wsmlseq)):
                wsmlseq = np.append(wsmlseq, [0])

            wsmlseq = np.asarray([wsmlseq])

            if 'gnn' in modeltype:
                # crop/expand ast adjacency matrix to dense
                wsmledges = np.asarray(wsmledges.todense())
                wsmledges = wsmledges[:config['maxastnodes'], :config['maxastnodes']]
                tmp = np.zeros((config['maxastnodes'],config['maxastnodes']), dtype='int32')
                tmp[:wsmledges.shape[0], :wsmledges.shape[1]] = wsmledges
                wsmledges = np.int32(tmp)
                wsmledges = np.asarray([wsmledges])
                pred = model.predict([wsmlseq, wsmledges, wfp], batch_size=1)
            else:
                pred = model.predict([wsmlseq, wfp], batch_size=1)

            pred = round(pred[0][0], 5)
            preds.append(pred)
            
        pickle.dump(preds,open("humanpredrnn%s%d.pkl" % (testsubject,testmethod),"wb"))


