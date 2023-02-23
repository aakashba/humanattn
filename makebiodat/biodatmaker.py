import tensorflow as tf
import tensorflow.keras
import pickle
import numpy as np
from myutils import prep, drop
from custom.graphlayer import GCNLayer
import gc
dataprep = '/nfs/projects/humanattn/data/javastmt/q90'

prep('loading tokenizers... ')
tdatstok = pickle.load(open('%s/tdats.tok' % (dataprep), 'rb'), encoding='UTF-8')
comstok = pickle.load(open('%s/coms.tok' % (dataprep), 'rb'), encoding='UTF-8')
#sdatstok = pickle.load(open('%s/sdats.tok' % (dataprep), 'rb'), encoding='UTF-8')
smltok = pickle.load(open('%s/smls.tok' % (dataprep), 'rb'), encoding='UTF-8')
drop()

prep('loading model... ')
biomodelfn = '../../humanattn/outdir/models/astgnn-fullemb_E70_1628693365.h5'
biomodel = tensorflow.keras.models.load_model(biomodelfn, custom_objects={"GCNLayer":GCNLayer})
drop()

prep('loading sequences... ')
seqdata = pickle.load(open('%s/dataset_graph.pkl' % (dataprep), 'rb'))
drop()

#biomodel._make_predict_function()

tts = [ 'train', 'val', 'test' ]

humanattns = dict()
tempins = list()
fpses = list()


maxastnodes = 400

klist= seqdata.keys()

klist = list(klist)
for k in klist:
    if "nodes" not in k and "edges" not in k:
        del seqdata[k]
    
gc.collect()

c = 0
for tt in tts:
    for fid in seqdata['s%s_nodes' % (tt)].keys():
        wsmlnodes = seqdata['s%s_nodes' % (tt)][fid]
        #wsmledges = seqdata['s%s_edges' % (tt)][fid]

        # crop/expand ast sequence
        wsmlnodes = wsmlnodes[:400]
        tmp = np.zeros(400, dtype='int32')
        tmp[:wsmlnodes.shape[0]] = wsmlnodes
        tnodes = np.int32(tmp)
        wsmlnodes = np.int32(wsmlnodes)
        fps = list()
        for node in wsmlnodes:
            fps.append([node])
        n = len(fps)
        fps = np.asarray(fps)

        tempnodes = np.asarray(n * [tnodes])
        
        #for i in range(0, 400):
        #    tempins.extend(wsmlnodes)
        
        #tempins.extend(tempin)
        #fpses.extend(fps)
        
        #if (len(tempins) >= 400) and (len(tempins) % (400 * batch_size) == 0):
            
            #tempins = np.asarray(tempins)
            #fpses = np.asarray(fpses)
            
            #print(tempins.shape)
            #print(fpses.shape)
        if 'gnn' in biomodelfn:
        # crop/expand ast adjacency matrix to dense
            wsmledges = seqdata['s%s_edges' % (tt)][fid]
            wsmledges = np.asarray(wsmledges.todense())
            wsmledges = wsmledges[:maxastnodes, :maxastnodes]
            tmp = np.zeros((maxastnodes,maxastnodes), dtype='int32')
            tmp[:wsmledges.shape[0], :wsmledges.shape[1]] = wsmledges
            wsmledges = np.int32(tmp)
            tempedges = np.asarray(n* [wsmledges])
            humanattnres = biomodel.predict([tempnodes,tempedges, fps], batch_size=n)
        else:
            humanattnres = biomodel.predict([tempnodes, fps], batch_size=n)

        #    n = 400
        #    final = [humanattnres[i * n:(i + 1) * n] for i in range((len(humanattnres) + n - 1) // n )]
            
        #    for res in final:
        #        whumanattn = list()
        #        for s in res:
        #            whumanattn.append(s[0])
            
        #        humanattns[fid] = whumanattn
        
        humanattns[fid] = humanattnres
        
            #tempins = list()
            #fpses = list()
        c += 1
        if c % 1000 == 0:
            print(c)
            gc.collect()
            

pickle.dump(humanattns, open('biodats_q90_astgnn-fullemb.pkl', 'wb'))
