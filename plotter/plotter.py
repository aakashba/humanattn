import matplotlib
import matplotlib.pyplot as plt
import math
import pickle
import numpy as np
import argparse
import scipy

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

extradata = pickle.load(open('/nfs/projects/funcom/data/javastmt/q90/dataset_short.pkl', 'rb'))
graphdata = pickle.load(open('/nfs/projects/funcom/data/javastmt/q90/dataset_graph.pkl', 'rb'))
graphtok = extradata['graphtok']

x = pickle.load(open("/nfs/dropbox/bio_predict_viz/biodats_q90_astgnn.pkl","rb"))

breadth=15

parser = argparse.ArgumentParser(description='')
parser.add_argument('--fid', type=int, default=None)
args=parser.parse_args()

fid = args.fid

test = x[fid]
test = np.squeeze(test)
print(test)

#test = softmax(test) * 5

length = math.ceil(len(test)/breadth)

new = np.zeros((length*breadth))
new[:len(test)] = test

new = np.reshape(new,(length,breadth))

plt.savefig('gnn'+str(fid)+'.pdf')

#print(graphdata['stest_nodes'][fid])

words = list()

for n in graphdata['stest_nodes'][fid]:
    print(graphtok.i2w[n], end=' ')
    words.append(graphtok.i2w[n])

def colorize(words, color_array):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    cmap = matplotlib.cm.get_cmap('RdBu')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    return colored_string
    
#words = 'The quick brown fox jumps over the lazy dog'.split()
#color_array = np.random.rand(len(words))
#s = colorize(words, color_array)

s = colorize(words, (5 * test).tolist())

with open('colorize'+str(fid)+'.html', 'w') as f:
    f.write(s)

print()

