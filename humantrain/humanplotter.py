import pickle
import matplotlib.pyplot as plt
import math
import argparse
import scipy
import numpy as np
import matplotlib

x = pickle.load(open("/nfs/projects/humanattn/data/eyesum/dataset.pkl","rb"))

parser = argparse.ArgumentParser(description='')
parser.add_argument('--fid', type=int, default=None)
parser.add_argument('--subject', type=str, default=None)
args=parser.parse_args()
graphtok = pickle.load(open("/nfs/projects/humanattn/data/eyesum/smls.tok","rb"), encoding='UTF-8')

fid = args.fid
subject = args.subject


pred = pickle.load(open("humanpredrnn"+subject+str(fid)+".pkl","rb"))
pred = np.asarray(pred)
pred = (15*pred)
print(pred)

method = x['nodes'][fid]
new = np.zeros((len(method)))

for sample in x['samples']:
    if sample[1]==fid and sample[2]==subject:
        new[sample[4]] = round(sample[5]/sample[6],5)

new = (15*new)
#new = 0.2+new
new = new.tolist()

print(new)
words=list()

for n in method:
    print(graphtok.i2w[n], end=' ')
    words.append(graphtok.i2w[n])

def colorize(words, color_array):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    cmap = matplotlib.cm.get_cmap('coolwarm')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    return colored_string

s = colorize(words, new)
p = colorize(words, pred)
with open('colorizecomprnn'+str(fid)+subject+'.html','w') as f:
    f.write(s)
    f.write("<br />\n")
    f.write(p)



