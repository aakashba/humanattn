import os
import pickle

dirlist = os.listdir("./pavgfull")

print(dirlist)

pred = dict()
actu = dict()


for f in dirlist:
    x = pickle.load(open("pavgfull/"+f,"rb"))
    fn = f.split("_")[-1].split(".")[0]
    method=int(fn[-1])
    user=fn[:-1]
    if method not in pred:
        pred[method]={}
    pred[method][user] = x

pickle.dump(pred,open("pavgpredfull.pkl","wb"))
