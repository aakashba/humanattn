import pickle
import statistics
from scipy.stats import pearsonr
x = pickle.load(open("pavgpredfull.pkl","rb"))
new = dict()

for method in x:
    for user in x[method]:
        r = x[method][user]
        for n in x[method]:
            if n != user:
                z = x[method][n]
                corr,_ = pearsonr(r,z)
                if user not in new:
                    new[user] = [corr]
                else:
                    new[user].append(corr)

for user in new:
    print(user)
    print(statistics.mean(new[user]))
