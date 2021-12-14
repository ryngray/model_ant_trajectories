import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

f = pd.read_csv("ants.txt", delimiter=',')

sub_traj = 0
epsilon = 100
traj_total = []
max_len = 200
print("DONE READING FILE")
for i in tqdm(np.unique(f['id'])):
    fi = f[f['id']==i]
    tmp_list = []
    for j in range(np.shape(fi)[0]):
        if(j==0):
            tmp_list.append(fi.iloc[j])
        elif(abs(fi.iloc[j]['x'] - fi.iloc[j-1]['x']) < epsilon and abs(fi.iloc[j]['y'] - fi.iloc[j-1]['y']) < epsilon):
            tmp_list.append(fi.iloc[j])
        else:
            #print("ERROR!", fi.iloc[j]['x'], fi.iloc[j-1]['x'])
            tmp_list = []
        if(len(tmp_list) == max_len):
            traj_total.append(tmp_list)
            tmp_list = []
    sub_traj+=1

print(np.shape(traj_total))
traj_total = np.asarray(traj_total)

names = ['x', 'y', 't', 'id', 'tmp']
tmp_traj = traj_total.copy()
df = pd.DataFrame(tmp_traj.reshape(-1, 5), columns=names)
df.index = np.repeat(np.arange(tmp_traj.shape[0]), tmp_traj.shape[1]) + 1
df.index.name = 'minute_id'

df.to_csv(".csv")  