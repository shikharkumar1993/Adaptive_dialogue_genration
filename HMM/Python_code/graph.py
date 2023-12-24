from hmmlearn.base import ConvergenceMonitor
import numpy as np
from hmmlearn import hmm
from hmmlearn.hmm import CategoricalHMM
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pomegranate as pg

data=pd.read_csv("C:/model_algo/data_hmm.csv")
or_s=data['State']
obs=data['Observations']
obs0=[0,0,0,0]
obs1=[0,0,0,0]
obs2=[0,0,0,0]
obs3=[0,0,0,0]
obs4=[0,0,0,0]
obs5=[0,0,0,0]
obsS1=[0,0,0,0,0,0]
obsS2=[0,0,0,0,0,0]
obsS3=[0,0,0,0,0,0]
obsS4=[0,0,0,0,0,0]
for i in range (0, len(or_s)):
    if (obs[i]==0 and or_s[i]==0):
        obs0[0]+=1
        obsS1[0]+=1
    elif(obs[i]==0 and or_s[i]==1):
        obs0[1]+=1
        obsS2[0] += 1
    elif(obs[i] == 0 and or_s[i] == 2):
        obs0[2]+=1
        obsS3[0] += 1
    elif(obs[i]==0 and or_s[i]==3):
        obs0[3]+=1
        obsS4[0] += 1
    elif(obs[i]==1 and or_s[i]==0):
        obs1[0]+=1
        obsS1[1] += 1
    elif(obs[i]==1 and or_s[i]==1):
        obs1[1]+=1
        obsS2[1] += 1
    elif(obs[i] ==1 and or_s[i] == 2):
        obs1[2]+=1
        obsS3[1] += 1
    elif(obs[i]==1 and or_s[i]==3):
        obs1[3]+=1
        obsS4[1] += 1
    elif (obs[i]==2 and or_s[i]==0):
        obs2[0]+=1
        obsS1[2] += 1
    elif(obs[i]==2 and or_s[i]==1):
        obs2[1]+=1
        obsS2[2] += 1
    elif(obs[i] == 2 and or_s[i] == 2):
        obs2[2]+=1
        obsS3[2] += 1
    elif(obs[i]==2 and or_s[i]==3):
        obs2[3]+=1
        obsS4[2] += 1
    elif (obs[i]==3 and or_s[i]==0):
        obs3[0]+=1
        obsS1[3] += 1
    elif(obs[i]==3 and or_s[i]==1):
        obs3[1]+=1
        obsS2[3] += 1
    elif(obs[i] == 3 and or_s[i] == 2):
        obs3[2]+=1
        obsS3[3] += 1
    elif(obs[i]==3 and or_s[i]==3):
        obs3[3]+=1
        obsS4[3] += 1
    elif (obs[i]==4 and or_s[i]==0):
        obs4[0]+=1
        obsS1[4] += 1
    elif(obs[i]==4 and or_s[i]==1):
        obs4[1]+=1
        obsS2[5] += 1
    elif(obs[i] == 4 and or_s[i] == 2):
        obs4[2]+=1
        obsS3[4] += 1
    elif(obs[i]==4 and or_s[i]==3):
        obs4[3]+=1
        obsS4[4] += 1
    elif (obs[i]==5 and or_s[i]==0):
        obs5[0]+=1
        obsS1[5] += 1
    elif(obs[i]==5 and or_s[i]==1):
        obs5[1]+=1
        obsS2[5] += 1
    elif(obs[i] == 5 and or_s[i] == 2):
        obs5[2]+=1
        obsS3[5] += 1
    elif(obs[i]==5 and or_s[i]==3):
        obs5[3]+=1
        obsS4[5] += 1
print(obs0)
print(obs1)
print(obs2)
print(obs3)
print(obs4)
print(obs5)
print(obsS1)
print(obsS2)
print(obsS3)
print(obsS4)
barWidth=0.20
fig=plt.subplots(figsize=(12,8))
obs_S0=np.arange(len(obsS1))
obs_S1=[x+barWidth for x in obs_S0]
obs_S2=[x+barWidth for x in obs_S1]
obs_S3=[x+barWidth for x in obs_S2]
print("abcd",obsS1[0:len(obsS1)-1])
print("defr",obs_S0)
plt.rcParams.update({'font.size': 12})

plt.rc('axes', labelsize=22)
plt.rc('axes',titlesize=22)
plt.bar(obs_S0[0:len(obs_S0)],obsS1[0:len(obsS1)], color='skyblue', width=barWidth,label="CA1")
plt.bar(obs_S1[0:len(obs_S1)],obsS2[0:len(obsS2)], color='brown', width=barWidth,label="CA2")
plt.bar(obs_S2[0:len(obs_S2)],obsS3[0:len(obsS3)], color='black', width=barWidth,label="CA3")
plt.bar(obs_S3[0:len(obs_S3)],obsS4[0:len(obsS4)], color='pink', width=barWidth,label="CA4")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.xlabel("Observations", fontsize=20)
plt.ylabel("Frequency",fontsize=20)
plt.xticks([r+barWidth for r in range(len(obsS1[0:len(obsS1)]))],['observ0','observ1','observ2','observ3','observ4','observ5'])
plt.legend()
plt.show()
plt.bar(obs_S0[len(obs_S0)-1],obsS1[len(obsS1)-1], color='red', width=barWidth,label="State0")
plt.bar(obs_S1[len(obs_S1)-1],obsS2[len(obsS2)-1], color='green', width=barWidth,label="State1")
plt.bar(obs_S2[len(obs_S2)-1],obsS3[len(obsS3)-1], color='blue', width=barWidth,label="State2")
plt.bar(obs_S3[len(obs_S3)-1],obsS4[len(obsS4)-1], color='purple', width=barWidth,label="State3")
plt.xlabel("Observations")
plt.ylabel("Frequency")
plt.xticks([0.1],["observ5"])
plt.legend()
plt.show()