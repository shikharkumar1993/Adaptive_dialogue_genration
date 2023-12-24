# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from hmmlearn.base import ConvergenceMonitor
import numpy as np
from hmmlearn import hmm
from hmmlearn.hmm import CategoricalHMM
import pandas as pd
import matplotlib.pyplot as plt
import pomegranate as pg

data=pd.read_csv("HMM/data/orignal_data.csv")
#print(data)
or_s=data['Communicative actions']
us_i=data['User']

us_i=np.array(us_i)

or_st=np.zeros(len(or_s))
for i in range(0,len(data)):
    if(or_s[i]=='S1'):
        or_st[i]=0
    elif(or_s[i]=='S2'):
        or_st[i]=1
    elif(or_s[i]=='S3'):
        or_st[i]=2
    elif(or_s[i]=='S4'):
        or_st[i]=3


or_st=np.array(or_st)
or_st=or_st.astype(int)

obs=data['Observations'].to_numpy()
states=["S1","S2","S3","S4"]
n_states=len(states)
observation=["No_resp","gestures","what","which","where","successfull"]
n_observation=len(observation)
start_probability = np.array([1.0, 0.0, 0.0, 0.0])



model=hmm.CategoricalHMM(n_components=4,random_state=42,n_iter=10000,init_params='mtcw')
model.startprob_ = start_probability
#model.transmat_ = transition_probability
#model.emissionprob_ = emission_probability
obs_seq_whole1 = np.array(obs).reshape(-1, 1)
#print(obs_seq_whole)
Y=model.fit(obs_seq_whole1)
print(model.score(obs_seq_whole1))
print(model.monitor_)
hidden_states = model.predict(obs_seq_whole1)
mscore=model.score(obs_seq_whole1)
obs_seq_whole=np.array(obs)
observations_sequence=obs_seq_whole1
max_rand=0
#for j in range(1,37):
#    b=np.where(us_i==j)[0]
#    print(b)
#    observations_sequence=obs_seq_whole[b[0]:b[len(b)-1]]
#    observations_sequence=observations_sequence.reshape(-1,1)
#    print('@@@@@@@@@@@@@@@@@@@')
#    print(j)
for i in range(1,100):
        model = hmm.CategoricalHMM(n_components=4, random_state=i, n_iter=10000,init_params='mtcw')
        model.startprob_ = start_probability
        # model.transmat_ = transition_probability
        # model.emissionprob_ = emission_probability
        #observations_sequence = np.array(obs).reshape(-1, 1)

        Y = model.fit(observations_sequence)
        mscore2=model.score(observations_sequence)
        if(mscore2>mscore):
            mscore=mscore2
            max_rand=i
model=hmm.CategoricalHMM(n_components=4,random_state=max_rand,n_iter=10000,init_params='mtcw')
model.startprob_ = start_probability
#model.transmat_ = transition_probability
#model.emissionprob_ = emission_probability
#observations_sequence = np.array(obs).reshape(-1, 1)

Y=model.fit(observations_sequence)
print(model.monitor_)
hidden_states = model.predict(observations_sequence)
print("max_rand")
print(max_rand)
print("Most likely hidden states:", hidden_states)
print("transition probability")
print(model.transmat_)
print("start probability")
print(model.startprob_)
print("emission probability")
print(model.emissionprob_)

log_probability, hidden_states = model.decode(observations_sequence,
                                              lengths=len(observations_sequence),
                                              algorithm='viterbi')

print('Log Probability :', log_probability)
print("Most likely hidden states:", hidden_states)
print("score of the model")
df=pd.DataFrame(hidden_states)
df.to_csv("C:/model_algo/hmm_hidden_state_result2.csv")
print(model.score(observations_sequence))
