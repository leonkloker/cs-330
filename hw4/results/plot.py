import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('hidden_exp.csv')
plt.figure()
plt.plot(df.iloc[:,0], df.iloc[:,1], label='MAML time')
plt.plot(df.iloc[:,0], df.iloc[:,3], label='ES time')
plt.xlabel('Hidden dimension')
plt.ylabel('Time (s)')
plt.legend()
plt.savefig('hidden_time.png')

plt.figure()
plt.plot(df.iloc[:,0], df.iloc[:,2], label='MAML memory')
plt.plot(df.iloc[:,0], df.iloc[:,4], label='ES memory')
plt.xlabel('Hidden dimension')
plt.ylabel('Memory (MB)')
plt.legend()
plt.savefig('hidden_mem.png')

