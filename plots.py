from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd

df = pd.read_csv('results_from_stdout.csv')

plt.plot(df.knockout, df.ERM_tr, label=r'ERM train')
plt.plot(df.knockout, df.ERM_val, label=r'ERM val')
plt.plot(df.knockout, df.BBSE_tr, label=r'BBSE train')
plt.plot(df.knockout, df.BBSE_val, label=r'BBSE val')
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('fraction of positive examples knocked out')
plt.savefig('accuracies.png')
plt.clf()
