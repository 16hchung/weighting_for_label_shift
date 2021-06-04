from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd

whole_df = pd.read_csv('results_from_stdout.csv')
whole_df['p_pos_test'] = (1-whole_df.knockout)*.5/((1-whole_df.knockout)*.5+.5)
whole_df['wt_true_pos'] = whole_df.p_pos_test / .5

def plot_for_id_or_ood(id_or_ood):
    df = whole_df[whole_df.id_or_ood_val == id_or_ood]
    # plot accuracy comparison
    plt.plot(df.knockout, df.ERM_tr, label=r'ERM train')
    plt.plot(df.knockout, df.ERM_val, label=r'ERM val')
    plt.plot(df.knockout, df.BBSE_tr, label=r'BBSE train')
    plt.plot(df.knockout, df.BBSE_val, label=r'BBSE val')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('fraction of positive examples knocked out')
    plt.savefig(f'accuracies_{id_or_ood}.png')
    plt.clf()


plot_for_id_or_ood('ood')
plot_for_id_or_ood('id')

# plot weight estimation
plt.plot(whole_df[whole_df.id_or_ood_val=='id'].wt_true_pos,  whole_df[whole_df.id_or_ood_val=='id'].wt_true_pos, color='black', linestyle='dashed')
plt.plot(whole_df[whole_df.id_or_ood_val=='ood'].wt_true_pos, whole_df[whole_df.id_or_ood_val=='ood'].wt_est_pos, label='OOD target domain')
plt.plot(whole_df[whole_df.id_or_ood_val=='id'].wt_true_pos,  whole_df[whole_df.id_or_ood_val=='id'].wt_est_pos,  label='ID target domain')
plt.legend()
plt.xlabel(r'true weights $\frac{q(y)}{p(y)}$')
plt.ylabel(r'estimated weights $\hat{C}_p^{-1}q(\hat{y})$')
plt.savefig(f'weights.png', bbox_inches='tight')
plt.clf()
