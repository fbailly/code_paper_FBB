import pickle
import pingouin as pg
import seaborn
import matplotlib.pyplot as plt

pk1 = open('./stats_df_1.pkl', 'rb')
pk2 = open('./stats_df_2.pkl', 'rb')
df1 = pickle.load(pk1)
df2 = pickle.load(pk2)

# DF1
# PLOT
plt.subplot(121)
seaborn.stripplot(y=df1['RMSE'], x=df1['co_contraction_level'], hue=df1['EMG_objective'])

# STATS
aov = pg.anova(dv='RMSE', between=['EMG_objective', 'co_contraction_level'],
               data=df1)
ptt = pg.pairwise_ttests(dv='RMSE', between=['EMG_objective', 'co_contraction_level'], data=df1, padjust='bonf')
pg.print_table(aov.round(3))
pg.print_table(ptt.round(3))

# DF2
# PLOT
plt.subplot(122)
seaborn.stripplot(y=df2['RMSE'], x=df2['Marker_noise_level_m'], hue=df2['EMG_objective'])
aov = pg.anova(dv='RMSE', between=['Marker_noise_level_m', "EMG_objective"],
               data=df2, detailed=True)
ptt = pg.pairwise_ttests(dv='RMSE', between=['Marker_noise_level_m', "EMG_objective"], data=df2, padjust='bonf')
pg.print_table(aov.round(3))
pg.print_table(ptt.round(3))

plt.show()