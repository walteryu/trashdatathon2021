# 01.02 correlation heatmap plot

# remove non-numeric columns
# https://stackoverflow.com/questions/13411544/delete-column-from-pandas-dataframe
df_num = df.drop('FullName', 1)
print('Output dataset info:')
print(df_num.info(), '\n')
# calculate correlation and generate mask for upper triangle
# https://seaborn.pydata.org/examples/many_pairwise_correlations.html
corr = df_num.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
# create heatmap with color palette
# https://seaborn.pydata.org/generated/seaborn.heatmap.html
# https://heartbeat.fritz.ai/seaborn-heatmaps-13-ways-to-customize-correlation-matrix-visualizations-f1c49c816f07
sns.heatmap(
    corr,
    mask=mask,
    cmap='coolwarm'
).set_title('Correlation Heatmap')
