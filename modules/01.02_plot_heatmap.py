# 01.02 correlation heatmap plot

# create correlation heatmap
# https://stackoverflow.com/questions/13411544/delete-column-from-pandas-dataframe
df_num = df.drop('FullName', 1)
print('Output dataset info:')
print(df_num.info(), '\n')
# https://seaborn.pydata.org/generated/seaborn.heatmap.html
# https://heartbeat.fritz.ai/seaborn-heatmaps-13-ways-to-customize-correlation-matrix-visualizations-f1c49c816f07
sns.heatmap(
    df_num.corr(),
    cmap='coolwarm'
).set_title('Correlation Heatmap')
