# 01.01 distribution plot

# plot distribution
# https://www.datacamp.com/community/tutorials/python-data-profiling
sns.histplot(
    df["SegScore"].dropna(),
    bins=10
).set_title('Distribution of Segment Score Rating')
# visualize histogram with log scale
# https://stackoverflow.com/questions/23913151/log-log-lmplot-with-seaborn
plt.yscale('log')
