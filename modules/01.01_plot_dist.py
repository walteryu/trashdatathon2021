# 01.01 distribution plot

# plot distribution
# https://www.datacamp.com/community/tutorials/python-data-profiling
sns.distplot(
    df["SegScore"].dropna()
).set_title('Distribution of Segment Score Rating')
