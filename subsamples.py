import pandas as pd

# cutoff value: 
# 1 second, 30 frames
# 2 seconds, 60 frames
# 5 seconds, 150 frames
# 10 seconds, 300 frames
# 15 seconds, 450 frames
# 20 seconds, 600 frames
# 30 seconds, 900 frames
# 40 seconds, 1200 frames
# 50 seconds, 1500 frames
# 60 seconds, 1800 frames
# 90 seconds, 2700 frames
# 120 seconds, 3600 frames
CUTOFF=150

def main():
    df = pd.read_csv("labeled_moments.csv")
    df['subsample'] = -1
    df['level'] = -1
    oldsamples = df.groupby('sample_id')
    res = pd.DataFrame(columns=df.columns)
    id = 0
    for name,group in oldsamples:
        level=0
        print(name)
        group = group.reset_index(drop=True)
        offset = 0
        while offset+CUTOFF < len(group):
            group.loc[offset:offset+CUTOFF-1, 'subsample'] = id
            group.loc[offset:offset+CUTOFF-1, 'level'] = level
            res = pd.concat([res, group.loc[offset:offset+CUTOFF-1,:]], ignore_index=True)
            id += 1
            level += 1
            offset += CUTOFF
    res = res.rename(columns={'sample_id': 'scenario_id'})
    res = res.rename(columns={'subsample': 'sample_id'})
    res.to_csv("all_subsamples_"+str(CUTOFF)+".csv",index=False,header=True)


if __name__ == "__main__":
    main()
