import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import combine_pvalues
import math
import os
import warnings
warnings.filterwarnings("ignore")


def visual_targets(df,samplecol="sample_id",addon=""):
    datacols = ['robot','other','screen']
    classcols = ["memorized","reason","quality","handpicked"]
    res = []
    for _,group in df.groupby(samplecol):
        sampleres = {}
        for col in classcols:
            val = group.reset_index().loc[0,col]
            sampleres[col] = val
        for col in datacols:
            vc = group[col].value_counts()
            if len(vc)>1:
                val = float(round(vc[1]/sum(vc)*100,2)) 
            elif vc.index.tolist()[0]==1:
                val = 100
            else:
                val = 0.0
            sampleres[col] = val
        res.append(sampleres)
    d = pd.DataFrame(res)
    d.to_csv("visualtargetsdist"+addon+".csv",index=False,header=True)
    p_values = d.loc[:,datacols].apply(stats.shapiro).iloc[1]  # [1] to get the p-values
    _, combined_p_value = combine_pvalues(p_values)
    print(f'Combined p-value from Shapiro-Wilk tests: {combined_p_value}') # significant --> not normal
    return d


def analyze_target_time(classn="memorized",group1=[0],group2=[1],addon="",handpicked=False):
    df = pd.read_csv("visualtargetsdist"+addon+".csv")
    if handpicked:
        df = df.loc[df['handpicked']==1,:]
    cols = ["robot","other","screen"]
    res = {}
    print("gaze-time distribution over visual targets")
    for c in cols:
        print(c)
        x = df.loc[df[classn].isin(group1), c]
        y = df.loc[df[classn].isin(group2), c]
        sig = stats_tests(x,y)
        res[c] = sig
    return res


def patterns(df,grouping='sample_id'):
    #Patterns:
    ##- screenleft-screenright-screenleft
    ##- screenright-screenleft-screenright
    ##- robot-screen-robot
    ##- screen-robot-screen

    def find_patterns(name,group):
        res = []
        memorized = group.loc[:,'memorized'].head(1).item()
        reason = group.loc[:,'reason'].head(1).item()
        quality = group.loc[:,'quality'].head(1).item()
        handpicked = group.loc[:,'handpicked'].head(1).item()
        df = group.loc[:,['robot', 'other','screen_left', 'screen_right']]
        pattern = [df.columns[(df==1).iloc[2]][0],df.columns[(df==1).iloc[1]][0],df.columns[(df==1).iloc[0]][0]]
        for i in range(3,len(df)):
            cur = df.columns[(df==1).iloc[i]][0]
            if cur != pattern[0]:
                res.append({'sample': name,'memorized': memorized, 'reason': reason, 'quality': quality, 'handpicked': handpicked,
                            'item1': pattern[2],'item2': pattern[1],'item3': pattern[0]})
                pattern.insert(0,cur)
                pattern.pop()
        return res

    res = []
    for name, group in df.groupby(grouping):
        res = res + find_patterns(name,group)
    results = pd.DataFrame(res)
    #results.to_csv('patterns_sampled_quality.csv')
    return results


def patterns_frequency(df:pd.DataFrame,addon='',original:pd.DataFrame=None):
    res = []
    tops = ["screenleft_screenright_screenleft","screenright_screenleft_screenright", 
            "screenleft_other_screenleft","screenleft_other_screenright","screenright_other_screenleft","screenright_other_screenright",
            "other_screenleft_other","other_screenright_other",
            "other_robot_other", "robot_other_robot"] #### DECIDED AFTER ANALYSIS OF patterns_sampled_quality.csv
    for sampleid, sample in df.groupby('sample'):
        subres = {"memorized": sample.loc[:,'memorized'].head(1).item(), "reason": sample.loc[:,'reason'].head(1).item(), "quality": sample.loc[:,'quality'].head(1).item(), "handpicked": sample.loc[:,'handpicked'].head(1).item(), "rest": 0}
        #cdf = df.loc[df['sample'] == sample, ['item1','item2','item3',classn]]
        sample[['item1','item2','item3']] = sample[['item1','item2','item3']].applymap(lambda x: x.replace("_",""))
        sample['pattern'] = sample[['item1','item2','item3']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        vc = sample['pattern'].value_counts()
        if original is not None:
            screentime = original.loc[original['sample_id']==sampleid,'screen'].sum()
            relvc = vc/screentime
        else:
            relvc = vc/vc.sum()*100
        for ind in tops:
            if ind in relvc.index:
                subres[ind] = relvc[ind]
            else:
                subres[ind] = 0
        for ind in relvc.index[~relvc.index.isin(tops)]:
            subres["rest"] = subres["rest"] + relvc[ind]

        subres['screen_screen'] = subres["screenleft_screenright_screenleft"]+subres["screenright_screenleft_screenright"]
        subres['screen_other'] = subres["screenleft_other_screenleft"]+subres["screenleft_other_screenright"]+subres["screenright_other_screenleft"]+subres["screenright_other_screenright"]+subres["other_screenleft_other"]+subres["other_screenright_other"]
        subres['robot_other'] = subres["other_robot_other"]+subres["robot_other_robot"]
        for key in tops:
            del subres[key]
        
        if subres['screen_screen'] < math.inf and subres['screen_other'] < math.inf and subres['robot_other'] < math.inf:
            res.append(subres)
    resdf = pd.DataFrame(res)
    resdf.to_csv("patterns"+addon+".csv",index=False,header=True)


def analyze_patterns(classn="memorized",group1=[0],group2=[1],addon='',handpicked=False):
    df = pd.read_csv("patterns"+addon+".csv")
    if handpicked:
        df = df.loc[df['handpicked']==1,:]
    df = df.dropna(how='any')
    print("relative portion of within screen gaze alternations")
    x = df.loc[df[classn].isin(group1), 'screen_screen']
    y = df.loc[df[classn].isin(group2), 'screen_screen']
    stats_tests(x,y)


def fixations(df:pd.DataFrame,addon=''):
    #datacols = ['robot','other','screen_left','screen_right']
    datacols = ['robot','other','screen']
    classcols = ["memorized","reason","quality","handpicked"]
    res = []
    i = 0
    for name,group in df.groupby('sample_id'):
        subres = {}
        for col in datacols:
            subres[col] = []
        for col in classcols:
            val = group.reset_index().loc[0,col]
            subres[col] = val
        res.append(subres)
        last = ""
        tmp = ""
        fixationlen = 0
        for _,row in group.iterrows():
            for col in datacols:
                if col == last and row[col]==1:
                    fixationlen += 1
                elif col != last and row[col] == 1:
                    tmp = col
            if tmp != last:
                if last != "":
                    res[i][last].append(fixationlen)
                fixationlen = 1
                last = tmp
        i += 1
    finalres = []
    for sampleres in res:
        tot = 0
        totlen = 0
        invalid = len(sampleres['screen'])==0 and len(sampleres['screen'])==0 and len(sampleres['other'])==0
        if not invalid:
            for col in datacols:
                tot += sum(sampleres[col])
                totlen += len(sampleres[col])
                if len(sampleres[col]) > 0:
                    sampleres[col] = round(sum(sampleres[col]) / len(sampleres[col]))
                else:
                    sampleres[col] = 0
            finalres.append(sampleres)
    dfr = pd.DataFrame(finalres)
    dfr.to_csv("fixations"+addon+".csv",index=False,header=True)


def analyze_fixations(classn="memorized",group1=[0],group2=[1],addon="",handpicked=False):
    df = pd.read_csv("fixations"+addon+".csv")
    if handpicked:
        df = df.loc[df['handpicked']==1,:]
    cols = ["robot","other","screen"]
    print("fixation durations on one visual target")
    for c in cols:
        print(c)
        x = df.loc[df[classn].isin(group1), c]
        y = df.loc[df[classn].isin(group2), c]
        stats_tests(x,y)


def stats_tests(x,y,printall=True):
    mwu = stats.mannwhitneyu(x,y)
    if mwu.pvalue < 0.05 or printall:
        eff =1-(2*mwu.statistic/(len(x)*len(y)))
        effstr = "high"
        if abs(eff) < 0.1:
            effstr = "negligible"
        elif abs(eff) < 0.3:
            effstr = "small"
        elif abs(eff) < 0.5:
            effstr = "medium"
        if mwu.pvalue < 0.05:
            print("$MWU="+str(mwu.statistic)+", n_1="+str(len(x))+", n_2="+str(len(y))+", p="+str(round(mwu.pvalue,3))+",$ Effect size by rank-biserial correlation: "+effstr)
        else:
            print("$MWU="+str(mwu.statistic)+", n_1="+str(len(x))+", n_2="+str(len(y))+", p="+str(round(mwu.pvalue,3))+"$")
    res = 1 if mwu.pvalue < 0.05 else 0
    return res


def time_sig(classn="memorized",group1=[0],group2=[1],addon="",handpicked=False):
    df = pd.read_csv("all_subsamples_150.csv")
    if handpicked:
        df = df.loc[df['handpicked']==1,:]
    cols = ["robot","other","screen"]
    for c in cols:
        print(c)
        for name,group in df.groupby('level'):
            print("level "+str(name))
            x = group.loc[group[classn].isin(group1), c]
            y = group.loc[group[classn].isin(group2), c]
            stats_tests(x,y,printall=False)


def main(dp, first60=False,handpicked=False):
    #classn = "memorized" 
    #classn = "reason"
    classn = "quality"
    group1 = [1,2]
    group2 = [3]
    addon = ""
    if first60:
        addon = "_60s"
    if "subsamples_900" in dp:
        analyze_target_time(classn=classn,group1=group1,group2=group2,addon=addon,handpicked=handpicked)
        analyze_fixations(classn=classn,group1=group1,group2=group2,addon=addon,handpicked=handpicked)
        analyze_patterns(classn=classn,group1=group1,group2=group2,addon=addon,handpicked=handpicked)
    elif "subsamples_150" in dp:
        time_sig(classn=classn,group1=group1,group2=group2,addon=addon,handpicked=handpicked)


def generate_files(dp,first60=False):
    df = pd.read_csv(dp)
    addon = ""
    if first60:
        addon = "_60s"
        df = df.loc[df['level'].isin([0,1])]
    visual_targets(df,samplecol="sample_id",addon=addon) 
    fixations(df,addon=addon)
    patterns_frequency(df=patterns(df,grouping='sample_id'),addon=addon)#,original=df)


if __name__ == "__main__":
    first60=False
    handpicked=False
    datapath = "all_subsamples_900.csv" # RUN subsamples.py
    subsamples = os.path.join(os.getcwd(), datapath)
    generate_files(subsamples,first60=first60)
    main(datapath,first60=first60,handpicked=handpicked)
