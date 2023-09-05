import numpy as np
import pandas as pd
import os
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
from aeon.classification.interval_based import DrCIFClassifier
from aeon.classification.hybrid import HIVECOTEV2
from sklearn.dummy import DummyClassifier
#from aeon.classification.deep_learning.resnet import ResNetClassifier
#from aeon.classification.deep_learning.inception_time import InceptionTimeClassifier
import scipy.stats as stats
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,recall_score,roc_auc_score,log_loss,mean_squared_error,mean_squared_log_error


def get_data(df, frac=0.8,classn='memorized'):

    def balance(df, num_samples=None):
        options = df.drop_duplicates(subset=['sample_id'])
        classnums = options[[classn]].value_counts()
        if num_samples is None:
            num_samples = min(classnums)
        class_lists = []
        for cl,count in enumerate(classnums):
            class_i = options[options[classn].isin([cl])]
            class_i_sample = class_i.sample(num_samples).sort_index()['sample_id']
            class_lists.append(class_i_sample)
        selec = pd.concat(class_lists)
        res = df[df['sample_id'].isin(selec)]
        return res

    train_ids = df.groupby('sample_id').first().sample(frac=frac).sort_index().index.to_list()
    testdf = df.loc[~df['sample_id'].isin(train_ids)]
    test_ids = testdf.groupby('sample_id').first().sort_index().index.to_list()
    train_df = df.loc[df['sample_id'].isin(train_ids)]
    balanced = balance(df=train_df)
    train_ids = balanced.groupby('sample_id').first().sort_index().index.to_list()
    samples = df.groupby('sample_id')
    y_train = []
    X_train = []
    y_test = []
    X_test = []
    cols = ['screen','robot','other','screen_left','screen_right']
    boole=True
    for name, group in samples:
        sample = []
        for col in cols:
            dim = group[col].to_list()
            sample.append(dim)
        if boole:
            boole=False
        if name in train_ids:
            X_train.append(sample)
            y_train.append(group[classn].head(1).item())
        elif name in test_ids:
            X_test.append(sample)
            y_test.append(group[classn].head(1).item())
    return np.array(X_train),np.array(y_train),np.array(X_test),np.array(y_test)


def test(model,X,y,labels):
    y_pred_prob = model.predict_proba(X)
    y_pred = y_pred_prob.argmax(axis=1)
    acc = accuracy_score(y,y_pred)
    bacc = balanced_accuracy_score(y,y_pred)
    f1 = f1_score(y,y_pred,average='weighted',labels=labels)
    recall = recall_score(y,y_pred,average='weighted',labels=labels)
    rocauc = 0 #roc_auc_score(y,y_pred_prob,average='weighted',multi_class='ovr',labels=labels)
    logloss = log_loss(y,y_pred_prob,labels=labels)
    mse = mean_squared_error(y,y_pred)
    lmse = mean_squared_log_error(y,y_pred)
    return [acc,bacc,f1,recall,rocauc,logloss,mse,lmse]


def rocket(X, y, k=1000):
    print("training rocket")
    hc2 = RocketClassifier(num_kernels=k)
    hc2.fit(X, y)
    return hc2


def cif(X, y, est=30, ints=5):
    print("training cif")
    clf = DrCIFClassifier(n_estimators=est, n_intervals=ints, att_subsample_size=5)
    clf.fit(X, y)
    return clf


def stc(X, y,n=10000):
    print("training stc")
    hc2 = ShapeletTransformClassifier(n_shapelet_samples=n)
    hc2.fit(X, y)
    return hc2


def tde(X, y,n=250,m=100):
    print("training tde")
    hc2 = TemporalDictionaryEnsemble(n_parameter_samples=n,max_ensemble_size=m)
    hc2.fit(X, y)
    return hc2


def hc2(X, y, est=200, ints=25):
    print("training hc2")
    hc2 = HIVECOTEV2(stc_params={"n_shapelet_samples": 10000},
                     drcif_params={"n_estimators": 30, "n_intervals": 5, "att_subsample_size": 10}, 
                     arsenal_params={"num_kernels": 1000, "n_estimators": 30}, 
                     tde_params={"n_parameter_samples": 250, "max_ensemble_size": 100},n_jobs=3)
    hc2.fit(X, y)
    return hc2


def random(X, y):
    print("training random model")
    hc2 = DummyClassifier(strategy="uniform")
    hc2.fit(X, y)
    return hc2


def train_test(name, X_train,y_train,X_test,y_test):
    if name == "rocket":
        model = rocket(X_train, y_train)
    elif name == "cif":
        model = cif(X_train, y_train)
    elif name == "stc":
        model = stc(X_train, y_train)
    elif name == "tde":
        model = tde(X_train, y_train)
    elif name == "hc2":
        model = hc2(X_train, y_train)
    elif name == "random":
        model = random(X_train, y_train)
    labels = list(set(y_train.tolist()+y_test.tolist()))
    scores = test(model, X_test, y_test, labels)
    return scores


def main(datafile, resfile):
    datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), datafile)
    df = pd.read_csv(datapath)
    models = ["random", "rocket", "cif", "stc", "tde", "hc2"]
    models = ["random","rocket"]
    problems = ["memorized", "reason", "quality"]
    res = pd.DataFrame(columns=['problem','model','accuracy','balanced_accuracy','f1','recall','rocauc','logloss','mse','lmse'])
    for p in problems:
        print(p)
        X_train, y_train, X_test, y_test = get_data(df,frac=0.8,classn=p)
        for m in models:
            scores = train_test(m,X_train,y_train,X_test,y_test)
            res.loc[len(res)] = [p,m] + scores
    res.to_csv(resfile,index=False,header=True)


def ml_significance_test(resf,mcompare="rocket",metric="f1"):
    df = pd.read_csv(resf)
    dct2 = {"hc2": "HIVE-COTE v2", "rocket": "ROCKET", "cif": "DrCIF", "stc": "STC", "tde": "TDE"}
    data1 = df.loc[df['model']=='random',:]
    data2 = df.loc[df['model']==mcompare,:]
    res = stats.ttest_ind(data1.loc[:,metric].to_list(),data2.loc[:,metric].to_list())
    init = "Over all the problems, there was a significant difference in the F1 score between " if res.pvalue<0.05 else "Over all the problems, there was not a significant difference in the F1 score between "
    meansstr = dct2[mcompare] + " ($M"+format(data2.loc[:,metric].mean())+", SD"+format(data2.loc[:,metric].std())+"$) and Dummy ($M"+format(data1.loc[:,metric].mean())+", SD"+format(data1.loc[:,metric].std())+"$)"
    resstr = "; $t("+str(len(data2)+len(data1))+")"+format(abs(res.statistic))+",p"+format(res.pvalue)+"$."
    print(init+meansstr+resstr)
    for name,group in data1.groupby('problem'):
        x = group.loc[:,metric]
        y = data2.loc[df['problem']==name,metric]
        res = stats.ttest_ind(x.to_list(),y.to_list())
        meansstr = dct2[mcompare] + " ($M"+format(x.mean())+", SD"+format(x.std())+"$) and Dummy ($M"+format(y.mean())+", SD"+format(y.std())+"$)"
        resstr = "; $t("+str(len(x)+len(y))+")"+format(abs(res.statistic))+",p"+format(res.pvalue)+"$."
        init = "There was a " if res.pvalue < 0.05 else "There was not a "
        middle = "significant difference in the F1 score for problem "+name+" between "
        print(init+middle+meansstr+resstr)


def read(resfile):
    models = ["cif", "rocket", "stc", "tde", "hc2"]
    for m in models:
        ml_significance_test(resfile,mcompare=m,metric="f1")


if __name__ == "__main__":
    d = 'labeled_moments.csv'
    resf = 'crossval_results.csv'
    main(datafile=d,resfile=resf)
    read(resf)
