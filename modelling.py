from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression,Ridge,Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN, KNeighborsRegressor as KNN_Reg
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score, roc_auc_score,precision_score,recall_score,roc_curve,mean_squared_error,mean_absolute_error,r2_score
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle


def print_measures_clf(y_pred,y_test,axis,y_scores=[]):
    print("\tConfusion Matrix")
    conf = confusion_matrix(y_pred,y_test)
    print("\t",conf[0],'\n\t',conf[1])
    print("\tAccuracy = {}\n\tRecall [Neg Pos] = {}\n\tPrecision [Neg Pos] = {}\n\tF1 score [Neg Pos]  = {}".format(accuracy_score(y_pred,y_test),recall_score(y_pred,y_test,average='weighted'),precision_score(y_pred,y_test,average='weighted'),f1_score(y_pred,y_test,average='weighted')))
    if len(y_scores) != 0:
        try:
            fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
            axis.plot(fpr,tpr)
            auc = roc_auc_score(y_test, y_scores,multi_class='ovr',average='weighted')
            print(f"\tAUC = {auc}\n")
        except:
            print("\tAUC ROC Multiclass error\n")

def get_measures_clf(y_pred,y_test,y_scores=[]):
    measures = {}
    measures["accuracy"]=accuracy_score(y_pred,y_test)
    measures["precision"]=precision_score(y_pred,y_test,average='weighted')
    measures["recall"]=recall_score(y_pred,y_test,average='weighted')
    measures["F1"]=f1_score(y_pred,y_test,average='weighted')
    if len(y_scores)!=0:
        try:
            measures["AUC"]=roc_auc_score(y_test, y_scores,multi_class='ovr',average='weighted')
        except:
            print("AUC ROC Multiclass error")
    return measures

def get_measures_reg(y_pred,y_test):
    measures = {}
    measures["mse"]=mean_squared_error(y_pred,y_test)
    measures["mae"]=mean_absolute_error(y_pred,y_test)
    measures["R2"]=r2_score(y_pred,y_test)
    return measures

def print_measures_reg(y_pred,y_test):
    print("\tMean Square Error = {}\n\tMean Absolute Erroe = {}\n\tCoefficient Of Determination  = {}\n".format(mean_squared_error(y_pred,y_test),mean_absolute_error(y_pred,y_test),r2_score(y_pred,y_test)))



def Logistic_model(df, target, path,axis):
    x = df.drop(target,axis=1)
    y = df[target]

    logistic = LogisticRegression()
    penalty_log=['l1', 'l2', 'elasticnet', 'none']
    C_log=np.logspace(0,5,20)
    l1_ratio_log=np.linspace(0,1,20)
    hyperparams_log = {'penalty':penalty_log,'C':C_log,'l1_ratio':l1_ratio_log}
    log_clf= RandomizedSearchCV(logistic,hyperparams_log,cv=5)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        best_logistic_model = log_clf.fit(x,y)
    for param in hyperparams_log.keys():
        print(f"Best {param} = {best_logistic_model.best_estimator_.get_params()[param]}")

    best_penalty=best_logistic_model.best_estimator_.get_params()['penalty']
    best_c = best_logistic_model.best_estimator_.get_params()['C']
    best_l1 = best_logistic_model.best_estimator_.get_params()['l1_ratio']

    if best_penalty == 'l1' or best_penalty == 'l2':
        logistic = LogisticRegression(penalty=best_penalty, C = best_c)
    elif best_penalty =='elasticnet':
        logistic = LogisticRegression(penalty=best_penalty, C = best_c, l1_ratio=best_l1)
    else:
        logistic = LogisticRegression()

    y_unique = y.value_counts().min()


    skf = StratifiedKFold(n_splits=min(10,y_unique), shuffle=True, random_state=1)
    lst_accu_stratified = []
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        logistic.fit(x_train,y_train)
        y_pred = logistic.predict(x_test)
        y_score = np.array(logistic.predict_proba(x_test)).max(axis=1)
        lst_accu_stratified.append(get_measures_clf(y_pred, y_test, y_score))
        print_measures_clf(y_pred, y_test, axis,y_score)

    measures = {}
    for key,value in lst_accu_stratified[0].items():
        measures[key] = 0
    for item in lst_accu_stratified:
        for key,value in item.items():
            measures[key] += value
    for key,value in measures.items():
        measures[key] = value/10

    filename = 'Logistic_model.sav'
    pickle.dump(logistic, open(path+filename, 'wb'))
    filename = 'logistic_measures.sav'
    pickle.dump(measures, open(path+filename, 'wb'))

    return measures

def KNN_model(df, target, path,axis):
    x = df.drop(target,axis=1)
    y = df[target]

    knn = KNN()
    n_neighbors = np.arange(0,20)
    weights=['uniform', 'distance']
    hyperparams_knn = {'n_neighbors':n_neighbors,'weights':weights}
    knn_clf = RandomizedSearchCV(knn,hyperparams_knn,cv=5)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        best_model = knn_clf.fit(x,y)
    for param in hyperparams_knn.keys():
        print(f"Best {param} = {best_model.best_estimator_.get_params()[param]}")
    best_n=best_model.best_estimator_.get_params()['n_neighbors']
    best_weights = best_model.best_estimator_.get_params()['weights']
    y_unique = y.value_counts().min()

    skf = StratifiedKFold(n_splits=min(10,y_unique), shuffle=True, random_state=1)
    lst_accu_stratified = []
    knn = KNN(n_neighbors=best_n,weights=best_weights)#add parmeters

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        knn.fit(x_train,y_train)
        y_pred = knn.predict(x_test)
        y_score = np.array(knn.predict_proba(x_test)).max(axis=1)
        lst_accu_stratified.append(get_measures_clf(y_pred, y_test, y_score))
        print_measures_clf(y_pred, y_test,axis ,y_score)

    measures = {}
    for key,value in lst_accu_stratified[0].items():
        measures[key] = 0
    for item in lst_accu_stratified:
        for key,value in item.items():
            measures[key] += value
    for key,value in measures.items():
        measures[key] = value/10

    filename = 'KNN_model.sav'
    pickle.dump(knn, open(path+filename, 'wb'))
    filename = 'KNN_measures.sav'
    pickle.dump(measures, open(path+filename, 'wb'))

    return measures

def KNNReg_model(df, target, path):
    x = df.drop(target,axis=1)
    y = df[target]

    knn = KNN_Reg()
    n_neighbors = np.arange(0,20)
    weights=['uniform', 'distance']
    hyperparams_knn = {'n_neighbors':n_neighbors,'weights':weights}
    knn_clf = RandomizedSearchCV(knn,hyperparams_knn,cv=5)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        best_model = knn_clf.fit(x,y)
    for param in hyperparams_knn.keys():
        print(f"Best {param} = {best_model.best_estimator_.get_params()[param]}")
    best_n=best_model.best_estimator_.get_params()['n_neighbors']
    best_weights = best_model.best_estimator_.get_params()['weights']

    skf = KFold(n_splits=10, shuffle=True, random_state=1)
    lst_accu_stratified = []
    knn = KNN_Reg(n_neighbors=best_n,weights=best_weights)#add parmeters

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        knn.fit(x_train,y_train)
        y_pred = knn.predict(x_test)
        lst_accu_stratified.append(get_measures_reg(y_pred, y_test))
        print_measures_reg(y_pred, y_test)

    measures = {}
    for key,value in lst_accu_stratified[0].items():
        measures[key] = 0
    for item in lst_accu_stratified:
        for key,value in item.items():
            measures[key] += value
    for key,value in measures.items():
        measures[key] = value/10

    filename = 'KNNReg_model.sav'
    pickle.dump(knn, open(path+filename, 'wb'))
    filename = 'KNNReg_measures.sav'
    pickle.dump(measures, open(path+filename, 'wb'))

    return measures

def DT_model(df, target, path,axis):
    x = df.drop(target,axis=1)
    y = df[target]

    dtc = DecisionTreeClassifier()
    criterion=["gini", "entropy"]
    splitter=["best", "random"]
    hyperparams_dtc = {'criterion':criterion,'splitter':splitter}
    dtc_clf = RandomizedSearchCV(dtc,hyperparams_dtc,cv=5)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        best_model = dtc_clf.fit(x,y)
    for param in hyperparams_dtc.keys():
        print(f"Best {param} = {best_model.best_estimator_.get_params()[param]}")

    best_crit=best_model.best_estimator_.get_params()['criterion']
    best_split = best_model.best_estimator_.get_params()['splitter']
    y_unique = y.value_counts().min()

    skf = StratifiedKFold(n_splits=min(10,y_unique), shuffle=True, random_state=1)
    lst_accu_stratified = []
    dtc = DecisionTreeClassifier(criterion=best_crit,splitter=best_split)#add parameters

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        dtc.fit(x_train,y_train)
        y_pred = dtc.predict(x_test)
        y_score = np.array(dtc.predict_proba(x_test)).max(axis=1)
        lst_accu_stratified.append(get_measures_clf(y_pred, y_test, y_score))
        print_measures_clf(y_pred, y_test, axis, y_score)

    measures = {}
    for key,value in lst_accu_stratified[0].items():
        measures[key] = 0
    for item in lst_accu_stratified:
        for key,value in item.items():
            measures[key] += value
    for key,value in measures.items():
        measures[key] = value/10

    filename = 'DT_model.sav'
    pickle.dump(dtc, open(path+filename, 'wb'))
    filename = 'DT_measures.sav'
    pickle.dump(measures, open(path+filename, 'wb'))

    return measures


def SVM_model(df, target, path, axis):
    x = df.drop(target,axis=1)
    y = df[target]

    svm = SVC()
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    C = np.logspace(0,5,20)
    hyperparams_svm = {'C':C, 'kernel':kernel}
    svm_clf = RandomizedSearchCV(svm,hyperparams_svm,cv=5)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        best_model = svm_clf.fit(x,y)
    for param in hyperparams_svm.keys():
        print(f"Best {param} = {best_model.best_estimator_.get_params()[param]}")

    best_kernel=best_model.best_estimator_.get_params()['kernel']
    best_C = best_model.best_estimator_.get_params()['C']
    y_unique = y.value_counts().min()

    skf = StratifiedKFold(n_splits=min(10,y_unique), shuffle=True, random_state=1)
    lst_accu_stratified = []
    svm = SVC(kernel=best_kernel,C=best_C, probability=True)

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        svm.fit(x_train,y_train)
        y_pred = svm.predict(x_test)
        y_score = np.array(svm.predict_proba(x_test)).max(axis=1)
        lst_accu_stratified.append(get_measures_clf(y_pred, y_test, y_score))
        print_measures_clf(y_pred, y_test, axis,y_score)

    measures = {}
    for key,value in lst_accu_stratified[0].items():
        measures[key] = 0
    for item in lst_accu_stratified:
        for key,value in item.items():
            measures[key] += value
    for key,value in measures.items():
        measures[key] = value/10

    filename = 'SVM_model.sav'
    pickle.dump(svm, open(path+filename, 'wb'))
    filename = 'SVM_measures.sav'
    pickle.dump(measures, open(path+filename, 'wb'))

    return measures

def Linear_model(df, target, path):
    x = df.drop(target,axis=1)
    y = df[target]

    linear = LinearRegression()

    skf = KFold(n_splits=10, shuffle=True)
    lst_accu_stratified = []

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        linear.fit(x_train,y_train)
        y_pred = linear.predict(x_test)
        lst_accu_stratified.append(get_measures_reg(y_pred, y_test))
        print_measures_reg(y_pred, y_test)

    measures = {}
    for key,value in lst_accu_stratified[0].items():
        measures[key] = 0
    for item in lst_accu_stratified:
        for key,value in item.items():
            measures[key] += value
    for key,value in measures.items():
        measures[key] = value/10

    filename = 'linear_model.sav'
    pickle.dump(linear, open(path+filename, 'wb'))
    filename = 'linear_measures.sav'
    pickle.dump(measures, open(path+filename, 'wb'))

    return measures

def Lasso_model(df, target, path):
    x = df.drop(target,axis=1)
    y = df[target]

    lasso = Lasso()
    alpha=np.linspace(0.1,1,20)
    hyperparams_log = {'alpha':alpha}
    log_clf= RandomizedSearchCV(lasso,hyperparams_log,cv=5)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        best_model = log_clf.fit(x,y)
    for param in hyperparams_log.keys():
        print(f"Best {param} = {best_model.best_estimator_.get_params()[param]}")

    best_a=best_model.best_estimator_.get_params()['alpha']

    skf = KFold(n_splits=10, shuffle=True, random_state=1)
    lst_accu_stratified = []
    lasso = Lasso(alpha=best_a)#add parameters

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        lasso.fit(x_train,y_train)
        y_pred = lasso.predict(x_test)
        lst_accu_stratified.append(get_measures_reg(y_pred, y_test))
        print_measures_reg(y_pred, y_test)

    measures = {}
    for key,value in lst_accu_stratified[0].items():
        measures[key] = 0
    for item in lst_accu_stratified:
        for key,value in item.items():
            measures[key] += value
    for key,value in measures.items():
        measures[key] = value/10

    filename = 'Lasso_model.sav'
    pickle.dump(lasso, open(path+filename, 'wb'))
    filename = 'lasso_measures.sav'
    pickle.dump(measures, open(path+filename, 'wb'))

    return measures

def Ridge_model(df,target, path):
    x = df.drop(target,axis=1)
    y = df[target]

    ridge = Ridge()
    alpha=np.linspace(0.1,1,20)
    hyperparams_log = {'alpha':alpha}
    log_clf= RandomizedSearchCV(ridge,hyperparams_log,cv=5)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        best_model = log_clf.fit(x,y)
    for param in hyperparams_log.keys():
        print(f"Best {param} = {best_model.best_estimator_.get_params()[param]}")

    best_a=best_model.best_estimator_.get_params()['alpha']

    skf = KFold(n_splits=10, shuffle=True, random_state=1)
    lst_accu_stratified = []
    ridge = Ridge(alpha=best_a)#add parameters

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        ridge.fit(x_train,y_train)
        y_pred = ridge.predict(x_test)
        lst_accu_stratified.append(get_measures_reg(y_pred, y_test))
        print_measures_reg(y_pred, y_test)

    measures = {}
    for key,value in lst_accu_stratified[0].items():
        measures[key] = 0
    for item in lst_accu_stratified:
        for key,value in item.items():
            measures[key] += value
    for key,value in measures.items():
        measures[key] = value/10

    filename = 'Ridge_model.sav'
    pickle.dump(ridge, open(path+filename, 'wb'))
    filename = 'Ridge_measures.sav'
    pickle.dump(measures, open(path+filename, 'wb'))

    return measures

def elastinet_model(df, target, path):
    x = df.drop(target,axis=1)
    y = df[target]

    elast = ElasticNet()
    alpha=np.linspace(0.1,1,20)
    l1_ratio=np.linspace(0,1,20)
    hyperparams_log = {'alpha':alpha, 'l1_ratio':l1_ratio}
    log_clf= RandomizedSearchCV(elast,hyperparams_log,cv=5)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        best_model = log_clf.fit(x,y)
    for param in hyperparams_log.keys():
        print(f"Best {param} = {best_model.best_estimator_.get_params()[param]}")

    best_a=best_model.best_estimator_.get_params()['alpha']
    best_l1=best_model.best_estimator_.get_params()['l1_ratio']


    skf = KFold(n_splits=10, shuffle=True, random_state=1)
    lst_accu_stratified = []
    elast = ElasticNet(alpha=best_a,l1_ratio=best_l1)#add parameters

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        elast.fit(x_train,y_train)
        y_pred = elast.predict(x_test)
        lst_accu_stratified.append(get_measures_reg(y_pred, y_test))
        print_measures_reg(y_pred, y_test)

    measures = {}
    for key,value in lst_accu_stratified[0].items():
        measures[key] = 0
    for item in lst_accu_stratified:
        for key,value in item.items():
            measures[key] += value
    for key,value in measures.items():
        measures[key] = value/10

    filename = 'elastinet_model.sav'
    pickle.dump(elast, open(path+filename, 'wb'))
    filename = 'elastinet_measures.sav'
    pickle.dump(measures, open(path+filename, 'wb'))

    return measures

def C_NB_model(df, target, path):
    x = df.drop(target,axis=1)
    y = df[target]

    cnb = CategoricalNB()
    alpha=np.linspace(0.1,1,20)
    hyperparams_log = {'alpha':alpha}
    log_clf= RandomizedSearchCV(cnb,hyperparams_log,cv=5)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        best_model = log_clf.fit(x,y)
    for param in hyperparams_log.keys():
        print(f"Best {param} = {best_model.best_estimator_.get_params()[param]}")
    best_a=best_model.best_estimator_.get_params()['alpha']
    y_unique = y.value_counts().min()

    skf = StratifiedKFold(n_splits=(10,y_unique), shuffle=True, random_state=1)
    lst_accu_stratified = []
    cnb = CategoricalNB(alpha=best_a)#add parameters

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        cnb.fit(x_train,y_train)
        y_pred = cnb.predict(x_test)
        y_score = np.array(cnb.predict_proba(x_test)).max(axis=1)
        lst_accu_stratified.append(get_measures_clf(y_pred, y_test, y_score))
        print_measures_clf(y_pred, y_test, y_score)

    measures = {}
    for key,value in lst_accu_stratified[0].items():
        measures[key] = 0
    for item in lst_accu_stratified:
        for key,value in item.items():
            measures[key] += value
    for key,value in measures.items():
        measures[key] = value/10

    filename = 'Cat_NB_model.sav'
    pickle.dump(cnb, open(path+filename, 'wb'))
    filename = 'Cat_NB_measures.sav'
    pickle.dump(measures, open(path+filename, 'wb'))

    return measures

def G_NB_measures(df, target, path,axis):
    x = df.drop(target,axis=1)
    y = df[target]

    y_unique = y.value_counts().min()


    skf = StratifiedKFold(n_splits=min(10,y_unique), shuffle=True, random_state=1)
    lst_accu_stratified = []
    gnb = GaussianNB()

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        gnb.fit(x_train,y_train)
        y_pred = gnb.predict(x_test)
        y_score = np.array(gnb.predict_proba(x_test)).max(axis=1)
        lst_accu_stratified.append(get_measures_clf(y_pred, y_test, y_score))
        print_measures_clf(y_pred, y_test, axis, y_score)

    measures = {}
    for key,value in lst_accu_stratified[0].items():
        measures[key] = 0
    for item in lst_accu_stratified:
        for key,value in item.items():
            measures[key] += value
    for key,value in measures.items():
        measures[key] = value/10

    filename = 'Gauss_NB_model.sav'
    pickle.dump(gnb, open(path+filename, 'wb'))
    filename = 'Gauss_NB_measures.sav'
    pickle.dump(measures, open(path+filename, 'wb'))

    return measures
