import preprocessing as pp
import modelling as md
import pandas as pd
import re
import matplotlib.pyplot as plt

def regression_summary(lin_ac,las_ac,rig_ac,ela_ac,knr_ac):
    print("\n\n\n**********************************************\n\nREGRESSION SUMMARY")
    print('\nLinear Regression\n')
    for key,value in lin_ac.items():
        print(key," : ",value)
    print('\nLasso Regression\n')
    for key,value in las_ac.items():
        print(key," : ",value)
    print('\nRidge Regression\n')
    for key,value in rig_ac.items():
        print(key," : ",value)
    print('\nElasticNet Regression\n')
    for key,value in ela_ac.items():
        print(key," : ",value)
    print('\nKNN Regression\n')
    for key,value in knr_ac.items():
        print(key," : ",value)
    print("\n")

def classification_summary(log_ac,knn_ac,dtc_ac,svm_ac,gnb_ac):
    print("\n\n\n**********************************************\n\CLASSIFICATION SUMMARY")
    print('\nLogistic Regression\n')
    for key,value in log_ac.items():
        print(key," : ",value)
    print('\nKNN\n')
    for key,value in knn_ac.items():
        print(key," : ",value)
    print("\nDecision Tree\n")
    for key,value in dtc_ac.items():
        print(key," : ",value)
    print("\nSVM\n")
    for key,value in svm_ac.items():
        print(key," : ",value)
    print("\nGuassian Naive Bayes\n")
    for key,value in gnb_ac.items():
        print(key," : ",value)


def regression_models(data_pca,target,path):
    print('\nLinear Regression\n')
    lin_ac=md.Linear_model(data_pca,target,path)
    print('\nLasso Regression\n')
    las_ac=md.Lasso_model(data_pca,target,path)
    print('\nRidge Regression\n')
    rig_ac=md.Ridge_model(data_pca,target,path)
    print('\nElasticNet Regression\n')
    ela_ac=md.elastinet_model(data_pca,target,path)
    print('\nKNN Regression\n')
    knr_ac=md.KNNReg_model(data_pca,target,path)
    regression_summary(lin_ac,las_ac,rig_ac,ela_ac,knr_ac)

def classification_models(data_pca,target,path):
    fig, axes = plt.subplots(5)
    fig.set_figheight(25)
    fig.set_figwidth(5)
    print('\nLogistic Regression\n')
    axes[0].set_title('Logistic ROC')
    log_ac = md.Logistic_model(data_pca,target,path,axes[0])

    print('\nKNN\n')
    axes[1].set_title('KNN ROC')
    knn_ac = md.KNN_model(data_pca,target,path,axes[1])

    print("\nDecision Tree\n")
    axes[2].set_title('Decision Tree ROC')
    dtc_ac = md.DT_model(data_pca,target,path,axes[2])

    print("\nSVM\n")
    axes[3].set_title('SVM ROC')
    svm_ac = md.SVM_model(data_pca,target,path,axes[3])

    print("\nGuassian Naive Bayes\n")
    axes[4].set_title('Guassian Naive Bayes ROC')
    gnb_ac = md.G_NB_measures(data_pca,target,path,axes[4])

    classification_summary(log_ac,knn_ac,dtc_ac,svm_ac,gnb_ac)

def autoML_run(filepath, target, name):
    print("Preprocessing\n")
    path = name+"/"
    data = pd.read_csv(filepath)
    field_types_raw = pp.get_input_types(data)
    field_types = pp.normalize_col_names(data,field_types_raw)
    pattern = re.compile('\W+')
    target = re.sub(pattern, '_', target.lower())
    problem = pp.identify_problem_type(data,field_types,target)
    print("Problem Type: ", problem)
    data_prop, field_types = pp.proper_dataframe(data,field_types,target,problem)
    data_no_null,drop_cols = pp.handle_null_values(data_prop)
    data_norm = pp.normalisation(data_no_null,target,field_types)
    count = pp.remove_outliers(data_norm,target,field_types)
    print("Outliers Count : ", count)
    data_pca,n = pp.pca_df(data_norm,target,field_types)
    if n==0:
        print("No features reduced in PCA")
    else:
        print("Reduced to N : ", n," features")
    print("\nModelling")
    pp.save_final_df(data_pca,name,path)
    if problem == 'regression':
        regression_models(data_pca,target,path)
    elif problem =='classification':
        classification_models(data_pca,target,path)
