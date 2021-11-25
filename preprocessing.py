import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import OrderedDict
import re
from pathlib import Path
"""
first read csv file
find out ouput class
classify regression or classification

input features
for each feature check type ( continuous, discrete)
for discrete check if its numeric or categorical
feature types : numeric (continuous, discrete)
                Text
                categorical
    if num of unique values is close to the number of rows
                ignore

Basic Preprocessing:
    Column names normalisation
    proper datatype for each column
    Null values
    For continous features check the range between columns
    If too much difference: normalize data
    Check for outliers and remove them (classwise for classification)
    Check for correllation coefficients
    if highly correlated- drop or pca

Model selection
    Regression:
        Linear Regression
        KNN
    Classification
        Logistic
        KNN
        SVM
        Decision Tree
        Naive Bayes

HyperParamater tuning - Randomized Grid search cross validation

Final Model results - Performance metrics

Display Best model - By comparing Performance

"""


def get_input_types(df):
    """Get the input types for each field in the DataFrame that corresponds
    to an input type to be fed into the model : ['text', 'categorical', 'numeric', 'datetime', 'ignore']
    A dict of {field_name: type} mappings.
    """
    fields = df.columns
    nrows = df.shape[0]
    avg_spaces = -1
    field_types = OrderedDict()

    for field in fields:
        # if field in col_types:
        #     field_types[field] = col_types[field]
        #     continue
        field_type = df[field].dtype
        num_unique_values = df[field].nunique()
        if field_type == 'object':
            avg_spaces = df[field].str.count(' ').mean()

        # Automatically ignore `id`-related fields
        if field.lower() in ['id', 'uuid', 'guid', 'pk', 'name']:
            field_types[field] = 'ignore'

        # Datetime is a straightforward data type.
        elif field_type == 'datetime64[ns]':
            field_types[field] = 'datetime'

        # Assume a float is always numeric.
        elif field_type == 'float64':
            field_types[field] = 'numeric'

        # If it's an object where the contents has
        # many spaces on average, it's text
        elif field_type == 'object' and avg_spaces >= 2.0:
            field_types[field] = 'text'

        # If the field has very few distinct values, it's categorical
        elif num_unique_values <= 10:
            field_types[field] = 'categorical'

        # If the field has many distinct integers, assume numeric.
        elif field_type == 'int64':
            field_types[field] = 'numeric'

        # If the field has many distinct nonintegers, it's not helpful.
        elif num_unique_values > 0.9 * nrows:
            field_types[field] = 'ignore'

        # The rest (e.g. bool) is categorical
        else:
            field_types[field] = 'categorical'

    # Print to console for user-level debugging
    print("Modeling with field specifications:")
    print("\n".join(["{}: {}".format(k, v) for k, v in field_types.items()]))
    print('\n')
    return field_types


def identify_problem_type(df, field_types,target):
    ftype = field_types[target]
    if ftype=='numeric':
        return 'regression'
    elif ftype == 'categorical':
        return 'classification'
    else:
        return 'improper_target'


def normalize_col_names(df,input_types):
    """Fixes unusual column names (e.g. Caps, Spaces)
    """
    pattern = re.compile('\W+')
    fields = [(re.sub(pattern, '_', field.lower()), field, field_type)
                   for field, field_type in input_types.items()]
    col_names = {row[1]:row[0] for row in fields}
    df.rename(columns=col_names,inplace=True)
    print("column name changes")
    print("\n".join(["{}: {}".format(k, v) for k, v in col_names.items()]))
    field_types = {row[0]:row[2] for row in fields}
    print('\n\n')
    return field_types


def proper_dataframe(df,field_types,target,problem_type):
    #   if problem_type=='regression':
    sel_field_types = {key:value for key,value in field_types.items() if value=='numeric' and key !=target}
    selected_cols = [key for key,value in field_types.items() if value=='numeric' and key !=target]
    sel_field_types[target] = field_types[target]
    selected_cols.append(target)
    data = df[selected_cols].copy()
    return (data,sel_field_types)

    # elif problem_type=='classification':
    #     sel_field_types = {key:value for key,value in field_types.items() if value=='categorical' or  value=='numeric'}
    #     selected_cols = [key for key,value in field_types.items() if  value=='numeric']
    #     data = df[selected_cols].copy()
    #     return (data,sel_field_types)


def handle_null_values(df):
    drop_cols=[]
    no_null = True
    for col in df.columns:
        null_cnt = df[col].isnull().sum()
        if 0<float(null_cnt)/ float(df.shape[0]) <0.1:
            medi = df[col].median()
            df[col]= df[col].apply(lambda x : medi if x==None else x)
            print(f"Replaced {null_cnt} values in {col} with median {medi}")
            no_null = False
        elif  null_cnt==0:
            no_null = no_null and True
        else:
            drop_cols.append(col)
    if no_null:
        print("No null values found")
    else:
        if len(drop_cols)>0:
            print("Dropped {len(drop_cols)} columns : {drop_cols}")
        else:
            print("No columns dropped")
    return (df.drop(columns=drop_cols),drop_cols)


def normalisation(df,target,field_types):
    cols = [key for key,value in field_types.items() if value=='numeric' and key!=target]
    rest_cols = [key for key,value in field_types.items() if value!='numeric']
    rest_cols.append(target)
    data = df[cols].to_numpy()
    if data.shape[1] !=0:
        scaler = StandardScaler().fit(data)
        data_scaled = scaler.transform(data)
        scaled_df = pd.DataFrame(data=data_scaled,columns=cols)
        for col in rest_cols:
            scaled_df[col]=df[col].copy()
        return scaled_df
    else:
        return df


def remove_outliers(df,target,field_types):
    cols = [key for key,value in field_types.items() if value=='numeric' and key!=target]
    rem_rows = []
    for col  in cols:
        Q1 = np.percentile(df[col], 25,interpolation = 'midpoint')
        Q3 = np.percentile(df[col], 75,interpolation = 'midpoint')
        IQR = Q3 - Q1
        upper = np.where(df[col] >= (Q3+1.5*IQR))
        lower = np.where(df[col] <= (Q1-1.5*IQR))
        rem_rows+=list(upper[0])
        rem_rows+=list(lower[0])
    rem_rows= list(set(rem_rows))
    df.drop(rem_rows, inplace = True,axis=0)
    df.reset_index(drop=True,inplace=True)
    return len(rem_rows)


def pca_df(df, target,field_types):
    X = df.drop(target,axis=1)
    all_numeric = True
    for key,value in field_types.items():
        all_numeric = all_numeric and value=='numeric'
        if not all_numeric:
            break
    if X.shape[1]>10 and all_numeric:
        principal=PCA(n_components=0.95)
        principal.fit(X)
        n = len(principal.explained_variance_ratio_)
        pca_x = PCA(n_components=n)
        pca_x.fit(X)
        X_pca = pca_x.transform(X)
        pca_df = pd.DataFrame(data=X_pca,columns= list(df.columns).remove(target))
        pca_df[target]=df[target].copy()
        return (pca_df, n)
    else:
        return (df,0)


def save_final_df(df,name,path):
    Path(path).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{path}pre_processed_{name}.csv")

def main():
    #testing code
    print()

if __name__ == '__main__':
    main()
