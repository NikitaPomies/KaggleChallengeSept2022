import numpy as np


def check_duplicates(df, delete = False):
    """
    Print the number of duplicates for each lines
    if delete is set to True, the duplicates are deleted.
    """
    print(f" Number of duplicates rows : {df.duplicated().sum()}, ({np.round(100*df.duplicated().sum()/len(df),1)}%)")

    if delete :
        df.drop_duplicates(inplace = True)


def remove_unusable_features(df_train, df_test, features):

    """
    Remove  categorical features where train and test dataframe have different unique values
    """
    for feat in features :
        if len(set(df_train[feat].unique()) ^ set(df_test[feat].unique()) )> 0 :
            print(f"cat. features {feat} takes different values in train and test set \n "
                  f" -> {feat} deleted \n")

            print(df_train[feat].unique(), df_test[feat].unique() ,"\n")
            del df_train[feat]
            del df_test[feat]
