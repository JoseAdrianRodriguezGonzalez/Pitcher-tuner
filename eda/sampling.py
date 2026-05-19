import json
import os 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import ast 
from scipy.stats import entropy
import sounddevice as sd 
import soundfile as sf
def create_folder(src):
    os.makedirs(src,exist_ok=True)
def open_json(src):
    with open(src) as f:
        data = json.load(f)
    return data 
def to_df(data):
    return pd.DataFrame.from_dict(data,orient="index")
def pitch_entropy(group):
    return entropy(group['pitch'].value_counts(normalize=True))
def pipeline():
    train=open_json("data/nsynth-train/examples.json") 
    train_df=to_df(train)
    create_folder("processed")
    train_df.to_csv("processed/train_df.csv")
    print(train_df.head())
    print(train_df.describe())
    #analyzing pitch 
    print(train_df["pitch"].describe())
    train_df["pitch"].hist(bins=50)
    
    #analyuzing instrument 
    print(train_df["instrument_family_str"].value_counts())
    sns.countplot(data=train_df,x='instrument_family_str')
    #pitch instrument family 
    plt.figure(figsize=(12,6))
    sns.boxplot(data=train_df,x='instrument_family_str',y='pitch')
    plt.xticks(rotation=45)

    #pitch velocity
    sns.scatterplot(data=train_df,x='velocity',y='pitch',alpha=0.3)
    #Analyzing qualities 
    qualities=train_df["qualities"]
    qualities_df=pd.DataFrame(qualities.tolist(),columns=[f"q{i}" for i in range(10)])
    df=pd.concat([train_df,qualities_df],axis=1)

    #sums 
    qualities_df.sum().sort_values(ascending=False)
    df_numeric = df.select_dtypes(include='number')
    df_numeric.corr()['pitch'].sort_values()
    print((df['pitch'] == df['note']).all())

    print(train_df.groupby('instrument_family_str')['pitch'].agg(['mean','std','min','max']))
    print(df.groupby('instrument_family_str').apply(pitch_entropy))
    plt.show()
    df_train_for_process=train_df[["note_str","pitch","instrument_family_str"]]
    
    val=open_json("data/nsynth-valid/examples.json") 
    df_val=to_df(val)
    test=open_json("data/nsynth-test/examples.json") 
    df_test=to_df(test)

    df_val_for_process=df_val[["note_str","pitch","instrument_family_str"]]
    df_test_for_process=df_test[["note_str","pitch","instrument_family_str"]]
    
    df_train_for_process.to_csv("train_df.csv",index=False)
    df_test_for_process.to_csv("test_df.csv",index=False)
    df_val_for_process.to_csv("val_df.csv",index=False)
    #####Creating the datasetin csv 
    
