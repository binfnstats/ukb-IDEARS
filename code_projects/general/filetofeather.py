

import pandas as pd
import os


import sys
chunksize=20000
cutoff=50

import time

path="/data3/mike/"

df_check=pd.read_csv(path+'hamish_preprocessing.csv', nrows=10,sep=',')

colsnew=df_check.columns

print(len(colsnew))


"""
df = pd.read_csv(path+'hamish_preprocessing.csv', chunksize=chunksize,sep=',')


def create_feathers():
    for i in range(9):

        print(i)
        if i<8:

            mini=i*1000
            maxi=(i+1)*1000-1
        if i==8:
            mini=i*1000
            rem=len(colsnew) % 8000
            maxi=8000+rem


        cols=list(set(['f.eid']+colsnew[mini:maxi]))
        df=pd.read_csv(ukb_file,sep='\t',usecols=cols)
        df.columns=[varmap[c] for c in df.columns]
        df=exc_feather(df,outfile=path+str(i)+'.feather')

def exc_feather(df,outfile='../data/ukb'):

    try:
        df.to_parquet(outfile+'.feather')

    except:
        print('running exception')
        for c in df.columns:
            if df[df[c].apply(lambda x: isinstance(x, str))].shape[0]>0:
                df[c]=df[c].astype(str)
        df.to_feather(outfile+'.feather') 
        
    return df




#def install(package):
 #   subprocess.check_call([sys.executable, "-m", "pip", "install", package])




i=0
for chunk in df:
    i=i+1

    start = time.time()
    print(i*chunksize)
    df=pd.DataFrame(chunk).reset_index()

    df.to_feather(path+'ukb'+str(i)+'.feather')

"""   


