import pandas as pd
import numpy as np
import umap

df=pd.read_parquet('df_PD_ctrl_chrom_1_22.parquet',use_threads=False)


pd_files=[f for f in os.listdir()  if '.parquet' in f and '1_22' not in f and '_1_' not in f]

for i,f in enumerate(pd_files):
	print(f)
    
    
    df=pd.read_parquet(f)

    if i==0:
        df_full=df.copy()
    else:
        df_full=pd.merge(df_full,df,on='eid',how='inner')

print("imported")

embedding = umap.UMAP(n_neighbors=200,
                      min_dist=0.3,
                      metric='correlation',n_components=20).fit_transform(df_full)

print('embedded')