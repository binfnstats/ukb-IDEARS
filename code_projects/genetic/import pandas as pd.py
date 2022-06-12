import pandas as pd
import numpy as np
import umap

df=pd.read_parquet('df_PD_ctrl_chrom_1_22.parquet')

print("imported")

embedding = umap.UMAP(n_neighbors=200,
                      min_dist=0.3,
                      metric='correlation',n_components=20).fit_transform(df_train)

print('embedded')