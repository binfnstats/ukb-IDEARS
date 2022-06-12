import os

import sys


sys.path.append("../../../ukb-dementia-shap/")


sys.path.append("../Pain/code/")
sys.path.append("../../Pain/code/")
from logic.data_processing.data_import import dataload
from logic.data_processing.data_processing import data_proc_main
from logic.analysis.analysis import AnalysisCharts
from logic.ml.classification_shap import IDEARs_funcs
from ukb_utils.utils import basic_funcs

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.offsets import DateOffset

import pandas as pd
import numpy as np
from scipy import stats
import re

ac=dataload()
dp=data_proc_main()
ml=IDEARs_funcs()
an=AnalysisCharts()



class data_setup():

    """
    This class is to extract key information from XMLDoc text and return as a set of dataframes
    """

    def __init__(self) -> None:


        self.pain_dic=dict({'Not pain-related':'Non Pain','Pain-related - New category "Arthritis-pain"':'Arth'})
        self.icd10s=dp.ukb_icd10()
        self.ukb_file='../../data/ukb50790.tab'

        #self.field_name_file=pd.read_csv('../../data/ukb_field_names.csv')
        self.field_name_file=pd.read_csv('/Users/michaelallwright/Documents/GitHub/UKB/data/ukb_field_names.csv')
        #self.field_name_data_dic=pd.read_csv('../../data/data_dic.csv')
        self.field_name_data_dic=pd.read_csv('/Users/michaelallwright/Documents/GitHub/UKB/data/data_dic.csv')

        self.code_map=pd.read_csv('/Users/michaelallwright/Documents/GitHub/UKB/data/code_map2.csv')

        self.compvars=['glycated_haemoglobin_hba1c_f30750_0_0','neutrophill_count_f30140_0_0',
          'creactive_protein_f30710_0_0','NLR','lymphocyte_count_f30120_0_0','monocyte_count_f30130_0_0',
          'basophill_count_f30160_0_0']

        self.varmap=None

    def count_nulls(self,df):
        cols=[]
        non_nulls=[]
        vals_all=[]
        for c in df.columns:
            if c!='eid':
                cols.append(c)
                ns=df[c][pd.notnull(df[c])].shape[0]
                non_nulls.append(ns)
                vals=dict(df[c].value_counts())
                vals_all.append(vals)

        df2=pd.DataFrame({'Column':cols,'non_nulls':non_nulls,'vals_all':vals_all})
        df2.sort_values(by='non_nulls',ascending=False,inplace=True)
        return df2

    def search_icd(self,strings='chronic pain',second_string='',non_strings='xxxxx',string_pat=True):
        mask=(self.icd10s['disease'].str.contains(strings,regex=True))&(~self.icd10s['disease'].str.contains(non_strings,regex=True))&\
    (self.icd10s['disease'].str.contains(second_string,regex=True))
        icd10_sub=list(self.icd10s.loc[mask,'code'])
        icd_df=self.icd10s.loc[mask,]
        
        if string_pat:
            icd10_sub='|'.join(icd10_sub)
            
        return icd10_sub,icd_df

    def sep_path(self,string,i=1):
        try:
            s=string.split('>')[i]
        except:
            s='N/A'
        return s

    def map_codes(self):
        code_map=self.code_map.copy()
        code_names=[]
        code_dicts=[]
        for c in code_map['Coding'].unique():
            df1=code_map.loc[code_map['Coding']==c,]
            dict1=dict(zip(df1['Value'],df1['Meaning']))
            code_names.append(c)
            code_dicts.append(dict1)
        code_mappings=pd.DataFrame({'code':code_names,'dicts':code_dicts})

        self.code_mappings=code_mappings

        return code_mappings

    def make_dicts(self):
        field_names=self.field_name_file.copy()
        data_dic=self.field_name_data_dic.copy()


        field_names['FieldID']=field_names['field.showcase'].astype(str)
        field_names['FieldID']=field_names['field.tab'].apply(lambda x:x.split('.')[1])
       
        data_dic['FieldID']=data_dic['FieldID'].astype(str)
        field_names2=pd.merge(field_names,data_dic,on='FieldID',how='left')

        field_names2['Path']=field_names2['Path'].astype(str)
        field_names2['field_type0']=field_names2.apply(lambda x:self.sep_path(x['Path'],0),axis=1)
        field_names2['field_type1']=field_names2.apply(lambda x:self.sep_path(x['Path'],1),axis=1)
        field_names2['field_type2']=field_names2.apply(lambda x:self.sep_path(x['Path'],2),axis=1)
        field_names2['field_type3']=field_names2.apply(lambda x:self.sep_path(x['Path'],3),axis=1)

        field_names2.loc[pd.notnull(field_names2['Coding']),'Coding']=\
    field_names2.loc[pd.notnull(field_names2['Coding']),'Coding'].astype(int)

        varmap=dict(zip(field_names['field.tab'],field_names['col.name']))

        colsnew=list(field_names.groupby('field.showcase').first()['field.tab'])
        colsnew2=list(field_names.groupby('field.showcase').first()['col.name'])

        colsnew_tp1=list(field_names.groupby('field.showcase').nth(1)['field.tab'])
        colsnew2_tp1=list(field_names.groupby('field.showcase').nth(1)['col.name'])

        self.varmap=varmap
        self.colsnew=colsnew

        dict_maps=dict({'varmap':varmap,'colsnew':colsnew,'colsnew2':colsnew2,'colsnew_tp1':colsnew_tp1,
                'colsnew2_tp1':colsnew2_tp1,'field_names2':field_names2})

        return dict_maps

    def make_parquet(self,cols,outfile='df_pain_ukb',parq_out=False):

        if self.varmap is None:
            print("getting apps time")
            self.make_dicts()

        cols=[c for c in self.colsnew if self.varmap[c] in cols]

        df=pd.read_csv(self.ukb_file,sep='\t',usecols=cols)
        df.columns=[self.varmap[c] for c in df.columns]

        if parq_out:
            df.to_parquet('../../data/'+outfile+'.parquet')

        return df



    def process_run(self,df,depvar='neuropathy',resize=1,remwords='xxxxxxx',resizeratio=20):
        df=ml.col_spec_chars(df=df)
        df=df.loc[pd.notnull(df[depvar]),]

        dropvars=list(set([c for c in df.columns if  re.search(ml.wordsremovePD,c)]+[c for c in df.columns if  re.search(remwords,c)]))

        shap_tuple=ml.run_entire_data_pd(df=df,drops=dropvars,wordsremove='consultant',outfile='test_pain',savefile=False,
        save_featslist=False,runs=2,holdout_ratio=0.5,depvar=depvar,agemin=10,agemax=90,resize=resize,resizeratio=resizeratio,verbose=False)
        
        return shap_tuple

    def ttest(self,df,var,depvar='polyneuropathy'):
    
        df1=df.loc[pd.notnull(df[var]),[var,depvar]]
        ttest_vals=stats.ttest_ind(df1[(df1[depvar]==1)][var],df1[(df1[depvar]==0)][var])

        return ttest_vals

    def runplots_static(self,df,depvar='poly_chronic',fig_name='diabetes_inflamm_polychronicpain'):
        k=len(self.compvars)
        fig = plt.figure(figsize=(25, 10*k))
        grid = plt.GridSpec(k, 2, hspace=0.45, wspace=0.3)

        for j,v in enumerate(self.compvars):
            for i in range(2):
                ax=fig.add_subplot(grid[j, i])

                df_diab2_use=df.loc[df['sex_f31_0_0']==i,]
                ax=sns.boxplot(x=df_diab2_use[depvar],y=df_diab2_use[v],showfliers = False,color='skyblue')
                plt.xticks(fontsize='35')
                plt.yticks(fontsize='35')
                plt.title(str(ml.mapvar(v)), fontsize='35')

                pval=str(round(list(self.ttest(df_diab2_use,v,depvar))[1],5))
                rangevars=df_diab2_use[v].quantile(0.75)-df_diab2_use[v].quantile(0.25)
                plt.text(0,rangevars,'p value '+pval,fontsize=24)


        plt.savefig(an.path_figures_pain+'fig_'+an.date_run+"_"+fig_name+'.jpg', dpi=300,bbox_inches='tight')
        plt.show()
        
        return None

    def pain_base(self,field='troubled_by_pain_or_discomfort_present_for_more_than_3_months_f120019_0_0'):
        df=pd.read_parquet('../../data/df_pain_ukb.parquet')
        mask=(df[field]==1)
        pain1_eids=list(df.loc[mask,'eid'])

        return pain1_eids

    def basic_diab_df(self):
        icd10_diabs=self.search_icd(strings='diabetes',non_strings='family|screening|insipidus|pregnancy',string_pat=True)[0]
        df_diab=dp.data_merge_dis(remwords='xxxxx',disease='diabetes',icd10s=icd10_diabs,outfile=None,use_icd10=True,
                 strcont=True,bef=True,years=0)
        df_diab=df_diab.loc[df_diab['diabetes']==1,]

        df_diab.drop(columns=['time_since_diabetes','diabetes'],inplace=True)
        df_diab['NLR']=df_diab['neutrophill_count_f30140_0_0']/df_diab['lymphocyte_count_f30120_0_0']

        return df_diab

    def basic_df_poly(self,bef=True,years=0,var_time='time_since_diab_poly'):

        icd10_diab_poly=self.search_icd(strings='diabetic polyneuropathy',
            non_strings='family|screening|insipidus|pregnancy',string_pat=True)[0]

        df_diab_poly=dp.data_merge_dis(remwords='xxxxx',disease='diab_poly',icd10s=icd10_diab_poly,outfile=None,use_icd10=True,
                 strcont=True,bef=bef,years=years)[['eid','diab_poly',var_time]]

        return df_diab_poly







    def diabetes_run(self,pain_field='troubled_by_pain_or_discomfort_present_for_more_than_3_months_f120019_0_0'):

        #preprocesses diabetes datasets to show polyneuropathy and polyneuropathy with pain at baseline

        df_diab=self.basic_diab_df( )
        #diabetic polyneuropathy baseline
        icd10_diab_poly=self.search_icd(strings='diabetic polyneuropathy',
            non_strings='family|screening|insipidus|pregnancy',string_pat=True)[0]

        df_diab_poly=dp.data_merge_dis(remwords='xxxxx',disease='diab_poly',icd10s=icd10_diab_poly,outfile=None,use_icd10=True,
                 strcont=True,bef=True,years=0)[['eid','diab_poly','time_since_diab_poly']]

        df_diab_poly_pros=dp.data_merge_dis(remwords='xxxxx',disease='diab_poly',icd10s=icd10_diab_poly,outfile=None,use_icd10=True,
                 strcont=True,bef=False,years=0)[['eid','diab_poly','time_to_diab_poly']]

        print('recs prosp diab poly',df_diab_poly_pros.loc[df_diab_poly_pros['diab_poly']==1,].shape[0])


        eids_diab_poly_base=list(df_diab_poly.loc[df_diab_poly['diab_poly']==1,'eid'].astype(str))

        print('num diab poly',len(eids_diab_poly_base))

        df_diab['polyneuropathy']=0
        df_diab.loc[df_diab['eid'].isin(eids_diab_poly_base),'polyneuropathy']=1

        df_diab_poly_base=df_diab.loc[df_diab['eid'].isin(eids_diab_poly_base),]


        pain1_eids=self.pain_base(field=pain_field)

        mask=df_diab_poly_base['eid'].isin(pain1_eids)
        df_diab_poly_base['poly_pain']=0
        df_diab_poly_base.loc[mask,'poly_pain']=1

        return df_diab_poly_base,df_diab

    def return_eids(self,df,string='polyneuropathy',icd10s=False,
                string_exc='family|screening|insipidus|pregnancy',years=2):
    

        mask_inc_snap=(df['disease_date']<df['date_assess'])
        mask_inc_pro=(df['disease_date']>=df['date_assess']+ DateOffset(years=years))
            
        df['dis_name_all']=df['disease_name_new']+' '+df['disease_name']
        
        if icd10s:
            mask_dis=(df['disease'].str.contains(string,regex=True))
        else:
            mask_dis=(df['dis_name_all'].str.contains(string,regex=True))&(~df['dis_name_all'].str.contains(string_exc,regex=True))
        
        mask_snap_inc=mask_inc_snap&mask_dis
        mask_pro_inc=mask_inc_pro&mask_dis
        mask_snap_exc=mask_dis&~mask_inc_snap
        mask_pro_exc=~mask_inc_pro&mask_dis
            
        eids_inc_snap=list(df.loc[mask_snap_inc,'eid'].unique())
        eids_inc_pro=list(df.loc[mask_pro_inc,'eid'].unique())
        
        eids_exc_snap=list(df.loc[mask_snap_exc,'eid'].unique())
        eids_exc_pro=list(df.loc[mask_pro_exc,'eid'].unique())
        
        disease_list=pd.DataFrame(df.loc[mask_dis,'dis_name_all'].value_counts())
        
        df_dict=dict({'eids_inc_snap':eids_inc_snap,'eids_inc_pro':eids_inc_pro,'eids_exc_snap':eids_exc_snap,'eids_exc_pro':eids_exc_pro,
                     'disease_list':disease_list})
       
    
        return df_dict

    def shapruns(self,run,df=df_diab,remwords='diabetes|H360|total_dis',depvar='polyneuropathy'):

        shap_obj=self.process_run(df=df,depvar=depvar,resize=1,resizeratio=5,remwords='diabetes|H360|total_dis')
        feats_all=ml.shapgraphs_tuple(shap_obj,max_disp=30,figname='SHAP IDEARS '+run)
        ml.ROCAUC_tuples(df_out_list=[shap_obj[2]],labels=['IDEARS - all'],cols=['blue'],figname='ROCAUC '+run)
        self.runplots_static(df=df,depvar=depvar,fig_name='Inflamm boxplots '+run)





