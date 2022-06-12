import os

import sys


sys.path.append("../../../ukb-dementia-shap/")


sys.path.append("../Pain/code/")
from logic.data_processing.data_import import dataload
from logic.data_processing.data_processing import data_proc_main
from logic.analysis.analysis import AnalysisCharts
from logic.ml.classification_shap import IDEARs_funcs
from ukb_utils.utils import basic_funcs

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.offsets import DateOffset
import datetime as dt

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

        self.date_suff=dt.datetime.today().strftime('%Y-%m-%d')
        self.pain_dic=dict({'Not pain-related':'Non Pain','Pain-related - New category "Arthritis-pain"':'Arth'})
        self.icd10s=dp.ukb_icd10()
        self.ukb_file='../../data/ukb50790.tab'

        #self.field_name_file=pd.read_csv('../../data/ukb_field_names.csv')
        self.field_name_file=pd.read_csv('/Users/michaelallwright/Documents/GitHub/UKB/data/ukb_field_names.csv')
        #self.field_name_data_dic=pd.read_csv('../../data/data_dic.csv')
        self.field_name_data_dic=pd.read_csv('/Users/michaelallwright/Documents/GitHub/UKB/data/data_dic.csv')

        self.field_name_full_file=pd.read_excel('/Users/michaelallwright/Documents/GitHub/UKB/data/ukb_field_names.xlsx',
            sheet_name='fieldnames_full')

        self.varmap=dict(zip(self.field_name_full_file["col.name"],self.field_name_full_file["Field"]))

        self.code_map=pd.read_csv('/Users/michaelallwright/Documents/GitHub/UKB/data/code_map2.csv')

        self.compvars=['glycated_haemoglobin_hba1c_f30750_0_0','cystatin_c_f30720_0_0','neutrophill_count_f30140_0_0',
          'creactive_protein_f30710_0_0','neutrophill_lymphocyte_ratio','lymphocyte_count_f30120_0_0',
          'monocyte_count_f30130_0_0','basophill_count_f30160_0_0']
        self.compvars_perc=['neutrophill_percentage_f30200_0_0','lymphocyte_percentage_f30180_0_0',
        'monocyte_percentage_f30190_0_0','basophill_percentage_f30220_0_0']

        #self.varmap=None

    def count_nulls(self,df):

        # returns a dataframe with the number of nulls and non nulls for each field as well as a list of values and counts
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

    def varnorm_mult(self,df,normvars,bd_var):
        bdowns=list(df[bd_var].unique())

        #for i,b in enumerate(bdowns):
        dfs=[df.loc[(df[bd_var]==b),] for b in bdowns]
        df_sums=[pd.DataFrame(df.groupby(normvars).size()).reset_index() for df in dfs]

        for i,df in enumerate(df_sums):
            df.columns=normvars+['recs_'+str(i)]

        df_out=df_sums[0].copy()
        for i in range(len(bdowns)-1):
            df_out=pd.merge(df_out,df_sums[i+1],on=normvars,how='inner')

        return df_out

    def agenorm(self,df,var):
        df_sum=pd.DataFrame(df.groupby(['age_when_attended_assessment_centre_f21003_0_0']).agg({var:['mean']})).reset_index()
        df_sum.columns=['age_when_attended_assessment_centre_f21003_0_0','mean'+var]

        df=pd.merge(df,df_sum,on='age_when_attended_assessment_centre_f21003_0_0',how='left')

        df[var]=df[var].mean()*df[var]/df['mean'+var]
        df.drop(columns=['mean'+var],inplace=True)
        return df

    def varnorm1(self,df,normvars,depvar):

        # rebalances dataframe to be equal across case and control as defined by depvar=1/0 across a list of variables which must be present in the data
        df1=df.copy()

        mask=(df[depvar]==1)
        
        df_case=df1.loc[mask,]
        df_ctrl=df1.loc[~mask,]


        cases=pd.DataFrame(df_case.groupby(normvars).size()).reset_index()
        ctrls=pd.DataFrame(df_ctrl.groupby(normvars).size()).reset_index()

        ctrls.columns=normvars+['ctrl_recs']
        cases.columns=normvars+['case_recs']
        ctrl_case=pd.merge(cases,ctrls,on=normvars,how='inner')
        ctrl_case['ratio']=(ctrl_case['ctrl_recs']/ctrl_case['case_recs'])
        
        max_mult=ctrl_case['ratio'].min()
        
        ctrl_case['case_samp']=max_mult

        return ctrl_case,df_ctrl,df_case,cases

    def varnorm(self,df,normvars,depvar,max_mult=None):

        # rebalances dataframe to be equal across case and control as defined by depvar=1/0 across a list of variables which must be present in the data
        df1=df.copy()

        mask=(df[depvar]==1)
        
        df_case=df1.loc[mask,]
        df_ctrl=df1.loc[~mask,]


        cases=pd.DataFrame(df_case.groupby(normvars).size()).reset_index()
        ctrls=pd.DataFrame(df_ctrl.groupby(normvars).size()).reset_index()

        ctrls.columns=normvars+['ctrl_recs']
        cases.columns=normvars+['case_recs']
        ctrl_case=pd.merge(cases,ctrls,on=normvars,how='inner')
        ctrl_case['ratio']=(ctrl_case['ctrl_recs']/ctrl_case['case_recs'])
        
        if max_mult==None:
            max_mult=ctrl_case['ratio'].min()
        
        ctrl_case['case_samp']=max_mult



        df_ctrl=pd.merge(df_ctrl,ctrl_case,on=normvars)

        grouped = df_ctrl.groupby(normvars, group_keys=False)
        df_ctrl=df_ctrl.loc[grouped.apply(lambda x: x.sample((x['case_samp']*x['case_recs']).astype(int).iloc[0])).index,]
        df_ctrl.drop(columns=['ctrl_recs','ratio','case_samp','case_recs'],inplace=True)
        df_out=pd.concat([df_ctrl,df_case],axis=0)
        #df_out=df_out.reset_index()
        
        return df_out
        

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

    def data_clean(self,df,depvar='neuropathy',remwords='xxxxxxx'):

        df=ml.col_spec_chars(df=df)

        
        df=df.loc[pd.notnull(df[depvar]),]

        if depvar=='AD':
            dropvars=list(set([c for c in df.columns if  re.search(ml.wordsremoveAD,c)]))

        else:
            dropvars=list(set([c for c in df.columns if  re.search(ml.wordsremovePD,c)]+
                [c for c in df.columns if  re.search(remwords,c)]))

        return df,dropvars



    def process_run(self,df,depvar='neuropathy',resize=1,remwords='xxxxxxx',resizeratio=20,runs=2,holdout_ratio=0.5,
        df_val_use=None,preprocess=True):

        df,dropvars=self.data_clean(df,depvar=depvar,remwords=remwords)
        
        shap_tuple=ml.run_entire_data_pd(df=df,drops=dropvars,wordsremove=remwords,outfile='test_pain',savefile=False,
        save_featslist=False,runs=runs,depvar=depvar,agemin=10,agemax=90,resize=resize,holdout_ratio=holdout_ratio,
        resizeratio=resizeratio,verbose=False,df_val_use=df_val_use,preprocess=preprocess)
        
        return shap_tuple

    def ttest(self,df,var,depvar='polyneuropathy'):
    
        df1=df.loc[pd.notnull(df[var]),[var,depvar]]
        ttest_vals=stats.ttest_ind(df1[(df1[depvar]==1)][var],df1[(df1[depvar]==0)][var])

        return ttest_vals

    def runplots_static(self,df,depvar='poly_chronic',
        fig_name='diabetes_inflamm_polychronicpain',perc=True,compvars=None,agenormvars=[],savefig=True,pltshow=True,
        splitvar='sex_f31_0_0',labels=dict({1:'Female',0:'Male'}),
        normvars=['age_when_attended_assessment_centre_f21003_0_0','sex_f31_0_0']):

        df=df.copy()
        if compvars==None:

            if perc:
                compvars=self.compvars+self.compvars_perc
            else:
                compvars=self.compvars

        for a in agenormvars:
            df=an.varnorm(df,a)


      

        k=len(compvars)
        fig = plt.figure(figsize=(25, 10*k))
        grid = plt.GridSpec(k, 2, hspace=0.45, wspace=0.3)

        splitvars=list(set(list(df.loc[pd.notnull(df[splitvar]),splitvar].unique())))

        compvars_use=[]
        pvals=[]
        genders=[]
        vals_case=[]
        vals_std_case=[]
        vals_ctrl=[]
        vals_std_ctrl=[]

        for j,v in enumerate(compvars):
            for i,x in enumerate(splitvars):

                if pltshow:
                    ax=fig.add_subplot(grid[j, i])

                df_diab2_use=df.loc[df[splitvar]==x,]

               

                if v in self.varmap:
                    title=str(self.varmap[v])
                else:
                    title=str(ml.mapvar(v))


               

                pval=str(round(list(self.ttest(df_diab2_use,v,depvar))[1],7))
                rangevars=df_diab2_use[v].quantile(0.75)-df_diab2_use[v].quantile(0.25)

                if pltshow:
                    ax=sns.boxplot(x=df_diab2_use[depvar],y=df_diab2_use[v],showfliers = False,color='skyblue')
                    plt.xticks(fontsize='35')
                    plt.yticks(fontsize='35')
                    plt.title((title), fontsize='35')
                    plt.text(0,rangevars,'p value '+pval,fontsize=24)
                    plt.title(labels[x]+'s: '+str(ml.mapvar(v)), fontsize='35')

                mask=(df_diab2_use[depvar]==1)
                mean_val_case=df_diab2_use.loc[mask,v].mean()
                std_case=df_diab2_use.loc[mask,v].std()
                std_case=str(round(mean_val_case,2))+' +/- '+str(round(std_case,2))
                mean_val_ctrl=df_diab2_use.loc[~mask,v].mean()
                std_ctrl=df_diab2_use.loc[~mask,v].std()
                std_ctrl=str(round(mean_val_ctrl,2))+' +/- '+str(round(std_ctrl,2))



                compvars_use.append(v)
                genders.append(i)
                pvals.append(pval)
                vals_case.append(mean_val_case)
                vals_ctrl.append(mean_val_ctrl)

                vals_std_case.append(std_case)
                vals_std_ctrl.append(std_ctrl)


        genders=['Male' if c==0 else 'Female' for c in genders]
        compvars_use=[self.varmap[c] if c in self.varmap else ml.mapvar(c) for c in compvars_use]


        df_out=pd.DataFrame({'Variable':compvars_use,splitvar:genders,'case value':vals_case,'case_vals_std':vals_std_case,
            'control value':vals_ctrl,'ctrl_vals_std':vals_std_ctrl,'p-value':pvals})

        try:
            df_out=df_out.pivot(index='Variable',columns=splitvar,values=['ctrl_vals_std','case_vals_std','p-value'])
        except:
            pass

        if savefig:
            plt.savefig(fig_name+'_'+an.date_run+'.jpg', dpi=300,bbox_inches='tight')
        if pltshow:
            plt.show()
        
        return df_out

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
    


        df['dis_name_all']=df['disease_name_new']+' '+df['disease_name']


        if icd10s:
            mask_dis=(df['disease'].str.contains(string,regex=True))
        else:
            mask_dis=(df['dis_name_all'].str.contains(string,regex=True))&(~df['dis_name_all'].str.contains(string_exc,regex=True))



        df['dis']=0
        df.loc[mask_dis,'dis']=1


        df_cases=df.loc[mask_dis,]
        mask_exc=~(df['eid'].isin(df_cases['eid']))
        df_ctrls=df.loc[mask_exc,]

    
        cases=pd.DataFrame(df_cases.groupby(['eid']).agg({'disease_date':'min','date_assess':'min'})).reset_index()
        ctrls=pd.DataFrame(df_ctrls.groupby(['eid']).agg({'disease_date':'min','date_assess':'min','death':'max'})).reset_index()

        cases['eid']=cases['eid'].astype(str)
        ctrls['eid']=ctrls['eid'].astype(str)

        mask_inc_snap=(cases['disease_date']<cases['date_assess'])
        mask_inc_pro=(cases['disease_date']>=cases['date_assess']+ DateOffset(years=years))

        cases_inc_pro=list(cases.loc[mask_inc_pro,'eid'].astype(str).unique())
        cases_inc_snap=list(cases.loc[mask_inc_snap,'eid'].astype(str).unique())

        cases_exc_pro=list(cases.loc[~mask_inc_pro,'eid'].astype(str).unique())
        cases_exc_snap=list(cases.loc[~mask_inc_snap,'eid'].astype(str).unique())
        
        mask_death=(ctrls['death']==1)
        ctrls_exc_pro=list(ctrls.loc[mask_death,'eid'].astype(str).unique())
        
        #eids to exclude
        eids_exc_snap=cases_exc_snap
        eids_exc_pro=cases_exc_pro+ctrls_exc_pro
        
        disease_list=pd.DataFrame(df.loc[mask_dis,'dis_name_all'].value_counts())
        df_dict=dict({'eids_inc_snap':cases_inc_snap,'eids_inc_pro':cases_inc_pro,'eids_exc_snap':eids_exc_snap,
            'eids_exc_pro':eids_exc_pro,'disease_list':disease_list,'cases':cases,'ctrl_deaths':ctrls_exc_pro})
       
    
        return df_dict


    def shapruns(self,run,df,remwords='diabetes|H360|total_dis',depvar='polyneuropathy',resizeratio=5,resize=1,perc=False,
        compvars=None,stream=False,runs=2,holdout_ratio=0.2):
        
        shap_obj=self.process_run(df=df,depvar=depvar,resize=resize,resizeratio=resizeratio,remwords=remwords,runs=runs,
            holdout_ratio=holdout_ratio)
        feats_all=ml.shapgraphs_tuple(shap_obj,max_disp=30,figname='SHAP IDEARS '+run+self.date_suff,stream=stream)
        ml.ROCAUC_tuples(df_out_list=[shap_obj[2]],labels=['IDEARS - all'],cols=['blue'],figname='ROCAUC '+run+self.date_suff,stream=stream)
        self.runplots_static(df=df,depvar=depvar,fig_name='Inflamm boxplots '+run+self.date_suff,perc=perc,compvars=compvars)

        return feats_all,shap_obj

    def shapruns_new(self,run,df,remwords='diabetes|H360|total_dis',depvar='polyneuropathy',resizeratio=5,resize=1,perc=False,
        compvars=None,stream=False,runs=2,barplots=1,holdout_ratio=0.5,df_val_use=None,preprocess=True):
        
        shap_obj=self.process_run(df=df,depvar=depvar,resize=resize,resizeratio=resizeratio,remwords=remwords,runs=runs,
            holdout_ratio=holdout_ratio,df_val_use=df_val_use,preprocess=preprocess)

        print(len(shap_obj))
        feats_all=ml.shapgraphs_tuple(shap_obj,max_disp=30,figname='SHAP IDEARS '+run+self.date_suff,stream=stream)
        aucs=ml.ROCAUC_tuples(df_out_list=[shap_obj[2]],labels=['IDEARS - all'],cols=['blue'],figname='ROCAUC '+run+self.date_suff,stream=stream)

        rets=dict({'feats_all':feats_all,'shaps':shap_obj,'aucs':aucs})
        if barplots==1:
            data_sum=self.runplots_static(df=df,depvar=depvar,fig_name='Inflamm boxplots '+run+self.date_suff,perc=perc,compvars=compvars)
            rets['data_sum']=data_sum
        return rets







