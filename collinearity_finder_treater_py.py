# %% [code]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:51:57 2020
This library can be used to find clusters of collinear variable in a larger data set and replace them with principal component(s) of the cluster. There is option to visualize the collinear relationship within a cluster.

@author: Nasseh Khodaie
"""
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA

class cluster():
    def __init__(self, pairs=None):
        '''
        Pairs = list of tuple e.g. [(node1,node2, weight1),(node3,node4,weight2)]
        '''
        self.pairs = set()
        self.nodes = set()
        self.name = None
        if pairs != None:
            for pair in pairs:
                self.nodes.update([pair[0],pair[1]])
                self.pairs.add(pair)

    def update_with(self, pair, force_update = False):
        if force_update:
            self.nodes.update([pair[0],pair[1]])
            self.pairs.add(pair)
        else:    
            if self.can_accept(pair):
                self.nodes.update([pair[0],pair[1]])
                self.pairs.add(pair)
            else:
                raise Exception(f'The pair {pair} can not be added to this cluster because it does not have any shared node with the current cluster nodes.')
        
    def can_accept(self, pair):
        return (pair[0] in self.nodes or pair[1] in self.nodes)
    
    def merge_with_cluster(self, cluster2, force_merge = False):
        def merge():
            self.nodes = self.nodes.union(cluster2.nodes)
            self.pairs = self.pairs.union(cluster2.pairs)
        
        if force_merge:
            merge()
        else:    
            if self.nodes.intersection(cluster2.nodes) != set():
                merge()
            else:
                raise Exception(f'The clusters can not be merged because they do not have any common node.')

    def plot(self, fig_size = (10,10), dpi= 200, max_line_width = 5, min_line_width = 1, min_alpha = 0.2, threshold=None, font_size=20):
        if threshold == None:
            threshold = min([pair[2] for pair in self.pairs])
        graph_pairs = [(pair[0],pair[1]) for pair in self.pairs]
        graph = nx.Graph()
        plt.figure(figsize = fig_size, dpi = dpi)
        graph.add_edges_from(graph_pairs)
        pos = nx.spring_layout(graph)

        for pair in self.pairs:
            nx.draw_networkx_nodes(graph, pos, node_size=300, edgecolors = (0.3,0.3,0.3), linewidths = 1, node_color = '#FFF')

            alph = (abs(pair[2])-threshold)/(1-threshold)
            if alph < min_alpha:
                alph = min_alpha
#            print(pair[2])

            w = (abs(pair[2])-threshold)/(1-threshold)*max_line_width
            if w < min_line_width:
                w = min_line_width

            color = 'b'    
            if pair[2]<0:
                color = 'r'

            nx.draw_networkx_edges(graph, pos, edge_color = color, edgelist = [[pair[0],pair[1]]], width = w, alpha = alph )
            nx.draw_networkx_labels(graph, pos, font_size=font_size, font_weight = 'bold', font_family='sans-serif', font_color=(0,0,0), alpha = 0.9)
            
        plt.title(self.name)
        plt.show()
       

def identify_cluster(X_data_df, threshold = 0.4, correlation_id_method = 'pearson'):
    cor = X_data_df.corr(method = correlation_id_method)

    #print(threshold)

    clusters = []
    for j,col in enumerate (cor.columns):
        for i,row in enumerate (cor.columns[0:j]):
            if abs(cor.iloc[i,j])>threshold:
                current_pair = (col,row, cor.iloc[i,j])
                current_pair_added = False
                for _c in clusters:
                    if _c.can_accept(current_pair):
                        _c.update_with(current_pair)
                        current_pair_added = True
                if current_pair_added == False:
                    clusters.append(cluster(pairs = [current_pair]))
    final_clusters = []
    for _cluster in clusters:
        added_to_final = False
        for final_c in final_clusters:
            if _cluster.nodes.intersection(final_c.nodes) != set():
                final_c.merge_with_cluster(_cluster)
                added_to_final = True
        if added_to_final == False:
            final_clusters.append(_cluster)
    for i, _cluster in enumerate(final_clusters):
        _cluster.name = f'cluster_{i}'
    return final_clusters



def _pca(X_data,n=1):
#    pc_explained_variance = []
#    pc_components = []
    pca = PCA(n_components = n, svd_solver = 'auto')
    pca.fit(X_data)
    X_pca = pca.transform(X_data)
#    print(pca.explained_variance_ratio_ , sum(pca.explained_variance_ratio_ ))
#    for i in range (n):
#        pc_explained_variance.append(pca.explained_variance_)
#        pc_components.append(pca.components_)        
    return (X_pca ,pca.explained_variance_ratio_, pca.components_, pca)               


class collinear_data():
    def __init__ (self, collinear_df):
        self.collinear_df = collinear_df
#        self._clusters = self.clusters(threshold = 0.7)
#        self.cluster_variables = {cl.name:cl.nodes for cl in self._clusters}
        self.pca_obj_dict = None
        
    def clusters(self, threshold = 0.4):
        self._clusters = identify_cluster(self.collinear_df , threshold = threshold)
        self.cluster_variables = {cl.name:cl.nodes for cl in self._clusters}
        for _cl in self._clusters:
            setattr(self,_cl.name, _cl)

    
    def _add_pc_to_collin_df(self, raw_data_df, pc_data, cluster_name, column_to_drop):
        raw_data = raw_data_df.copy()
        for i in range(pc_data.shape[1]):
            raw_data[f'{cluster_name}_pc{i}'] = pc_data[:,i]
        raw_data = raw_data.drop(column_to_drop, axis = 1)
        return raw_data
        
    def non_collinear_df(self, df,threshold = 0.4, min_total_variance_ratio_explained = 0.9, verbose = True):
        self.clusters(threshold=threshold)

        self.collinear_df=df.copy()
        final_df = self.collinear_df.copy()#self.collinear_df.copy()
#        conversion_dict={}
        pca_obj_dict = {}
#        cluster_linearity_index_dict={}
        for cluster_ in self._clusters:
#            print ('**',len(cluster_.nodes))
            for num_component in range(1,len(cluster_.nodes)): 
                pc_data, expl_variance, component, pca_obj = _pca(self.collinear_df[cluster_.nodes], n=num_component)
                dic1=list({str(cluster_.nodes)[1:-1]})
                print(dic1)

                if sum(expl_variance) > min_total_variance_ratio_explained:
                    break
            if verbose:
                print ('*'*10)
                print (cluster_.name)
                print (f'feature name = {str(cluster_.nodes)[1:-1]}')
                print (f'number of PC needed = {len(expl_variance)}')
                for i , variance in enumerate(expl_variance):
                    print (f'explained variance by PC_{i} = {variance}')
#            print ('Super_param shape is: ', pc_data.shape)
#            cluster_linearity_index_dict[cluster_.name] = expl_variance[0]/expl_variance[1]
#            final_df = final_df.drop(cluster_.nodes, axis = 1)
#            for i in range(pc_data.shape[1]):
#                final_df[f'{cluster_.name}_pc{i}'] = pc_data[:,i]
            
            final_df = self._add_pc_to_collin_df(final_df, pc_data, cluster_.name,cluster_.nodes)
#            conversion_dict[cluster_.name] = pd.Series(component[0][0], index = cluster_.nodes)
            pca_obj_dict[cluster_.name]  = pca_obj
#        self.conversion_dict = conversion_dict
        self.pca_obj_dict = pca_obj_dict
#        self.cluster_collinearity_index= cluster_linearity_index_dict

        return final_df
    
    def convert_new_collin_data(self, sample_collin_df):
        '''
        Converts a dataframe containing collinear variables to the \
        non_collinear version that can be used with the non_collinear \
        training set. This function is meant to be used after the clusters \
        are identified so first run non_collinear_df method to identify 
        clusters and create conversion_dict.
        '''
        final_result = sample_collin_df.copy()
#        if self.conversion_dict == None:
#            raise Exception ("'conversion_dict' missing. Please run 'non_collinear_df' method first.")
#        for cl in  self._clusters:
#            final_result[cl.name] = (point_ds[self.conversion_dict[cl.name].index]*self.conversion_dict[cl.name]).sum()
#            final_result = final_result.drop(list(self.conversion_dict[cl.name].index)) #drop does not work
#        return final_result
        if self.pca_obj_dict == None:
            raise Exception ("'conversion_dict' missing. Please run 'non_collinear_df' method first.")
        for cl in  self._clusters:
            collin_data = final_result[cl.nodes]
            pc_data = self.pca_obj_dict[cl.name].transform(collin_data)
            final_result = self._add_pc_to_collin_df(final_result, pc_data, cl.name, cl.nodes)
#            
#            
#            final_result[cl.name] = (point_ds[self.conversion_dict[cl.name].index]*self.conversion_dict[cl.name]).sum()
#            final_result = final_result.drop(list(self.conversion_dict[cl.name].index)) #drop does not work
        return final_result
def sample_data(file):
    return pd.read_csv(file, index_col = 'Time')




if __name__ == '__main__':
    raw_data = sample_data('../input/sample-multicollinear-data/sample_X_data.csv')
# =============================================================================
# We should first define a threshold for identifying collinear pairs. Two 
# variables are collinear if their abs(Pearson correlation parameter) > threshold.    
# =============================================================================
  
    thresh = 0.7

# =============================================================================
# You can identify the clusters and  visualize them with graphs without 
# doing any processing. Uncomment he next three lines if you want to do so.    
# =============================================================================
#    clusters = identify_cluster(raw_data, threshold = thresh)
#    for cl in clusters:
#        cl.plot()

# =============================================================================
# Now the normal way of using this library. First create a collinear_data 
# object by providing the raw data which is a Pandas dataframe.
# =============================================================================

    collin_data = collinear_data(raw_data)

# =============================================================================
# Let's create a non-collinear version of the data set. Under the hood,this is
# done by first identifying collinear pairs of variables, then clustering the
# the collinear pairs that share a varaible. Finally, principal component (PC)  
# of the cluster will be calculated and replace the cluster variables in the 
# original data set. The number of PC used to represent the cluster is 
# determined by the minimum amount variance ration needed to be explained by
# by the PCs combined. By default, enough PCs will be added to explain 90% of
# variance. Minimum explained variance ratio can be changed if needed.
# =============================================================================
    
    non_collin_data = collin_data.non_collinear_df(threshold = thresh, \
                                                   min_total_variance_ratio_explained = 0.9)

# =============================================================================