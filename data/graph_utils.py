import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # remove info messages of stellargraph
import pickle
import torch
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from typing import Union
from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
from gensim.models import Word2Vec
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.convert import from_networkx as torch_from_networkx
from data.data_utils import load_features_and_labels


# Input file paths
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
PATH_CHART_EVENTS = 'data/physionet.org/files/mimiciii/1.4/CHARTEVENTS.csv.gz'
PATH_PATIENT_WARDS = os.path.join(PROCESSED_DATA_DIR, 'patient-wards.csv')
PATH_COLUMNS_AND_LABELS = os.path.join(PROCESSED_DATA_DIR, 'patient-ward_columns_and_labels.csv')

# Ouput file paths
PATH_PATIENT_WARD_CAREGIVER_MAPPING = os.path.join(PROCESSED_DATA_DIR, 'patient-ward_caregiver_mapping.csv')
PATH_LINK_DICT = {
    'wards': os.path.join(PROCESSED_DATA_DIR, 'graph_links_wards.csv'),
    'caregivers': os.path.join(PROCESSED_DATA_DIR, 'graph_links_caregivers.csv'),
    'all': os.path.join(PROCESSED_DATA_DIR, 'graph_links_all.csv'),
}

# Data features to be used to create links between patient-wards
ADMISSION_COLUMNS = ['SUBJECT_ID', 'HADM_ID']
CHARTEVENT_COLUMNS = ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'CGID']
CAREGIVER_COLUMNS = ['SUBJECT_ID', 'HADM_ID', 'INTIME', 'OUTTIME', 'CURR_WARDID', 'CHARTTIME', 'CGID']
WARD_COLUMNS = ['SUBJECT_ID', 'HADM_ID', 'INTIME', 'OUTTIME', 'CURR_WARDID']
REGENERATE_LINKS_FROM_SCRATCH = False
REGENERATE_STELLAR_DATA_FROM_SCRATCH = False


def main():
    """ Build all datasets and link sets needed for GNN and control experiments
    """
    if REGENERATE_LINKS_FROM_SCRATCH:
        # Generate graph links for graph-based algorithms
        link_patient_wards_to_caregivers()  # (14_618_862 x 10); approx 4 hours
        generate_caregiver_links()  # (588_390 x 2); approx. 2 minutes
        generate_ward_links()  # (184_272 x 2); approx. 3 minutes
        merge_ward_and_caregiver_links()  # (722_996 x 2), i.e., [src, dest]

    # Try to load edges to check everything went goods
    for link_cond in PATH_LINK_DICT.keys():
        load_edges(link_cond)


def link_patient_wards_to_caregivers():
    """ Add links between patients that were visited by the same caregiver at the same time
    """
    print('Linking caregivers to patient-wards')
    # Load caregiver data
    df_caregivers = pd.read_csv(PATH_CHART_EVENTS, usecols=CHARTEVENT_COLUMNS) # takes approx. 8 minutes (330_712_483 x 7)
    df_caregivers['CHARTTIME'] = pd.to_datetime(df_caregivers['CHARTTIME'])
    df_caregivers['day'] = df_caregivers['CHARTTIME'].dt.day
    df_caregivers['month'] = df_caregivers['CHARTTIME'].dt.month
    df_caregivers['year'] = df_caregivers['CHARTTIME'].dt.year

    # Load patient-ward location data
    df_wards = pd.read_csv(PATH_PATIENT_WARDS, usecols=WARD_COLUMNS)  # takes a few seconds (261_857 x 5)
    df_wards['INTIME'] = pd.to_datetime(df_wards['INTIME'])
    df_wards['OUTTIME'] = pd.to_datetime(df_wards['OUTTIME'])

    # Link caregivers to patient-wards
    df_final = pd.merge(df_caregivers, df_wards, how='left',
                        left_on=ADMISSION_COLUMNS,  # not sure about how all these merges are done
                        right_on=ADMISSION_COLUMNS)  # takes approx 1.8 hours (1_664_662_896 x 10)
    df_final = df_final[(df_final['CHARTTIME'] >= df_final['INTIME']) &
                        (df_final['CHARTTIME'] <= df_final['OUTTIME'])]  # takes approx. 2 hours (329_366_946 x 10)
    df_final = df_final.drop_duplicates()  # takes approx. 30 minutes (14_618_862 x 10)
    df_final.to_csv(PATH_PATIENT_WARD_CAREGIVER_MAPPING, encoding='utf-8')
    

def generate_caregiver_links():
    """ Generate links between patients having the same caregiver simultaneously
    """
    print('Generating graph links using patient-caregiver data')
    # Load patient-ward and caregiver data and merge them in a single dataframe
    df_data = pd.read_csv(PATH_COLUMNS_AND_LABELS, usecols=WARD_COLUMNS)
    df_data['PWARD_ID'] = df_data.index  # to keep track of row ids
    df_ward_caregiver = pd.read_csv(PATH_PATIENT_WARD_CAREGIVER_MAPPING,
                                    usecols=CAREGIVER_COLUMNS)
    df_patients = pd.merge(df_ward_caregiver, df_data, how='left',  # 18M samples
                           left_on=WARD_COLUMNS, right_on=WARD_COLUMNS)
    
    # Group patients by caregiver ids, only if visited on the same day
    df_patients['CHARTTIME'] = pd.to_datetime(df_patients['CHARTTIME'])
    df_patients['CHARTDATE'] = df_patients['CHARTTIME'].dt.date  # only day
    used_columns = ['SUBJECT_ID', 'HADM_ID', 'PWARD_ID', 'CHARTDATE', 'CGID']
    df_patients = df_patients.filter(items=used_columns)
    df_patients = df_patients.drop_duplicates()  # 250k samples -> faster!
    df_grouped = df_patients.groupby(['CHARTDATE', 'CGID'])  # 1 minute
    
    # Add links between patients visited by the same caregiver on the same day
    links = set()
    patient_dict = dict(zip(df_data['PWARD_ID'], df_data['HADM_ID']))  # useful?
    for _, group in tqdm(df_grouped, desc='Computing links'):
        src_patient_wards = group['PWARD_ID'].unique().tolist()
        for src_id in src_patient_wards:
            tgt_patient_wards = [i for i in src_patient_wards if i != src_id]
            for tgt_id in tgt_patient_wards:
                if patient_dict[src_id] != patient_dict[tgt_id]:  # why this????
                    links.add(frozenset([src_id, tgt_id]))  # unordered

    # Generate data file containing caregiver links
    link_dict = {'src': [list(link)[0] for link in links],
                 'dst': [list(link)[1] for link in links]}
    df_caregiver_links = pd.DataFrame.from_dict(link_dict)
    df_caregiver_links.to_csv(PATH_LINK_DICT['caregivers'])


def generate_ward_links():
    """ Generate links between patients that visited the same ward at the same time
    """
    print('Generating graph links using in-ward patient data')
    # Load patient-ward data
    df_data = pd.read_csv(PATH_COLUMNS_AND_LABELS, usecols=WARD_COLUMNS)
    df_grouped = df_data.groupby('CURR_WARDID')

    # Add connections between parients that visited the same room at the same time
    links = set()
    for group_name, group in tqdm(df_grouped, desc='Computing links'):
        for row in tqdm(group.itertuples(), desc=' - Group %s' % group_name,
                        leave=False, total=len(group)):
            contact_patients = group[(group['INTIME'] < row.OUTTIME) &
                                     (group['OUTTIME'] > row.INTIME) &
                                     (group.index != row.Index)]
            for contact_patient in contact_patients.index:
                links.add(frozenset([row.Index, contact_patient]))  # unordered

    # Generate data file containing ward links
    link_dict = {'src': [list(link)[0] for link in links],
                 'dst': [list(link)[1] for link in links]}
    df_ward_links = pd.DataFrame.from_dict(link_dict)
    df_ward_links.to_csv(PATH_LINK_DICT['wards'])
    

def merge_ward_and_caregiver_links():
    """ Merge links coming from having a common ward, and having common caregivers
    """
    print('Generating a new set of links combining in-ward and caregiver links')
    df_ward_links = pd.read_csv(PATH_LINK_DICT['wards'], index_col=[0])
    df_cg_links = pd.read_csv(PATH_LINK_DICT['caregivers'], index_col=[0])
    df_all = pd.concat([df_ward_links, df_cg_links], ignore_index=True)
    df_all = df_all.drop_duplicates()
    df_all.to_csv(PATH_LINK_DICT['all'])
    

def load_edges(link_cond: str,
               node_ids: pd.Index=None
               ) -> pd.DataFrame:
    """ Load edges between patient-wards, given {'wards', 'caregivers', 'all'}
        condition
    """
    # Load edges and remove edges of absent nodes (e.g., for under-sampling)
    if link_cond == 'no':
        return pd.DataFrame.from_dict({'src': [], 'dst': []})
    edges = pd.read_csv(PATH_LINK_DICT[link_cond], index_col=[0])
    if node_ids is not None:  # remove edges that are not in node_ids
        edges = edges[edges['src'].isin(node_ids) & edges['dst'].isin(node_ids)]
    if __name__ == '__main__':
        print('Loaded graph edges successfully: %s' % (edges.shape,))
    return edges


def account_for_duplicate_nodes(node_ids: pd.Index,
                                edges: pd.DataFrame
                                ) -> tuple[pd.Index, pd.DataFrame]:
    """ Update duplicate nodes with unique ids and copy edges from the original,
        in case of over-sampling (i.e., some nodes are duplicated)
    """
    # Identify nodes that are not unique
    plural_node_ids = node_ids.value_counts()
    plural_node_ids = plural_node_ids[plural_node_ids > 1]
    max_node_id = max(node_ids)
        
    # Go through all duplicate nodes
    node_ids_to_add, edges_to_add = [], []
    for node_id, count in tqdm(
            plural_node_ids.items(), leave=False, total=len(plural_node_ids),
            desc=' - Nodes and edges updated for over-sampling'):
        # Identify new unique ids for duplicate nodes (but keep one original)
        new_node_ids = list(range(max_node_id + 1, max_node_id + count))
        node_ids_to_add.extend(new_node_ids)
        max_node_id += count - 1  # count - 1 = len(new_node_ids)
        
        # Add new edges for updated node ids, copying original node edges
        for new_node_id in new_node_ids:
            new_edges = edges[(edges['src'] == node_id) |
                              (edges['dst'] == node_id)]\
                             .replace(node_id, new_node_id)
            edges_to_add.append(new_edges)
    
    # Update and return new node_ids and edges
    node_ids = node_ids.unique().append(pd.Index(node_ids_to_add))
    edges = pd.concat([edges] + edges_to_add, ignore_index=True)
    return node_ids, edges


def load_graph_features_and_labels(setting_cond, balanced_cond, link_cond):
    """ Load features from embeddings built with node2vec (word2vec trained on
        node sequences using a random edge-walker)
    """
    dataset = IPCDataset(setting_cond, balanced_cond, link_cond, 'stellar')
    X, y = {}, {}
    if setting_cond == 'inductive':
        for split in ['train', 'dev', 'test']:
            data = dataset.get_split(split)
            X[split] = data.x.numpy()
            y[split] = data.y.numpy()
    elif setting_cond == 'transductive':
        data = dataset.get_split('whole')
        for split in ['train', 'dev', 'test']:
            X[split] = data.x[data.masks[split]].numpy()
            y[split] = data.y[data.masks[split]].numpy()
    return X, y
        

class IPCDataset(InMemoryDataset):
    """ Dataset containing graph data, and node indices for train, dev and test
    """
    def __init__(self, setting_cond, balanced_cond, link_cond, format='torch'):
        super(IPCDataset, self).__init__()
        print(' - Creating data graph')
        # Load features, labels, node ids, and initialize data
        self.setting_cond = setting_cond
        self.balanced_cond = balanced_cond
        self.link_cond = link_cond
        self.format = format
        X, y, ids = load_features_and_labels(balanced_cond)
        split_list = []
        
        # Create graph for a transductive setting
        if setting_cond == 'transductive':
            graph = self.create_transductive_graph(X, y, ids)
            split_list.append(graph)
        
        # Create graphs for an inductive setting
        elif setting_cond == 'inductive':
            for split in ['train', 'dev', 'test']:
                X_, y_, ids_ = X[split], y[split], ids[split]
                graph = self.create_graph(X_, y_, ids_, split=split)
                split_list.append(graph)
        
        # Define splits and collate data into a nice dataset
        self.split_indices = {'whole': 0, 'train': 0, 'dev': 1, 'test': 2}
        self.data, self.slices = self.collate(split_list)
    
    def get_split(self, name):
        """ Workaround to get dataset splits by split name instead of indices
        """
        return self[self.split_indices[name]]
    
    def create_transductive_graph(self,
                                  X: dict[np.ndarray],
                                  y: dict[pd.DataFrame],
                                  node_ids: dict[pd.Index],
                                  ) -> Union[StellarGraph, Data]:
        """ Create transductive graph using nodes, node labels, node ids, and
            appends train, dev, and test masks to the graph
        """
        # Create graph using the totality of nodes, node labels, and node ids
        X_ = np.concatenate((X['train'], X['dev'], X['test']))  # np.ndarray
        y_ = pd.concat((y['train'], y['dev'], y['test']))  # pd.DataFrame
        ids_ = node_ids['train'].append(node_ids['dev'])\
                                .append(node_ids['test'])  # pd.Index
        graph = self.create_graph(X_, y_, ids_)
        
        # Create masks to retrieve train, dev, and test predictions
        masks = {k: torch.zeros(X_.shape[0], dtype=torch.bool)
                 for k in ('train', 'dev', 'test')}
        masks['train'][:len(X['train'])] = True
        masks['dev'][len(X['train']):len(X['train']) + len(X['dev'])] = True
        masks['test'][-len(X['test']):] = True
        
        # Return final graph, after adding transductive masks
        graph.masks = masks
        return graph
    
    def create_graph(self,
                     X: np.ndarray,
                     y: pd.Series,
                     node_ids: pd.Index,
                     split: int='',
                     ) -> Data:
        """ Create graph using nodes, node labels, and node ids
        """
        # Load edges, and update node ids and edges in case of over-sampling
        edges = load_edges(self.link_cond, node_ids)
        node_ids, edges = account_for_duplicate_nodes(node_ids, edges)
        
        # Initialize graph and add nodes features and labels
        nx_graph = nx.Graph()
        for node_id, feat, lbl in zip(node_ids, X, y):
            node_info = {'x': feat.tolist(), 'y': float(lbl)}
            nx_graph.add_node(node_id, **node_info)
            
        # Add edges and return pytorch-geometric or stellar-graph object
        nx_graph.add_edges_from(edges.values)
        if self.format == 'torch':
            return torch_from_networkx(nx_graph)
        elif self.format == 'stellar':
            return self.create_embedded_graph(nx_graph, split)
    
    def create_random_walks(self,
                            graph: StellarGraph,
                            max_walk_length: int=32,
                            n_random_walks_per_root_node: int=10,
                            p_param: float=0.5,
                            q_param: float=2.0,
                            ) -> list:
        """ Create node sentences by walking randomly along graph edges
        """
        random_walker = BiasedRandomWalk(graph)
        random_walks = random_walker.run(
            nodes=list(graph.nodes()),
            length=max_walk_length,
            n=n_random_walks_per_root_node,
            p=p_param,
            q=q_param,
        )
        return random_walks
    
    def create_embedded_graph(self,
                              nx_graph: nx.Graph,
                              split: str='',  # for checkpoint retrieval
                              dim: int=128,
                              window: int=5,
                              ) -> list[np.ndarray]:
        # Initialize graph and data checkpoint path
        stellar_graph = StellarGraph.from_networkx(nx_graph)
        pickle_path = os.path.join(PROCESSED_DATA_DIR,
                                   '%s_balanced' % self.balanced_cond,
                                   'stellar',
                                   '%s_setting' % self.setting_cond,
                                   '%s_links' % self.link_cond,
                                   'node_embeddings_%s' % split)
        
        # Load checkpoint if data is already generated
        loading_failed = False
        if not REGENERATE_STELLAR_DATA_FROM_SCRATCH:
            try:
                with open(pickle_path, 'rb') as f:
                    node_embeddings = pickle.load(f)
            except:
                loading_failed = True
                
        # Else, regenerate node embeddings
        if REGENERATE_LINKS_FROM_SCRATCH or loading_failed:
            os.makedirs(os.path.split(pickle_path)[0], exist_ok=True)
            walks = self.create_random_walks(stellar_graph)
            str_walks = [[str(n) for n in walk] for walk in walks]
            model = Word2Vec(str_walks, vector_size=dim, window=window, sg=1)
            nodes = list(stellar_graph.nodes())
            node_embeddings = [model.wv[str(node)] for node in nodes]
            with open(pickle_path, 'wb') as f:
                pickle.dump(node_embeddings, f)
        
        # Return pyg-graph after adding node embeddings
        pyg_graph = torch_from_networkx(nx_graph)
        pyg_graph.x = torch.from_numpy(np.stack(node_embeddings))
        # updated_node_features = [pyg_graph.x, torch.from_numpy(node_embeddings)]
        # pyg_graph.x = torch.cat(updated_node_features, dim=-1)
        return pyg_graph
    