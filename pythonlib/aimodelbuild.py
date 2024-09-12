import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from PIL import UnidentifiedImageError 
import os
import networkx as nx
import matplotlib.pyplot as plt

class aienginmodelbuild:
    def __init__(self, datafile_df,kgdatafilename):
        self.datafilename = datafile_df
        self.kgdatafilename =kgdatafilename

    def datapreparation(self):
        df1 = self.datafilename
        
        df = df1[df1['image_name'].notnull()& df1['ocr_det_arr'].notnull()]
        df.loc[:, 'Text'] = 'Component - ' + df['Text'].astype(str)

        #df.loc[:, 'cc_segment_image'] = df['cc_segment_image'].replace(['', None], pd.NA)
        #df.loc[:, 'cc_segment_image'] = df['cc_segment_image'].fillna(df['image_name'])

        header_cols = [col for col in df.columns if col.startswith('Header_')]
        # Sort 'Header_' columns numerically
        header_cols.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else float('inf'))

        # Reorder the columns with 'Header_' columns first, followed by 'Header_image', and then 'Text'
        new_cols_order = header_cols + ['Text'] + ['cc_segment_image']
        new_cols_order.remove('Header_style')
        
        filtered_df = df[new_cols_order]

        filtered_df = filtered_df.dropna(axis=1, how='all')

        
        self.knowledge_graph(filtered_df)
        return filtered_df
    
    def knowledge_graph(self,filtered_df):
        #knowledgedbgraph=[]
        if not os.path.exists(self.kgdatafilename):
            os.makedirs(self.kgdatafilename)
            print(f"Directory '{self.kgdatafilename}' created.")
        df_list = [d for _, d in filtered_df.groupby(['Header_image'])]
        for list_split in df_list:
            df_list_split = pd.DataFrame(list_split)
            print(df_list_split.shape)
    
            G = nx.DiGraph()

            # Iterate through DataFrame rows to add nodes and edges
            for _, row in df_list_split.iterrows():
                parent = row['Header_0']
                #parent_Header_image = row['Header_image']
                parent_Header_image = ''.join(e for e in row['Header_image'] if e.isalnum())+'.png'
                for child in row[1:]:
                    if pd.notnull(child) and parent != child:  # Check if the child is not null
                        G.add_edge(parent, child)
                        #print(parent,",", child)
                        parent = child

            # Manually modify node names to avoid ":" characters
            modified_node_names = {node: node.replace(':', '') for node in G.nodes()}
            #knowledgedbgraph.append(parent_Header_image)
            nx.relabel_nodes(G, modified_node_names, copy=False)

            # Convert networkx graph to pydot
            dot_graph = nx.drawing.nx_pydot.to_pydot(G)

            # Render the DOT representation
            dot_graph.set_rankdir('LR')  # Set direction left to right
            dot_graph.set_node_defaults(shape='box', style='filled', fillcolor='lightblue')
            dot_graph.write_png(os.path.join(self.kgdatafilename, parent_Header_image))

        #return knowledgedbgraph




    