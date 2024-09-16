import numpy as np
import pandas as pd


class PenatOracul():
    def __init__(self, collection, price_csv_path, not_found_final_res, main_type='chair', panel_size=16):
        self.collection = collection
        self.prices = pd.read_csv(price_csv_path)
        self.main_type = main_type
        self.cached_starting_obj = None
        self.panel_size = panel_size
        self.not_found_final_res = not_found_final_res
        self.final_category_mapping = { # i fucking hate the previous developer
            'table': 'dining_table',
            'plant': 'potted_plant'
        }
    
    def get_starting_objects(self, ):
        if self.cached_starting_obj is not None:
            return self.cached_starting_obj
        all_ids = self.collection.get(where={'type': self.main_type})['ids']
        np.random.shuffle(all_ids)
        all_ids = all_ids[:1000]
        
        search_embs = self.collection.get(ids=all_ids, include=['embeddings'], where={'type': self.main_type})['embeddings']
        query_res = self.collection.query(query_embeddings=search_embs, where={'type': self.main_type}, n_results=10)
        ret_ids = query_res['ids']
        ret_ids_flat = [x for xs in ret_ids for x in xs]
        
        df = pd.DataFrame({'name': ret_ids_flat, 't': [1 for _ in ret_ids_flat]})
        res_df = df.groupby(by=['name']).count().sort_values('t', ascending=False)[:3 * self.panel_size]
        proposed_elems = res_df.index.tolist()
        
        proposed_embs = self.collection.get(ids=proposed_elems, include=['embeddings'], where={'type': self.main_type})['embeddings']
        query_res = self.collection.query(query_embeddings=proposed_embs, where={'type': self.main_type}, n_results=10)
        
        cooc = dict([(x, 0) for x in proposed_elems])
        for close_ones in query_res['ids']:
            for el in close_ones:
                if el in cooc:
                    cooc[el] += 1
        sorted_cooc = sorted(cooc.items(), key=lambda x: x[1]) # ascending count, from least cooc to most
        self.cached_starting_obj = [sorted_cooc[i][0] for i in range(self.panel_size)]
        return self.cached_starting_obj 

    def get_distances(self, object):
        object_emb = self.collection.get(ids=[object], where={'type': self.main_type}, include=['embeddings'])['embeddings'][0]
        res = self.collection.query(query_embeddings=object_emb, where={'type': self.main_type}, n_results=101)
        df = pd.DataFrame({
            'name': res['ids'][0][1:],  # first element is itself with distance 0
            'dist': res['distances'][0][1:]
        })
        
        return df
    
    def sample_from_distances(self, distance_df, step):
        if step == 2:
            distance_df['dist'] = np.exp(-1 * ((distance_df['dist'] / 0.25) ** 2))
        elif step == 3:
            distance_df['dist'] = np.exp(-1 * ((distance_df['dist'] / 0.1) ** 2))
        else:
            raise Exception("Steps should be 2 or 3")
        distance_df['dist'] = distance_df['dist'] / distance_df['dist'].sum()
        
        inds = np.random.choice(len(distance_df), size=(self.panel_size, ), replace=False, p=distance_df['dist'])
        return distance_df['name'][inds].tolist()
    
    def select_with_price_range(self, ids, min_price, max_price, object_type):
        candidates = self.prices[ # don't sort by object_type, because category is sometimes empty in links.csv
            (self.prices['price'] >= min_price) & 
            (self.prices['price'] <= max_price)
        ]
        # print(object_type, min_price, max_price, "INTER:", len(candidates[candidates['name'].isin(ids)]))
        found_element = None
        candidate_ids = set(candidates['name'].to_list())
        for closest_id in ids:
            if closest_id in candidate_ids:
                found_element = closest_id
                break
        if found_element is not None:
            res = candidates[candidates['name'] == closest_id].iloc[0][['name', 'price', 'link']].to_dict()
            return res
        else:
            return self.not_found_final_res

    def run_final_step(self, image, limits):
        object_emb = self.collection.get(ids=[image], where={'type': self.main_type}, include=['embeddings'])['embeddings'][0]
        outp = {}
        for element_type, lims in limits.items():
            element_type = self.final_category_mapping.get(element_type, element_type) # change it, or keep the same
            res = self.collection.query(query_embeddings=object_emb, where={'type': element_type}, n_results=3000, include=[])
            res_dict = self.select_with_price_range(res['ids'][0], lims['min'], lims['max'], element_type)
            outp[element_type] = res_dict
        return outp
            
    
    def run_general_step(self, step_n, image=None):
        assert step_n == 1 or image is not None, "In steps after 1, mast provide image"
        if step_n == 1:
            return self.get_starting_objects()
        else:
            dist = self.get_distances(image)
            return self.sample_from_distances(dist, step=step_n)
