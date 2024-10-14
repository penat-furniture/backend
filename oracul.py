import numpy as np
import pandas as pd


class PenatOracul():
    def __init__(self, collection, price_csv_path, not_found_final_res, main_type='chair', panel_size=16):
        self.collection = collection
        self.prices = pd.read_csv(price_csv_path)
        self.main_type = main_type
        self.cached_starting_obj_pool = None
        self.cached_starting_obj_weights = None
        self.panel_size = panel_size
        self.not_found_final_res = not_found_final_res
    
    def get_starting_objects(self, ):
        if self.cached_starting_obj_pool is not None:
            res = np.random.choice(self.cached_starting_obj_pool, self.panel_size, replace=False, p=self.cached_starting_obj_weights)
            return list(res)
        all_ids = self.collection.get(where={'type': self.main_type})['ids']
        np.random.shuffle(all_ids)
        all_ids = all_ids[:1000] # pick random 1000 obj
        
        search_embs = self.collection.get(ids=all_ids, include=['embeddings'], where={'type': self.main_type})['embeddings']
        query_res = self.collection.query(query_embeddings=search_embs, where={'type': self.main_type}, n_results=10)
        ret_ids = query_res['ids'] # get for each of sampled, get 10 closest
        ret_ids_flat = [x for xs in ret_ids for x in xs]
        
        df = pd.DataFrame({'name': ret_ids_flat, 't': [1 for _ in ret_ids_flat]})
        res_df = df.groupby(by=['name']).count().sort_values('t', ascending=False)[:5 * self.panel_size]
        proposed_elems = res_df.index.tolist() # get 5 * panel_size most popular objects
        
        proposed_embs = self.collection.get(ids=proposed_elems, include=['embeddings'], where={'type': self.main_type})['embeddings']
        query_res = self.collection.query(query_embeddings=proposed_embs, where={'type': self.main_type}, n_results=10)
        
        cooc = dict([(x, 0) for x in proposed_elems])
        for close_ones in query_res['ids']:
            for el in close_ones:
                if el in cooc:
                    cooc[el] += 1
        self.cached_starting_obj_pool = list(cooc.keys())
        self.cached_starting_obj_weights = 1 / np.array(list(cooc.values()))
        self.cached_starting_obj_weights = self.cached_starting_obj_weights / self.cached_starting_obj_weights.sum()
        return self.get_starting_objects() 

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
    
    def select_with_price_range(self, ids, min_price, max_price, choose_top_n):
        candidates = self.prices[ # don't sort by object_type, because category is sometimes empty in links.csv
            (self.prices['price'] >= min_price) & 
            (self.prices['price'] <= max_price)
        ]
        candidate_ids = set(candidates['name'].to_list())
        print(len(ids), len(candidate_ids))
        good_ids = [idx for idx in ids if idx in candidate_ids]
        
        if len(good_ids) > 0:
            chosen_id = np.random.choice(good_ids[:choose_top_n], 1)[0]
            res = candidates[candidates['name'] == chosen_id].iloc[0][['name', 'price', 'link']].to_dict()
            res['price'] = str(res['price'])
            print("OK")
            return res
        else:
            print("NF")
            return self.not_found_final_res

    def run_final_step(self, image, limits):
        object_emb = self.collection.get(ids=[image], where={'type': self.main_type}, include=['embeddings'])['embeddings'][0]
        outp = {}
        for element_type, lims in limits.items():
            print(element_type)
            res = self.collection.query(query_embeddings=object_emb, where={'type': element_type}, n_results=3000, include=[])
            choose_top_n = 1 if element_type == self.main_type else 10
            res_dict = self.select_with_price_range(res['ids'][0], lims['min'], lims['max'], choose_top_n)
            outp[element_type] = res_dict
        return outp
            
    
    def run_general_step(self, step_n, image=None):
        assert step_n == 1 or image is not None, "In steps after 1, mast provide image"
        if step_n == 1:
            return self.get_starting_objects()
        else:
            dist = self.get_distances(image)
            return self.sample_from_distances(dist, step=step_n)