import numpy as np
import pandas as pd


class PenatOracul():
    def __init__(self, collection, price_csv_path, not_found_final_res, main_type='chair', panel_size=16):
        self.collection = collection
        self.prices = pd.read_csv(price_csv_path)
        self.main_type = main_type
        
        if main_type == 'all':
            self.where_filter = None 
        else:
            self.where_filter = {'type': self.main_type}
        
        self.cached_starting_obj_pool = None
        self.cached_starting_obj_weights = None
        self.panel_size = panel_size
        self.not_found_final_res = not_found_final_res
        self.distance_func = {
            2: self.get_kinda_normal_f(0.4),
            3: self.get_kinda_normal_f(0.3),
            4: self.get_kinda_normal_f(0.2),
            5: self.get_kinda_normal_f(0.1),
            6: self.get_kinda_normal_f(0.05),
            # 7: self.get_kinda_normal_f(0.5),
            # 8: self.get_kinda_normal_f(0.3),
            # 9: self.get_kinda_normal_f(0.2),
            # 10: self.get_kinda_normal_f(0.1),
        }
        
    def get_kinda_normal_f(self, k):
        f = lambda x: np.exp(-1 * ((x / k) ** 2))
        return f
    
    def upd_where_filter(self, room):
        print(room)
        orig_filter = self.where_filter
        if orig_filter is None:
            new_filter = {f'is_{room}': True}
            return new_filter
        new_filter = {
            '$and': [
                orig_filter,
                {f'is_{room}': True}
            ]
        }
        
        print(new_filter)
        return new_filter
    
    def get_starting_objects(self, room):
        where_filter = self.upd_where_filter(room)
        if self.cached_starting_obj_pool is not None:
            res = np.random.choice(self.cached_starting_obj_pool, self.panel_size, replace=False, p=self.cached_starting_obj_weights)
            return list(res)
        all_ids = self.collection.get(where=where_filter)['ids']
        np.random.shuffle(all_ids)
        all_ids = all_ids[:1000] # pick random 1000 obj
        
        search_embs = self.collection.get(ids=all_ids, include=['embeddings'], where=where_filter)['embeddings']
        query_res = self.collection.query(query_embeddings=search_embs, where=where_filter, n_results=10)
        ret_ids = query_res['ids'] # get for each of sampled, get 10 closest
        ret_ids_flat = [x for xs in ret_ids for x in xs]
        
        df = pd.DataFrame({'name': ret_ids_flat, 't': [1 for _ in ret_ids_flat]})
        res_df = df.groupby(by=['name']).count().sort_values('t', ascending=False)[:5 * self.panel_size]
        proposed_elems = res_df.index.tolist() # get 5 * panel_size most popular objects
        
        proposed_embs = self.collection.get(ids=proposed_elems, include=['embeddings'], where=where_filter)['embeddings']
        query_res = self.collection.query(query_embeddings=proposed_embs, where=where_filter, n_results=10)
        
        cooc = dict([(x, 0) for x in proposed_elems])
        for close_ones in query_res['ids']:
            for el in close_ones:
                if el in cooc:
                    cooc[el] += 1
        self.cached_starting_obj_pool = list(cooc.keys())
        self.cached_starting_obj_weights = 1 / np.array(list(cooc.values()))
        self.cached_starting_obj_weights = self.cached_starting_obj_weights / self.cached_starting_obj_weights.sum()
        return self.get_starting_objects() 

    def get_distances(self, object, room):
        print(object)
        where_filter = self.upd_where_filter(room)
        object_emb = self.collection.get(ids=[object], where=where_filter, include=['embeddings'])['embeddings'][0]
        res = self.collection.query(query_embeddings=object_emb, where=where_filter, n_results=500)
        df = pd.DataFrame({
            'name': res['ids'][0][1:],  # first element is itself with distance 0
            'dist': res['distances'][0][1:]
        })
        
        return df
    
    def get_starting_objects_random(self, room):
        where_filter = self.upd_where_filter(room)
        all_ids = self.collection.get(where=where_filter)['ids']
        np.random.shuffle(all_ids)
        return all_ids[:self.panel_size]
    
    def sample_from_distances(self, distance_df, step):
        if step not in self.distance_func:
            raise Exception(f"Bad step. Got {step}. Should be {list(self.distance_func.keys())}")
        print("BEFORE", distance_df['dist'].min(), distance_df['dist'].max())
        distance_df['dist'] = self.distance_func[step](distance_df['dist'])            
        distance_df['dist'] = distance_df['dist'] / distance_df['dist'].sum()
        print("AFTER", distance_df['dist'].min(), distance_df['dist'].max())
        
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

    def run_final_step(self, image, limits, room):
        where_filter = self.upd_where_filter(room)
        print(where_filter)
        object_emb = self.collection.get(ids=[image], where=where_filter, include=['embeddings'])['embeddings'][0]
        outp = {}
        for element_type, lims in limits.items():
            print(element_type)
            where_filter = {'$and': [
                {'type': element_type},
                {f'is_{room}': True}
            ]}
            res = self.collection.query(query_embeddings=object_emb, where=where_filter, n_results=3000, include=[])
            choose_top_n = 1 if element_type == self.main_type else 10
            res_dict = self.select_with_price_range(res['ids'][0], lims['min'], lims['max'], choose_top_n)
            outp[element_type] = res_dict
        return outp
            
    
    def run_general_step(self, step_n, image=None, room=None):
        assert step_n == 1 or image is not None, "In steps after 1, mast provide image"
        if step_n == 1:
            return self.get_starting_objects_random(room) #self.get_starting_objects()
        else:
            dist = self.get_distances(image, room)
            return self.sample_from_distances(dist, step=step_n)