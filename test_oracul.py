import pandas
import numpy as np
import pandas as pd
import chromadb
from oracul import PenatOracul


if __name__ == "__main__":
    chroma_client = chromadb.PersistentClient(path="/Users/ksc/penat/chroma_cache")
    oracul = PenatOracul(
        collection=chroma_client.get_or_create_collection("common"),
        price_csv_path='/Users/ksc/penat/Penat for Peter/data/links.csv',
        not_found_final_res={
            "name": "Не найдено",
            "image": 'https://static01.nyt.com/images/2016/09/28/us/17xp-pepethefrog_web1/28xp-pepefrog-superJumbo.jpg',
            "link": None,
            "price": None,
            "limits": {"min": 0, "max": 1e6, "loaded": True}
        },
        main_type='chair',
        panel_size=16
    )
    
    full_lims = {
        "bed": {
            "min": 0,
            "max": 91990
        },
        "plant": {
            "min": 0,
            "max": 40000
        },
        "couch": {
            "min": 0,
            "max": 2088000
        },
        "table": {
            "min": 0,
            "max": 2800000
        },
        "chair": {
            "min": 0,
            "max": 20000
        },
        "closet": {
            "min": 0,
            "max": 1e6
        }
    }
    
    print(oracul.run_general_step(1))
    print(oracul.run_general_step(2, 'AkVFEYusMuXdl1Z'))
    print(oracul.run_final_step('AkVFEYusMuXdl1Z', limits=full_lims))