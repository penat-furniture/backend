import os
from typing import Optional, Union
import pandas as pd
import numpy as np
import chromadb
from oracul import PenatOracul

SERVER_URL = os.getenv("SERVER_URL", "https://cdn.penat.su")
PATH_TO_LINKS = './data/links.csv'
PATH_TO_CHROMA_CACHE = './data/chroma_cache'
AVAILABLE_CATEGORIES = ['bed', 'chair', 'couch', 'dining_table', 'potted_plant'] # chroma already has closet

def resolve_path(name: str, image_type: str = "chair") -> str:
    return f"{SERVER_URL}/images/{image_type}/{name}" # This needs CDN to have the correct names

chroma_client = chromadb.PersistentClient(path=PATH_TO_CHROMA_CACHE)
global_oracul = PenatOracul(
    collection=chroma_client.get_or_create_collection("common"),
    price_csv_path=PATH_TO_LINKS,
    not_found_final_res={
        "name": "Не найдено",
        "image": resolve_path("not_found.png", "icons"),
        "link": None,
        "price": None,
        "limits": {"min": 0, "max": 1e6, "loaded": True},
    },
    main_type='chair',
    panel_size=16
)

def get_limits() -> dict:
    res = {}
    for el in AVAILABLE_CATEGORIES:
        res[el] =  {
            "min": 0,
            "max": 1e10
        }
    return res

def get_images(step: str = "1", image: Optional[str] = None, limits: Optional[dict] = None,
               old: Optional[bool] = None) -> Union[list, dict]:
    if step == "1" or step == "2" or step == "3":
        oracul_outp = global_oracul.run_general_step(step_n=int(step), image=image)
        return [
            {
                "name": name,
                "image": resolve_path(name, "chair"),
            }
            for name in oracul_outp
        ]
    else:
        oracul_outp = global_oracul.run_final_step(image=image, limits=limits)
        res = {'images': {}}
        
        for category_name in oracul_outp: 
            res['images'][category_name] = {
                "name": oracul_outp[category_name]['name'],
                "image": resolve_path(oracul_outp[category_name]['name'], category_name), 
                "link": oracul_outp[category_name]['link'],
                "price": str(oracul_outp[category_name]['price']),
                "limits": limits
            }
        return res
