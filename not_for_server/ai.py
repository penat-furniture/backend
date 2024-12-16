import warnings
warnings.filterwarnings("ignore", message=".*You are using `torch.load` with.*")
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

import os
from tqdm import tqdm
from PIL import Image
import chromadb
import open_clip
import torch
import pandas as pd 


# Quick license check and acknowledgements
# open_clip has open license: https://github.com/mlfoundations/open_clip?tab=License-1-ov-file
# chromadb has apache2.0 licence: ok for commertial https://github.com/chroma-core/chroma?tab=Apache-2.0-1-ov-file#readme
# pretrained models have their own licences, they are listed here: https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md
# We chose CLIP-ViT-L-14-DataComp with MIT license: https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K#citation

def extract_embeddings_and_save(dir_path):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')
    model.eval() 
    mps_device = torch.device("mps")
    model = model.to(mps_device)

    embs = {}
    files = os.listdir(dir_path)
    files = [f for f in files if f[-5:] == '.jpeg'] # BE CAREFUL
    print(f"About to process: {len(files)} files")
    
    with torch.no_grad():
        for f in tqdm(files):
            file_path = os.path.join(dir_path, f)
            image = preprocess(Image.open(file_path)).unsqueeze(0).to(mps_device)
            image_features = model.encode_image(image)[0]
            image_features = image_features / (image_features ** 2).sum() ** 0.5 # make mse == cossim
            file_code = f.split('.')[0]  # assume f is 'file_code.smth'
            embs[file_code] = image_features.cpu()
    return embs


def add_embeddings_to_chroma(embs, chroma_client, collection_name, meta_mapping):
    collection = chroma_client.get_or_create_collection(name=collection_name)
    all_embs = [x.tolist() for x in embs.values()]
    all_ids = list(embs.keys())
    batch_upload_size = 500
    for start in range(0, len(all_ids), batch_upload_size):
        collection.add(
            embeddings=all_embs[start:start + batch_upload_size],
            ids=all_ids[start:start + batch_upload_size],
            metadatas=[meta_mapping[el] for el in all_ids[start:start + batch_upload_size]]
        )
    print('ok')
    

def make_room_metadata_mapping(links_path):
    df = pd.read_csv(links_path)
    print(df['room'].unique())
    mapping = {}
    for index, row in df.iterrows():
        mapping[row['name']] = {
            'type': row['type'], 
            'is_kitchen': 'kitchen' in row['room'],
            'is_bedroom': 'bedroom' in row['room'],
            'is_living_room': 'living_room' in row['room'],
        }
    return mapping


if __name__ == "__main__":
    # add many folders one after another
    chroma_client = chromadb.PersistentClient(path="/Users/ksc/penat/room_update_chroma_cache")
    meta_mapping = make_room_metadata_mapping("/Users/ksc/penat/new_data/links.csv")
    for fldr in ['c']: #, 'a', 'b', 'd', 'e', 'f']:
        embs = extract_embeddings_and_save(f'/Users/ksc/penat/new_data/{fldr}')
        add_embeddings_to_chroma(embs, chroma_client, collection_name='common', meta_mapping=meta_mapping)
    