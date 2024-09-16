import warnings
warnings.filterwarnings("ignore", message=".*You are using `torch.load` with.*")
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

import os
from tqdm import tqdm
from PIL import Image
import chromadb
import open_clip
import torch


# Quick license check and acknowledgements
# open_clip has open license: https://github.com/mlfoundations/open_clip?tab=License-1-ov-file
# chromadb has apache2.0 licence: ok for commertial https://github.com/chroma-core/chroma?tab=Apache-2.0-1-ov-file#readme
# pretrained models have their own licences, they are listed here: https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md
# We chose CLIP-ViT-L-14-DataComp with MIT license: https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K#citation

def extract_embeddings_and_save(dir_path):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')
    model.eval() 

    embs = {}
    files = os.listdir(dir_path)
    files = [f for f in files if f[-5:] == '.jpeg'] # BE CAREFUL
    print(f"About to process: {len(files)} files")
    
    with torch.no_grad():
        for f in tqdm(files):
            file_path = os.path.join(dir_path, f)
            image = preprocess(Image.open(file_path)).unsqueeze(0)
            image_features = model.encode_image(image)[0]
            image_features = image_features / (image_features ** 2).sum() ** 0.5 # make mse == cossim
            file_code = f.split('.')[0]  # assume f is 'file_code.smth'
            embs[file_code] = image_features
    return embs


def add_embeddings_to_chroma(embs, chroma_client, collection_name, metadata_val):
    collection = chroma_client.get_or_create_collection(name=collection_name)
    all_embs = [x.tolist() for x in embs.values()]
    all_ids = list(embs.keys())
    batch_upload_size = 500
    for start in range(0, len(all_ids), batch_upload_size):
        collection.add(
            embeddings=all_embs[start:start + batch_upload_size],
            ids=all_ids[start:start + batch_upload_size],
            metadatas=[{'type': metadata_val} for i in range(len(all_ids[start:start + batch_upload_size]))]
        )
    print('ok')

if __name__ == "__main__":
    # add many folders one after another
    chroma_client = chromadb.PersistentClient(path="/Users/ksc/penat/chrom_chache_export")
    for category in ['bed', 'dining_table', 'potted_plant', 'chair', 'couch']:
        embs = extract_embeddings_and_save(f'data/{category}')
        add_embeddings_to_chroma(embs, chroma_client, collection_name='common', metadata_val=category)
    
    # you can also comment the previous part and just run this for new categories
    chroma_client = chromadb.PersistentClient(path="/Users/ksc/penat/chrom_chache_export")
    embs = extract_embeddings_and_save(f'data/closet')
    add_embeddings_to_chroma(embs, chroma_client, collection_name='common', metadata_val='closet')
    