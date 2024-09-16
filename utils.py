import math
import os
import random
import string
from typing import Optional, Union
import pandas as pd
import numpy as np

SERVER_URL = os.getenv("SERVER_URL", "https://cdn.penat.su")

chairs_set = pd.read_csv('data/set_of_chair.csv')
links_set = pd.read_csv('data/links.csv')
links_set['names_jpeg'] = links_set['name'] + ".jpeg"
set_of_chair = pd.read_csv('data/set_of_chair.csv')

names = ['bed', 'plant', 'couch', 'table', 'chair']
object_names = ["beds", "plants", "couches", "tables", "chairs"]
all_objects_to_sell_num = []
all_objects_to_sell_names = []
set_ = pd.read_csv('data/XY_database.csv')
df = pd.read_csv('data/set_of_chair.csv')
limits = {}
for i in range(5):
    csv_file_path = f"objects/{names[i]}_nums.csv"
    all_objects_to_sell_num.append(pd.read_csv(csv_file_path))
    csv_file_path = f"objects/{names[i]}_names.csv"
    all_objects_to_sell_names.append(pd.read_csv(csv_file_path))
    all_object_prices = links_set[links_set['names_jpeg'].isin(all_objects_to_sell_names[i]['name'])]['price']
    max_price = all_object_prices.max()
    limits[names[i]] = {
        "min": 0,
        "max": math.ceil(max_price) if not np.isnan(max_price) else 1e6,
    }


def resolve_path(name: str, image_type: str = "chairs") -> str:
    return f"{SERVER_URL}/images/{image_type}/{name}"

def get_first_chair_images():
    first_choice = pd.DataFrame(columns=set_of_chair.columns)
    k = 0
    for i in set_of_chair.columns:
        if i == "name":
            continue
        k += 1
        res = set_of_chair.nlargest(10, i).sample(1)
        first_choice = pd.concat([first_choice, res], axis=0, ignore_index=False)

    first_choice = [
        {
            "name": row["name"],
            "image": resolve_path(row["name"], "chairs"),
        }
        for _, row in first_choice.iterrows()
    ]
    return first_choice


def get_new_pictures(image_name):
    global df
    df_columns_except_name = df.columns.difference(['name'])
    given_vector = df[df['name'] == image_name][df_columns_except_name].values[0]
    df['distance'] = np.sqrt(((df[df_columns_except_name] - given_vector) ** 2).sum(axis=1))
    df = df.sort_values(by='distance')
    df = df.drop(columns='distance')
    k, scale = 3, 4
    num_samples = 16
    gamma_indices = []

    while len(gamma_indices) < num_samples:
        sample = np.random.gamma(shape=k, scale=scale, size=1).astype(int)[0]
        if sample not in gamma_indices and 0 <= sample < len(df):
            gamma_indices.append(sample)

    sampled_vectors = df.iloc[gamma_indices]
    return [
        {
            "name": row["name"],
            "image": resolve_path(row["name"], "chairs"),
        }
        for _, row in sampled_vectors.iterrows()
    ]


def closest_from_set(chosen_chair):
    global set_
    common_columns = set_.columns.intersection(chosen_chair.columns)
    distances = ((set_[common_columns] - chosen_chair[common_columns]) ** 2).mean(axis=1)
    closest_row_index = distances.idxmin()
    nearest_neighbor = set_.iloc[closest_row_index]
    return nearest_neighbor


# TODO: optimize


def final_res(picture_name, price_limits=None):
    global chairs_set, all_objects_to_sell_num, all_objects_to_sell_names, links_set, names, limits, object_names, SERVER_URL

    max_prices = {name: links_set[links_set['names_jpeg'].str.contains(name)]['price'].max() for name in names}

    if price_limits is None:
        price_limits = {name: {"min": 0, "max": max_prices[name] if pd.notna(max_prices[name]) else 1e6} for name in names}

    chosen_chair = chairs_set[chairs_set['name'] == picture_name].drop(columns='name')
    full_flat = closest_from_set(chosen_chair)

    res = {name: {} for name in names}
    for i, name in enumerate(names):
        print(name)
        only_obj = full_flat[full_flat.index.str.contains(name)]

        max_limit = limits[names[i]]['max']
        min_limit = limits[names[i]]['min']

        mse_list = np.mean((only_obj.values - all_objects_to_sell_num[i].values) ** 2, axis=1)
        mse_df = pd.DataFrame({'name': all_objects_to_sell_names[i]['name'], 'mse': mse_list})
        mse_df = mse_df.sort_values(by='mse')

        links_filtered = links_set[links_set['names_jpeg'].isin(mse_df['name'])].copy()
        links_filtered = links_filtered.dropna(subset=['price'])

        min_price, max_price = price_limits[name]['min'], price_limits[name]['max']
        valid_prices_mask = (links_filtered['price'] >= min_price) & (links_filtered['price'] <= max_price)

        combined_df = links_filtered.copy()
        print(len(combined_df))
        print(len(mse_df))
        print(combined_df.columns)
        print(mse_df.columns)
        # print(combined_df['names_jpeg'][0])
        # print(combined_df['name'][0])
        # print(mse_df['name'][0])
        mse_df = mse_df[mse_df['name'].isin(combined_df['names_jpeg'])]
        # return
        combined_df['mse'] = mse_df['mse'].values

        combined_df = combined_df[valid_prices_mask]
        combined_df = combined_df.sort_values('mse').head(5)

        if combined_df.empty:
            res[name] = {
                "name": "Не найдено",
                "image": resolve_path("not_found.png", "icons"),
                "link": None,
                "price": None,
                "limits": {"min": 0, "max": max_prices[name] if pd.notna(max_prices[name]) else 1e6, "loaded": True}
            }
        else:
            nearest_vector = combined_df.sample(n=1).iloc[0]

            if name == "chair" and min_price <= links_set[links_set['names_jpeg'] == picture_name]['price'].values[0] <= max_price:
                found_name = picture_name
            else:
                found_name = nearest_vector['name']

            found_price = nearest_vector['price']
            found_link = nearest_vector['link']

            image_name = found_name if found_name.endswith('.jpeg') else f"{found_name}.jpeg"
            image_path = resolve_path(image_name, object_names[i])

            res[name] = {
                "name": found_name,
                "image": image_path,
                "link": found_link,
                "price": str(found_price),
                "limits": {"min": min_limit, "max": max_limit}
            }
    return res
#
#
#
# def final_res(picture_name, price_limits=None):
#     global chairs_set, all_objects_to_sell_num, all_objects_to_sell_names, links_set, names
#     if price_limits is None:
#         price_limits = {}
#         for name in names:
#             price_limits[name] = {
#                 "min": 0,
#                 "max": 1e10,
#             }
#
#     chosen_chair = chairs_set[chairs_set['name'] == picture_name]
#     chosen_chair = chosen_chair.drop(columns='name')
#     full_flat = closest_from_set(chosen_chair)
#     res = {
#         "chair": {},
#         "bed": {},
#         "plant": {},
#         "couch": {},
#         "table": {},
#     }
#     for i in range(len(names)):
#         only_obj = full_flat[full_flat.index.str.contains(names[i])]
#         mse_list = [np.mean((abs(only_obj - vector)) ** 2) for vector in all_objects_to_sell_num[i].iloc]
#         min_limit = limits[names[i]]['min']
#         max_limit = limits[names[i]]['max']
#         price_limit = (price_limits[names[i]]['min'], price_limits[names[i]]['max'])
#         mse_list_filtered = []
#         for j in range(len(mse_list)):
#             current_price = \
#                 links_set[links_set['name'] + ".jpeg" == all_objects_to_sell_names[i].iloc[j]['name']]['price'].values[
#                     0]
#             if price_limit[0] <= current_price <= price_limit[1]:
#                 mse_list_filtered.append(mse_list[j])
#             else:
#                 mse_list_filtered.append(1e10)
#         lowest_indices = np.argsort(mse_list_filtered)[:5]
#         lowest_indices = [_ for _ in lowest_indices if mse_list_filtered[_] != 1e10]
#         if len(lowest_indices) == 0:
#             res[names[i]] = {
#                 "name": "Не найдено",
#                 "image": resolve_path("not_found.png", "icons"),
#                 "link": None,
#                 "price": None,
#                 "limits": {
#                     "min": min_limit,
#                     "max": max_limit,
#                     "loaded": True,
#                 }
#             }
#             continue
#         nearest_index = np.random.choice(lowest_indices)
#         nearest_vector = all_objects_to_sell_names[i].iloc[nearest_index]
#         if names[i] == "chair" and price_limits[names[i]]['min'] <= \
#                 links_set[links_set['name'] + ".jpeg" == picture_name]['price'].values[0] <= \
#                 price_limits[names[i]]['max']:
#             # print(f"chair: {nearest_vector['name']}")
#             # print(f'price: {links_set[links_set["name"] + ".jpeg" == nearest_vector["name"]]["price"].values[0]}')
#             # print(f'price_limits: {price_limits[names[i]]["min"]} {price_limits[names[i]]["max"]}')
#             found_name = picture_name
#         else:
#             found_name = nearest_vector['name']
#         found_link = links_set[links_set['name'] + ".jpeg" == found_name]['link'].values[0]
#         found_price = str(links_set[links_set['name'] + ".jpeg" == found_name]['price'].values[0])
#         res[names[i]] = {
#             "name": found_name,
#             "image": resolve_path(found_name, object_names[i]),
#             "link": found_link,
#             "price": found_price,
#             "limits": {
#                 "min": min_limit,
#                 "max": max_limit,
#             }
#         }
#     return res


def get_limits() -> dict:
    return limits



def get_images(step: str = "1", image: Optional[str] = None, limits: Optional[dict] = None,
               old: Optional[bool] = None) -> Union[list, dict]:
    if step == "1":
        return get_first_chair_images()
    elif step == "2":
        return get_new_pictures(image)
    elif step == "3":
        return get_new_pictures(image)
    elif step == "4":
        return final_res_old(image, limits) if old else final_res(image, limits)
    else:
        return []
