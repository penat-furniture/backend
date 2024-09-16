# Penat Backend 

This is an attempt by Peter to modernize the Penat backend. 
The main things happen in main root folder in utils.py
Demo and code to process new image categoris are in `not_for_server` folder. This folder will be copied to the server, but it wont be used.


# How it used to work
- Label data by assigning one category to each item. Categories differ for chairs, beds, etc.
- Load a pretrained model, fine-tune it to predict these classes.
- Class probabilities are now embeddings for these categories. Different categories have different lengths.
- During chair steps, you get a list of all chairs ordered by MSE between their embeddings. Then you ignore distances, just take the order of elements, and sample indices with Gamma(3, 4) — the mean of indices in this sampling is 11. Sampling was the same for each step.
- For the final step:
    - We take images of pretty flats.
    - Run them through object detection, get 0 or 1 item of all of our categories, calculate embeddings for each found item.
    - We get a table of (number of pretty flat images, sum of all embedding lengths) with a lot of blanks, where there were no objects of that type in some flats. Then we fill in the blanks by taking the closest row from this table and filling the missing values with items from it.
    - In the final step, we get the chair embedding, find the closest row in this table, and retrieve the desired embeddings of other categories.
    - Then we somehow filter prices and return the closest ones to the selected embeddings from the table.

# How it works now
- Take a pretrained CLIP model and run it for all images. All images have embeddings of the same size. We can also query this database with text.
- Then we put these embeddings in a vector database with metadata of their type. This is the only way we access embeddings.
- In the chair steps, we get the 100 closest ones and sample from them. We calculate probability like this: `np.exp(-1 * ((distance / 0.25) ** 2))`. This is inspired by the normal distribution probability density, with the center at distance = 0. We decrease 0.25 to 0.1 in the third step. This can be thought of as lowering the temperature in sampling — less exploration, more exploitation.
- In the final step, since we didn't modify any embeddings, we get the closest elements by calculating distances between the final chair and all beds, couches, etc., and pick the closest. We also take the price from a separate CSV and filter results accordingly.

# Key Differences and Benefits:
- Pretrained model instead of many custom ones:
    - The CLIP model is trained for image retrieval.
    - The CLIP model is better — bigger size, more data, published in 2023.
    - Embeddings are longer and contain more information.
    - Embeddings are all the same size.
    - Embeddings for different categories are compatible.
    - Bonus: can be queried with text.
- Use of a vector DB. Benefits:
    - Quicker.
    - A lot of features.
    - Why write it yourself when you can use a good open-source option???
- Variable sampling variance for different steps:
    - We can control exploration/exploitation at different steps.
- Direct lookup of the closest elements in different categories:
    - Just works better and is more straightforward.

# About the Code
- On the backend server, we need to:
    - [DONE] Substitute the code in `utils.py` with the code in `my_utils.py`.
    - [DONE] Add the file `oracul.py`.
    - [DONE] Add the folder `chroma_cache` with content matching the local DB.
    - [DONE] Install extra libraries; should be only `chromadb==0.5.5`.
    - Test if it runs. Be careful with category names and all paths.
- To add embeddings for new categories:
    - See `ai.py`.
    - Can be done locally, then upload the content of `chroma_cache` to the backend server.
    - Processing 1k images takes ~10min. But you only need to do it once.
- Demo:
    - Install needed libraris, check not_for_server/requirements.txt
    - From the root of this project, run `python3 not_for_server/app.py`.
    - Needs paths to folders `/data` and `/chroma_cache`. `/data` folder needs subfolders `bed`, `chair`, with jpeg images inside. Chroma cache and links csv are used in production, so by default demo will point to those files.
    - Will appear at http://127.0.0.1:8080/.
    - If you use https://ngrok.com/, this demo can be shared with other people.
