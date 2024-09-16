import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)
from oracul import PenatOracul
# I fucking hate relative imports. I didnt figure out how to do it properly

from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import os
import chromadb
from oracul import PenatOracul  # Import the PenatOracul class

app = Flask(__name__)
app.secret_key = 'your_secret_key'

CHROMA_CACHE_PATH = "./data/chroma_cache"
LINKS_CSV_PATH = "./data/links.csv"
IMAGES_DATA_PATH= '/Users/ksc/penat/data'

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_CACHE_PATH)
assert 'common' in [x.name for x in chroma_client.list_collections()], f"bad client, it has only collections: {[x.name for x in chroma_client.list_collections()]}"

possible_items = ['bed', 'chair', 'closet', 'couch', 'dining_table', 'potted_plant']

# Define the full limits for the final step
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
        "max": 1e6
    },
    "closet": {
        "min": 0,
        "max": 1e6
    }
}

@app.route('/')
def index():
    return render_template('index.html', possible_items=possible_items)

@app.route('/select_category', methods=['POST'])
def select_category():
    item_index = int(request.form['item_index'])
    main_type = possible_items[item_index]
    session['main_type'] = main_type
    session['step'] = 1  # Initialize step
    session['selected_image'] = None

    # Initialize PenatOracul
    collection = chroma_client.get_collection("common")
    oracul = PenatOracul(
        collection=collection,
        price_csv_path='/Users/ksc/penat/links.csv',  # Update with your path
        not_found_final_res={
            "name": "Не найдено",
            "image": 'https://static01.nyt.com/images/2016/09/28/us/17xp-pepethefrog_web1/28xp-pepefrog-superJumbo.jpg',
            "link": None,
            "price": None,
            "limits": {"min": 0, "max": 1e6, "loaded": True}
        },
        main_type=main_type,
        panel_size=16  # Ensure panel_size is 16
    )
    # Store necessary parameters in session
    session['panel_size'] = 16

    # Get the starting images
    image_ids = oracul.run_general_step(step_n=1)
    session['image_ids'] = image_ids
    session['oracul_params'] = {
        'main_type': main_type,
        'panel_size': 16
    }

    return redirect(url_for('cycle'))

@app.route('/cycle', methods=['GET', 'POST'])
def cycle():
    oracul_params = session.get('oracul_params')
    if oracul_params is None:
        return redirect(url_for('index'))

    main_type = oracul_params['main_type']
    panel_size = oracul_params['panel_size']

    # Re-initialize PenatOracul
    collection = chroma_client.get_collection("common")
    oracul = PenatOracul(
        collection=collection,
        price_csv_path='/Users/ksc/penat/links.csv',  # Update with your path
        not_found_final_res={
            "name": "Не найдено",
            "image": "",
            "link": None,
            "price": None,
            "limits": {"min": 0, "max": 1e6, "loaded": True}
        },
        main_type=main_type,
        panel_size=panel_size
    )

    if request.method == 'POST':
        selected_index = int(request.form['selected_index'])
        image_ids = session['image_ids']
        selected_image = image_ids[selected_index]
        session['selected_image'] = selected_image
        step = session.get('step', 1) + 1  # Increment step
        session['step'] = step

        if step <= 3:
            # Get the next set of images
            image_ids = oracul.run_general_step(step_n=step, image=selected_image)
            session['image_ids'] = image_ids

            # Prepare image titles
            image_titles = [f"{i} | " for i in range(len(image_ids))]

            images_and_titles = zip(image_ids, image_ids, image_titles)  # IDs are used for filenames and display

            return render_template('cycle.html', images_and_titles=images_and_titles)
        else:
            # Final step
            limits = full_lims  # You can adjust limits as needed
            final_results = oracul.run_final_step(selected_image, limits=limits)

            session['final_results'] = final_results

            return redirect(url_for('final_step'))

    else:
        # For GET request, use existing image_ids from session
        image_ids = session.get('image_ids')
        step = session.get('step', 1)

        if image_ids is None or step > 3:
            # Start from step 1 if no image_ids in session or step exceeded
            image_ids = oracul.run_general_step(step_n=1)
            session['image_ids'] = image_ids
            session['step'] = 1

        # Prepare image titles
        image_titles = [f"{i} | " for i in range(len(image_ids))]

        images_and_titles = zip(image_ids, image_ids, image_titles)  # IDs are used for filenames and display

        return render_template('cycle.html', images_and_titles=images_and_titles)

@app.route('/final_step')
def final_step():
    final_results = session.get('final_results')
    if final_results is None:
        return redirect(url_for('index'))

    return render_template('final.html', final_results=final_results)

@app.route('/images/<path:filename>')
def serve_image(filename):
    # Check if the filename is a URL (for the not_found_final_res image)
    if filename.startswith('http://') or filename.startswith('https://'):
        return redirect(filename)

    # For other images, serve from the appropriate folder
    # Since the images can be from different categories, we need to handle that
    # Assuming images are stored in folders named after their categories

    # First, check if the image exists in any of the category folders
    possible_folders = [os.path.join(IMAGES_DATA_PATH, cat) for cat in possible_items]
    for folder_path in possible_folders:
        image_path = os.path.join(folder_path, filename + '.jpeg')
        if os.path.exists(image_path):
            return send_from_directory(folder_path, filename + '.jpeg')

    # If image not found, return a placeholder or handle accordingly
    # For simplicity, redirect to a placeholder image
    placeholder_url = 'https://via.placeholder.com/200x200?text=Image+Not+Found'
    return redirect(placeholder_url)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
