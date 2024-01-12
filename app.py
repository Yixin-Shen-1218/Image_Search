import difflib
import json

import numpy as np
from PIL import Image
from CNN_feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
from SIFT_BOW import img_Retrieve

app = Flask(__name__)

# Read img features extracted by CNN
fe = FeatureExtractor()
features = []
img_paths = []

for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(f"{feature_path}"))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)

# Read img features extracted by SIFT
SIFT_features = np.zeros((140, 1024), 'float32')
i = 0
for SIFT_feature_path in Path("./static/SIFT_feature").glob("*.npy"):
    SIFT_features[i] = np.load(f"{SIFT_feature_path}")
    i = i + 1
SIFT_features = np.array(SIFT_features)

# Read images and the corresponding labels as dictionary
with open("static/img_label/label.json", encoding='utf-8') as js:
    json_data = js.read()
    img_label = json.loads(json_data)

# the flower species in the database
species = ['astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy', 'carnation', 'common daisy',
           'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip', 'water lily']


# calculate the similarity between two strings
def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # read the post elements
        file = request.files["query_img"]
        text = request.form["search_text"]
        method = request.form["select_menu"]
        # print(file)
        # print(text)
        # print(method)

        # if text and img are all not empty, search by img
        if text != '' and file.filename != '':
            # save query img
            img = Image.open(file.stream)  # PIL image
            uploaded_img_path = "static/uploaded/" + \
                                datetime.now().isoformat().replace(":", ".") + "_" + file.filename
            img.save(uploaded_img_path)
            print(img)

            # Run search by CNN
            if method == 'CNN':
                query = fe.extract(img)
                dists = np.linalg.norm(features - query, axis=1)  # L2 distances to the features
                ids = np.argsort(dists)[:10]
                scores = [(dists[id], img_paths[id]) for id in ids]
                print(scores)
                print(len(scores))
            else:
                # Run search by SIFT and BOW
                SIFT_centres = np.load('static/SIFT_centres/SIFT_centres.npy')
                dists, ids = img_Retrieve(uploaded_img_path, SIFT_features, SIFT_centres)
                scores = [(dists[id], img_paths[id]) for id in ids]

            return render_template("index.html", search_query_text=text, query_path=uploaded_img_path, scores=scores)
        # if text is not empty, search by text
        elif text != '':
            similarities = []
            for specie in species:
                similarity = string_similar(text, specie)
                similarities.append(similarity)
            max_index = similarities.index(max(similarities))

            print(similarities)
            print("max specie = ", species[max_index])

            match_specie = species[max_index]

            img_path_list = []
            for item in img_label:
                if list(item.values())[0] == match_specie:
                    img_path_list.append("./static/img/" + list(item.keys())[0])

            print(img_path_list)

            scores = [(match_specie, img_path_list[id]) for id in range(10)]

            return render_template("index.html", search_query_text=text, scores=scores)
        # if img is not empty, search by img
        elif file.filename != '':
            # save query img
            img = Image.open(file.stream)  # PIL image
            uploaded_img_path = "static/uploaded/" + \
                                datetime.now().isoformat().replace(":", ".") + "_" + file.filename
            img.save(uploaded_img_path)

            # Run search by CNN
            if method == 'CNN':
                query = fe.extract(img)
                dists = np.linalg.norm(features - query, axis=1)  # L2 distances to the features
                ids = np.argsort(dists)[:10]
                scores = [(dists[id], img_paths[id]) for id in ids]
                print(scores)
            else:
                # Run search by SIFT and BOW
                SIFT_centres = np.load('static/SIFT_centres/SIFT_centres.npy')
                dists, ids = img_Retrieve(uploaded_img_path, SIFT_features, SIFT_centres)
                scores = [(dists[id], img_paths[id]) for id in ids]

            return render_template("index.html", query_path=uploaded_img_path, scores=scores)
        # if there is nothing posted, prompt warning
        else:
            print("nothing")
            return render_template("index.html", text_empty='Please enter a name or image of flower',
                                   image_empty='Please enter a name or image of flower')

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run()
