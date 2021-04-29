"""Somerville Projector

This script generates 2D projections of the somerville_happiness_2018
dataset.

It assumes that it is in the same folder as somerville_happiness_survey.csv

This script requires that `pandas`, sklearn, and numpy be installed within 
the Python environment you are running this script in.

It can be run without arguments on the command line:

	python project_somerville.py

and should generate three files:

- pca.json
- mds.json
- tsne.json

Consult the documentation for all three techniques to understand their usage.

https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import json

df = pd.read_csv('somerville_happiness_survey.csv', index_col=0)

raw_data = df[['city_services', 'housing_cost', 'schools', 'trust_police', 'streets_sidewalks', 'events']]
raw_data_indices = raw_data.index.values
num_data_pts = raw_data.shape[0]

# Just some code loading up junk data, when you finish your solution code, you can comment this out.
# Currently just makes every point have either a 0 or 1 for its x and y coordinate
# pca_results = [[np.random.choice([0,1]), np.random.choice([0,1])] for k in range(num_data_pts)]
# mds_results = [[np.random.choice([0,1]), np.random.choice([0,1])] for k in range(num_data_pts)]
# tsne_results = [[np.random.choice([0,1]), np.random.choice([0,1])] for k in range(num_data_pts)]

# BEGIN_SOLUTION_CODE
pca = PCA(n_components=2)
pca_results = pca.fit_transform(raw_data)

mds = MDS(n_components=2)
mds_results = mds.fit_transform(raw_data)

tsne = TSNE(n_components=2, perplexity=2)
tsne_results = tsne.fit_transform(raw_data)
# Hint: In here, you want to use the projections on the raw_data to fill the variables pca_results, mds_results, and tsne_results
# END_SOLUTION_CODE

# Then, we have some code which gets the projections into an easy-to-deal-with format, as expected
# by the front end.  It also writes out to the json files. You don't have to change this.
pca_results_json_obj = {}
mds_results_json_obj = {}
tsne_results_json_obj = {}

for (j, uid) in enumerate(raw_data_indices):
	pca_results_json_obj[str(uid)] = {'x': str(pca_results[j][0]), 'y': str(pca_results[j][1])}
	mds_results_json_obj[str(uid)] = {'x': str(mds_results[j][0]), 'y': str(mds_results[j][1])}
	tsne_results_json_obj[str(uid)] = {'x': str(tsne_results[j][0]), 'y': str(tsne_results[j][1])}

# Finally, we write them out to json.
with open('pca.json', 'w') as outfile:
    json.dump(pca_results_json_obj, outfile)

with open('mds.json', 'w') as outfile:
    json.dump(mds_results_json_obj, outfile)

with open('tsne.json', 'w') as outfile:
    json.dump(tsne_results_json_obj, outfile)