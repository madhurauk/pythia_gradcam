"""Compute avg of correlations and any other results."""
import numpy as np
import pickle
import pdb


print("\n**** PEARSON *****")
with open('CORRS/pythia_pearson_corr.pickle', 'rb') as handle:
    pythia_corr = pickle.load(handle)

with open('CORRS/squint_pearson_corr.pickle', 'rb') as handle:
    squint_corr = pickle.load(handle)

with open('CORRS/sort_pearson_corr.pickle', 'rb') as handle:
    sort_corr = pickle.load(handle)

pythia_avg = np.mean(list(pythia_corr.values()))
print("pythia avg: ", pythia_avg)
squint_avg = np.mean(list(squint_corr.values()))
print("squint avg: ", squint_avg)
sort_avg = np.mean(list(sort_corr.values()))
print("sort avg: ", sort_avg)


print("\n**** SPEARMAN *****")
with open('CORRS/pythia_spearman_corr.pickle', 'rb') as handle:
    pythia_corr = pickle.load(handle)

with open('CORRS/squint_spearman_corr.pickle', 'rb') as handle:
    squint_corr = pickle.load(handle)

with open('CORRS/sort_spearman_corr.pickle', 'rb') as handle:
    sort_corr = pickle.load(handle)

pythia_avg = np.mean(list(pythia_corr.values()))
print("pythia avg: ", pythia_avg)
squint_avg = np.mean(list(squint_corr.values()))
print("squint avg: ", squint_avg)
sort_avg = np.mean(list(sort_corr.values()))
print("sort avg: ", sort_avg)
