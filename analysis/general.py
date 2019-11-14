from collections import defaultdict

import csv
import glob
import os


def get_files_for_labels(root_folder, file_to_genre_path):
  # get all the mp3 file names from thhe folder
  file_names = glob.glob(os.path.join(root_folder, '*.mp3'))

  # load the labels
  file_to_labels = dict()
  label_set = set()
  for row in csv.reader(open(file_to_genre_path, 'r')):
    k, v = row
    file_to_labels[int(k)] = int(v)
    label_set.add(v)

  label_map = {fname: file_to_labels[int(os.path.basename(
      fname).split('.')[0])] for fname in file_names}

  inverse_map = defaultdict(list)
  for k, v in label_map.items():
    inverse_map[v].append(k)

  return inverse_map, label_set
