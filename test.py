from pycocoevalcap.eval import calculate_metrics
import numpy as np
import json


def create_dataset(array):
    dataset = {'annotations': []}

    for i, caption in enumerate(array):
        dataset['annotations'].append({
            'image_id': i,
            'caption': caption
        })
    return dataset


def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    train = load_json('./results/val.json')
    datasetGTS = {'annotations': []}
    datasetRES = {'annotations': []}

    for i, image_id in enumerate(train):
        datasetGTS['annotations'].append({
            'image_id': i,
            'caption': train[image_id]['GT']
        })
        datasetRES['annotations'].append({
            'image_id': i,
            'caption': train[image_id]['Pred']
        })

    rng = range(len(train))
    print calculate_metrics(rng, datasetGTS, datasetRES)
