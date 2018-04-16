from pycocoevalcap.eval import calculate_metrics
import numpy as np


def create_dataset(array):
    dataset = {'annotations': []}

    for i, caption in enumerate(array):
        dataset['annotations'].append({
            'image_id': i,
            'caption': caption
        })
    return dataset


if __name__ == '__main__':
    y_pred = np.load('./results/{}'.format('y_pred_val.npz'))['arr_0']
    y_true = np.load('./results/{}'.format('y_true_val.npz'))['arr_0']

    rng = range(len(y_true))
    datasetGTS = create_dataset(y_pred)
    datasetRES = create_dataset(y_true)
    print calculate_metrics(rng, datasetGTS, datasetRES)
