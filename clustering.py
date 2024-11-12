import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import imageio

import hdbscan
import torch
import umap
import umap.plot
import torchvision.transforms.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid



# matplotlib.use('TkAgg', force=True)
# matplotlib.use('Agg')
plt.ion()


def load_files(folder_path):
    loaded_files_dict = {
        # 'dino_reps': torch.load(f"{folder_path}/reps_{folder_path.split('_')[0]}.pt", map_location='cpu'),
        'days': torch.load(f"{folder_path}/days_{folder_path.split('_')[0]}.pt", map_location='cpu'),
        'bags': torch.load(f"{folder_path}/bags_{folder_path.split('_')[0]}.pt", map_location='cpu'),
        'umap': joblib.load(f"{folder_path}/umap_{folder_path.split('_')[0]}.sav"),
        # 'umap_reps': np.load(f"{folder_path}/umap_embeddings.npy"),
    }

    return loaded_files_dict


def dbscan_clustering(embeddings=None, folder_path=None, min_cluster_size=5000):
    if folder_path is not None and os.path.isfile(f"{folder_path}/dbscan_labels.npy"):
        return np.load(f"{folder_path}/dbscan_labels.npy")

    if embeddings is None:
        raise ValueError('expecting embeddings')

    hdbscan_labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=min_cluster_size).fit_predict(embeddings)
    clustering_percent = (hdbscan_labels >= 0).sum() / len(hdbscan_labels)
    print(f"{clustering_percent*100:.3f}% of the data was clustered")
    np.save(f"{folder_path}/dbscan_labels", hdbscan_labels)

    return hdbscan_labels


def make_umap_points_figures(results_dict, labels_keys=None, figures_dir=None):
    labels_keys = [None, 'bags', 'days', 'dbscan_labels'] if labels_keys is None else labels_keys
    figures_dir = 'figures' if figures_dir is None else figures_dir

    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    for lk in labels_keys:
        if lk is None:
            umap.plot.points(results_dict['umap'])
        else:
            umap.plot.points(results_dict['umap'], labels=results_dict[lk])
        plt.title(f"Umap points {lk}")
        plt.savefig(f"{figures_dir}/umap_points_{lk}")
        plt.close()


def make_days_progress(umap_mapper, days_labels, figures_dir=None):
    figures_dir = 'figures' if figures_dir is None else figures_dir
    figures_dir += '/days_animation'

    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    range_days = range(int(days_labels.min()), int(days_labels.max())+1)

    for i in range_days:
        umap.plot.points(umap_mapper, labels=(days_labels <= i))
        plt.legend()
        plt.savefig(f"{figures_dir}/umap_points_day_{i}")
        plt.close()

    with imageio.get_writer(f"{figures_dir}/cells_days.gif", mode='i') as writer:
        for i in range_days:
            for _ in range(5):
                image = imageio.imread(f"{figures_dir}/umap_points_day_{i}.png")
                writer.append_data(image)


class SubSampledUmapMock:
    def __init__(self, umap_reducer, sub_sampling_size):
        self.indices = np.random.choice(len(umap_reducer.embedding_), sub_sampling_size, replace=False)
        self.embedding_ = umap_reducer.embedding_[self.indices]


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=True)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def plot_mutual_labels_heatmap(x, y, xlabel=None, ylabel=None):
    xrange = np.arange(int(max(x.min(), 0)), int(x.max()+1))
    yrange = np.arange(int(max(y.min(), 0)), int(y.max()+1))

    m = np.array([[len(np.where((x == xl) & (y == yl))[0]) / len(np.where(x == xl)[0]) for xl in xrange] for yl in
                  yrange])

    fig, ax = plt.subplots()
    im = ax.imshow(m)

    plt.colorbar(im)

    # Show all ticks and label them with the respective list entries
    # ax.set_xticks(xrange, labels=xlabel)
    # ax.set_yticks(yrange, labels=ylabel)
    # ax.set_xticks(xrange)
    # ax.set_yticks(yrange)
    plt.xticks(np.arange(len(xrange)), labels=xrange)
    plt.yticks(np.arange(len(yrange)), labels=yrange)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # # Loop over data dimensions and create text annotations.
    # for i in xrange:
    #     for j in yrange:
    #         text = ax.text(j, i, m[i, j],
    #                        ha="center", va="center", color="w")

    ax.set_title(f"{xlabel}-{ylabel} mutual distribution")
    fig.tight_layout()
    plt.show()


def my_main(conf):
    files_folder = conf+'_files'
    data_dict = load_files(files_folder)
    data_dict['dbscan_labels'] = dbscan_clustering(data_dict['umap'].embedding_, folder_path=files_folder)

    make_umap_points_figures(data_dict, figures_dir=f"figures/{conf}")

    make_days_progress(data_dict['umap'], data_dict['days'], figures_dir=f"figures/{conf}")


if __name__ == '__main__':
    # my_main('finetuned')
    # my_main('pretrained')

    dataset = datasets.ImageFolder('processed_data', transform=transforms.ToTensor())

    data_dict = load_files('finetuned_files')
    data_dict['dbscan_labels'] = dbscan_clustering(data_dict['umap'].embedding_, folder_path='finetuned_files')

    dbscan_labels = dbscan_clustering(folder_path='finetuned_files')

    idx = 1
    l_indices = np.random.choice(np.where(dbscan_labels == idx)[0], 16, replace=False)
    show(make_grid([dataset[i][0] for i in l_indices]))

    print('done')

