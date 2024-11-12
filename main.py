import torch
from torchvision import datasets, transforms
import umap
import joblib
import numpy as np


from dino.utils import load_pretrained_weights
import dino.vision_transformer as vits

BatchSize = 256
DINOArch = 'vit_small'
PatchSize = 8


def load_dino_model(dino_arch=DINOArch, patch_size=PatchSize, path_to_saved_model=None):
    pre_trained_dino_model = vits.__dict__[dino_arch](patch_size=patch_size, num_classes=0)
    if path_to_saved_model is not None:
        load_pretrained_weights(pre_trained_dino_model, path_to_saved_model, 'teacher', DINOArch, 8)
    else:
        pre_trained_dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')

    pre_trained_dino_model.cuda()
    pre_trained_dino_model.eval()

    return pre_trained_dino_model


def get_samples(dataloader, model):
    imgs, labels = next(iter(dataloader))
    imgs = imgs.to('cuda')

    return imgs, model(imgs)


def calc_and_save_representations(model, dataloader, model_name):
    classes = dataloader.dataset.classes

    reps = torch.Tensor()
    days = torch.Tensor()
    bags = torch.Tensor()

    for i, (imgs, labels) in enumerate(dataloader):

        imgs = imgs.to('cuda')

        if i % 500 == 0:
            print(f"finished {i*500} iterations")

        output = model(imgs)
        reps = torch.cat([reps, output.to('cpu')])

        # convert index label to class name
        labels.apply_(lambda x: int(classes[x]))
        # extract day and bag from class name
        days = torch.cat([days, labels % 100])
        bags = torch.cat([bags, labels//100])

    torch.save(reps, f'reps_{model_name}.pt')
    torch.save(days, f'days_{model_name}.pt')
    torch.save(bags, f'bags_{model_name}.pt')

    print(f"finish model {model_name}")


def load_dino_embedding_files(folder_path):
    dino_reps_ = torch.load(f"{folder_path}/reps_{folder_path.split('_')[0]}.pt", map_location='cpu')
    days_ = torch.load(f"{folder_path}/days_{folder_path.split('_')[0]}.pt", map_location='cpu')
    bags_ = torch.load(f"{folder_path}/bags_{folder_path.split('_')[0]}.pt", map_location='cpu')

    return dino_reps_, days_, bags_


def calc_umap(inputs):
    # reducer = umap.UMAP()
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.0)
    reducer.fit(inputs)
    return reducer


if __name__ == '__main__':
    # Create DataLoader
    dataset = datasets.ImageFolder('processed_data', transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BatchSize,
        shuffle=False,
    )

    # Load DINO model
    dino_model = load_dino_model()                                                                   # pre-trained DINO
    ft_dino_model = load_dino_model(path_to_saved_model='../finetuned_dino/checkpoint.pth')          # finetuned DINO

    # Calculate Model's embeddings
    calc_and_save_representations(dino_model, data_loader, 'pretrained')
    calc_and_save_representations(ft_dino_model, data_loader, 'finetuned')

    # Load DINO embeddings
    dino_reps, days, bags = load_dino_embedding_files('pretrained_files')

    # Create UMAP reducer object and save it
    umap_reducer = calc_umap(dino_reps.numpy())
    joblib.dump(umap_reducer, 'pretrained_files/umap_pretrained.sav')
    np.save('pretrained_files/umap_embeddings2', umap_reducer.embedding_)

    # Load DINO embeddings
    dino_reps, days, bags = load_dino_embedding_files('finetuned_files')

    # Create UMAP reducer object and save it
    umap_reducer = calc_umap(dino_reps.numpy())
    joblib.dump(umap_reducer, 'finetuned_files/umap_finetuned.sav')
    np.save('finetuned_files/umap_embeddings2', umap_reducer.embedding_)

    print('done')
