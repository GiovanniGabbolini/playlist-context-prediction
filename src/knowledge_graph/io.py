import logging
from tqdm import tqdm
import numpy as np
from src.knowledge_graph.construct_tree import construct_tree
import os
from src.consts import preprocessed_dataset_path


def save_trees(l, start_from_batch=0, folder_name="trees_interestingness"):
    """Build and saves a number of trees, using the method construct_tree.
       The construction happens in batch.

    Args:
        l (list): list of dictionaries containing entity keys
        start_from_batch (int, optional): Defaults to 0.
        folder_name (str, optional): Defaults to "trees_interestingness".

    """
    logging.getLogger("root.features").setLevel(logging.ERROR)

    batch_size = 100

    batch_n = -1
    trees = []
    for idx, d in tqdm(enumerate(l)):

        if idx % batch_size == 0:
            batch_n += 1

        if batch_n >= start_from_batch:

            g = construct_tree(d)

            trees.append(g)

            if len(trees) == batch_size:
                np.save(f"{preprocessed_dataset_path}/{folder_name}/{batch_n}", trees)
                trees = []

    if len(trees) > 0:
        np.save(f"{preprocessed_dataset_path}/{folder_name}/{batch_n}", trees)


def load_trees_generator(folder_name="trees_interestingness"):
    """Returns a generator list able to read all the trees saved in batches by save_trees in a given folder.
    Every element of the generator is a lambda expression, that, if called, returns the corresponding batch.

    Args:
        folder_name (str, optional): Defaults to "trees_interestingness".

    Returns:
        [list]
    """

    def _get_generator(idx, folder_name):
        return lambda: list(
            np.load(
                f"{preprocessed_dataset_path}/{folder_name}/{idx}.npy",
                allow_pickle=True,
            )
        )

    trees_generator = []
    idx = 0
    while os.path.exists(f"{preprocessed_dataset_path}/{folder_name}/{idx}.npy"):
        trees_generator.append(_get_generator(idx, folder_name))
        idx += 1

    return trees_generator


def load_trees(folder_name="trees_interestingness", n_batches=-1):
    """Read back the trees saved in batch by the former method.

    Args:
        folder_name (str, optional)
        n_batches (int, optional): Specify the number of files (or batches) that should be read. If -1, read all the files in the directory

    Returns:
        [list]: list of trees ready to use
    """
    idx = 0
    trees = []
    while True:
        try:
            batch_trees = np.load(
                f"{preprocessed_dataset_path}/{folder_name}/{idx}.npy",
                allow_pickle=True,
            )
            trees += list(batch_trees)
            idx += 1

            if idx == n_batches:
                break

        except FileNotFoundError:
            break
    return trees
