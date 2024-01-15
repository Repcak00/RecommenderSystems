from typing import Literal
from random import sample, choice
import torch
import torch.nn as nn


def cosine_similarity(a, b) -> float:
    a_magnitude = torch.sqrt(torch.sum(torch.pow(a, 2)))
    b_magnitude = torch.sqrt(torch.sum(torch.pow(b, 2)))
    return torch.sum(a * b) / (a_magnitude * b_magnitude)


def contrastive_step(
    x: torch.Tensor,
    x_plus: torch.Tensor,
    x_minus: torch.Tensor,
    encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    # obtain embeddings
    emb_x = encoder(x)
    emb_plus = encoder(x_plus)
    emb_minus = encoder(x_minus)

    # obtain similarities
    similarity_plus = cosine_similarity(emb_x, emb_plus)
    similarirty_minus = cosine_similarity(emb_x, emb_minus)

    # optimize loss
    # (maximize similarity between x and x_plus, minimize similarity between x and x_minus)
    optimizer.zero_grad()
    loss = similarirty_minus - similarity_plus
    loss.backward(retain_graph=True)
    optimizer.step()


def get_similar_example(
    embs: torch.Tensor, technique: Literal["subset", "shuffle"]
) -> torch.Tensor:
    """Obtaines images embeddings sequence which is similar to given
    for contrastive learning

    Args:
        embs: images embedding sequence - tensor of shape:
            (n_embeddings, embedding_size)
        technique: string to choose which augmenting technique to use
            when creating similar example
            - ``"subset"`` - get random subsequence
            - ``"shuffle"`` - get sequence of the same length with some elements
            in different order
    """
    with torch.no_grad():
        n_embs = embs.shape[0]
        if n_embs == 1:
            return embs
        if technique == "subset":
            n_embs_subsequence = choice(list(range(1, n_embs)))
            indexes = sorted(sample(list(range(n_embs)), k=n_embs_subsequence))
            return embs[torch.tensor(indexes), :]
        if technique == "shuffle":
            indexes = list(range(n_embs))
            for _ in range(3):  # number of replacements
                # get random index
                index_to_replace = choice(list(range(n_embs - 1)))
                # replace positions: index, (index+1)
                temp = indexes[index_to_replace]
                indexes[index_to_replace] = indexes[index_to_replace + 1]
                indexes[index_to_replace + 1] = temp
            return embs[torch.tensor(indexes), :].clone().detach()
