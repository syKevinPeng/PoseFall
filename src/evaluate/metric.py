import numpy as np
from scipy import linalg
import torch

def calculate_activation_statistics(activations):
    """Calculation of the statistics used by the FID.
    Params:
    -- activations : List of activations. The dimensions are
                     (N, D), where N is the number of activations
                     and D their dimensionality.
    Returns:
    -- mu    : The mean over the activations.
    -- sigma : The covariance matrix of the activations.
    """
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.+
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calculate_fid(statistics_1, statistics_2):
    return calculate_frechet_distance(statistics_1[0], statistics_1[1],
                                      statistics_2[0], statistics_2[1])


def calculate_accuracy(model, motion_loader, num_labels, classifier, device):
    confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)
    with torch.no_grad():
        for batch in motion_loader:
            batch_prob = classifier(batch)["yhat"]
            batch_pred = batch_prob.max(dim=1).indices
            for label, pred in zip(batch["y"], batch_pred):
                confusion[label][pred] += 1

    accuracy = torch.trace(confusion)/torch.sum(confusion)
    return accuracy.item(), confusion


import torch
import numpy as np

def compute_hamming_score(self, batch):
    # apply sigmoid to yhat
    yhat = torch.sigmoid(batch["yhat"]).round()
    ygt = batch["y"]
    # calculate how many elements are equal
    hamming = torch.sum(yhat == ygt).item()
    return hamming/len(ygt)

def compute_exact_match(self, batch):
    attr_size = batch["attribute_size"]
    yhat = torch.sigmoid(batch["yhat"]).round()
    ygt = batch["y"]
    return accuracy_score(ygt.cpu().numpy(), yhat.detach().cpu().numpy())


# from action2motion
def calculate_diversity(activations, labels, num_labels, seed=None):
    diversity_times = 200
    multimodality_times = 20
    labels = labels.astype(np.longfloat)
    num_motions = len(labels)

    diversity = 0

    if seed is not None:
        np.random.seed(seed)
        
    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += torch.dist(activations[first_idx, :],
                                activations[second_idx, :])
    diversity /= diversity_times

    # multimodality = 0
    # label_quotas = np.repeat(multimodality_times, num_labels)
    # while np.any(label_quotas > 0):
    #     # print(label_quotas)
    #     first_idx = np.random.randint(0, num_motions)
    #     first_label = labels[first_idx]
    #     print(f'first label: {first_label}')
    #     if not label_quotas[first_label]:
    #         continue

    #     second_idx = np.random.randint(0, num_motions)
    #     second_label = labels[second_idx]
    #     while first_label != second_label:
    #         second_idx = np.random.randint(0, num_motions)
    #         second_label = labels[second_idx]

    #     label_quotas[first_label] -= 1

    #     first_activation = activations[first_idx, :]
    #     second_activation = activations[second_idx, :]
    #     multimodality += torch.dist(first_activation,
    #                                 second_activation)

    # multimodality /= (multimodality_times * num_labels)

    return diversity.item()#, multimodality.item()