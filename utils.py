import random
import math
import numpy as np
import argparse
import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms
from image_list import ImageList
import os
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances

def get_optimizer(args, config_optim, parameters):
    if config_optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config_optim.lr, weight_decay=config_optim.weight_decay,
                          betas=(config_optim.beta1, 0.999), amsgrad=config_optim.amsgrad,
                          eps=config_optim.eps)
    elif config_optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config_optim.lr, weight_decay=config_optim.weight_decay)
    elif config_optim.optimizer == 'SGD':
        if args.setting == 'ADA':
            return optim.SGD(parameters, lr=config_optim.lr, momentum=0.9, weight_decay=config_optim.weight_decay, nesterov = True)
        else:
            return optim.SGD(parameters)
    elif config_optim.optimizer == 'Adadelta':
        return optim.Adadelta(parameters)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config_optim.optimizer))

def get_dataset(args, config):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if config.data.dataset == "visda":
        train_transform = transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    elif config.data.dataset == "office-31":
        train_transform = transforms.Compose([
            ResizeImage(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    elif config.data.dataset == "office-home":
        train_transform = transforms.Compose([
            ResizeImage(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        raise NotImplementedError(
            "Options: visda, office, home.")

    val_tranform = transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    s_dset_path = os.path.join(config.data.dataroot, '{}_list.txt'.format(config.data.source_domain))
    t_dset_path = os.path.join(config.data.dataroot, '{}_list.txt'.format(config.data.target_domain))

    train_source_dataset = ImageList(s_dset_path, transform=train_transform)
    test_source_dataset = ImageList(s_dset_path, transform=val_tranform)
    train_target_dataset = ImageList(t_dset_path, transform=train_transform)
    val_dataset = ImageList(t_dset_path, transform=val_tranform)
    tgt_selected_ds = ImageList(empty=True, transform=train_transform)

    return train_source_dataset, test_source_dataset, train_target_dataset, val_dataset, tgt_selected_ds

class ResizeImage(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

def accuracy(output, target, topk=(1,)):

    maxk = min(max(topk), output.size()[1])
    # output = torch.softmax(-(output - 1)**2,  dim=-1)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def cast_label_to_one_hot(y_labels_batch, config, return_prototype=True):

    y_one_hot_batch = nn.functional.one_hot(y_labels_batch, num_classes=config.data.num_classes).float()

    return y_one_hot_batch

def reparameterize(z_mean, z_var, distribution='normal'):
    if distribution == 'normal':
        q_z = torch.distributions.normal.Normal(z_mean, z_var)
    else:
        raise NotImplemented
    return q_z

def reparameterize2(z_mean, z_var, z_dim = 256, distribution='normal'):
    if distribution == 'normal':
        q_z = torch.distributions.normal.Normal(z_mean, z_var)
        p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
    else:
        raise NotImplemented

    return q_z, p_z

def compute_distill_loss(stu_logit, tea_logit, labels_target=None, T=1.0):
    tea_out = F.softmax(tea_logit)
    conf_tea = torch.max(tea_out, 1)[0]
    unsup_mask = conf_mask = (conf_tea > 0.9).float()
    unsup_mask_count = conf_mask_count = conf_mask.sum()

    aug_loss = kl_div_with_logit(stu_logit / T, tea_logit / T)
    unsup_loss = (aug_loss * unsup_mask).mean()

    return unsup_loss, conf_mask_count, unsup_mask_count

def kl_div_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = (q *logq).sum(dim=1).mean(dim=0)
    qlogp = (q *logp).sum(dim=1).mean(dim=0)

    return qlogq - qlogp

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def Conditional_Adversarial_Loss(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)), coeff)
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)), coeff)       
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 
    

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
    return optimizer

def kmeans_plus_plus_opt(X1, X2, n_clusters, init=[0], random_state=np.random.RandomState(1234), n_local_trials=None):
	n_samples, n_feat1 = X1.shape
	_, n_feat2 = X2.shape
	# x_squared_norms = row_norms(X, squared=True)
	centers1 = np.empty((n_clusters+len(init)-1, n_feat1), dtype=X1.dtype)
	centers2 = np.empty((n_clusters+len(init)-1, n_feat2), dtype=X1.dtype)

	idxs = np.empty((n_clusters+len(init)-1,), dtype=np.long)

	# Set the number of local seeding trials if none is given
	if n_local_trials is None:
		# This is what Arthur/Vassilvitskii tried, but did not report
		# specific results for other than mentioning in the conclusion
		# that it helped.
		n_local_trials = 2 + int(np.log(n_clusters))

	# Pick first center randomly
	center_id = init

	centers1[:len(init)] = X1[center_id]
	centers2[:len(init)] = X2[center_id]
	idxs[:len(init)] = center_id

	# Initialize list of closest distances and calculate current potential
	distance_to_candidates = outer_product_opt(centers1[:len(init)], centers2[:len(init)], X1, X2).reshape(len(init), -1)

	candidates_pot = distance_to_candidates.sum(axis=1)
	best_candidate = np.argmin(candidates_pot)
	current_pot = candidates_pot[best_candidate]
	closest_dist_sq = distance_to_candidates[best_candidate]

	# Pick the remaining n_clusters-1 points
	for c in range(len(init), len(init)+n_clusters-1):
		# Choose center candidates by sampling with probability proportional
		# to the squared distance to the closest existing center
		rand_vals = random_state.random_sample(n_local_trials) * current_pot
		candidate_ids = np.searchsorted(closest_dist_sq.cumsum(),
										rand_vals)
		# XXX: numerical imprecision can result in a candidate_id out of range
		np.clip(candidate_ids, None, closest_dist_sq.size - 1,
				out=candidate_ids)

		# Compute distances to center candidates
		distance_to_candidates = outer_product_opt(X1[candidate_ids], X2[candidate_ids], X1, X2).reshape(len(candidate_ids), -1)

		# update closest distances squared and potential for each candidate
		np.minimum(closest_dist_sq, distance_to_candidates,
				   out=distance_to_candidates)
		candidates_pot = distance_to_candidates.sum(axis=1)

		# Decide which candidate is the best
		best_candidate = np.argmin(candidates_pot)
		current_pot = candidates_pot[best_candidate]
		closest_dist_sq = distance_to_candidates[best_candidate]
		best_candidate = candidate_ids[best_candidate]

		idxs[c] = best_candidate

	return None, idxs[len(init)-1:]

def row_norms(X, squared=False):
	norms = np.einsum('ij,ij->i', X, X)

	if not squared:
		np.sqrt(norms, norms)
	return norms

def outer_product_opt(c1, d1, c2, d2):
	B1, B2 = c1.shape[0], c2.shape[0]
	t1 = np.matmul(np.matmul(c1[:, None, :], c1[:, None, :].swapaxes(2, 1)), np.matmul(d1[:, None, :], d1[:, None, :].swapaxes(2, 1)))
	t2 = np.matmul(np.matmul(c2[:, None, :], c2[:, None, :].swapaxes(2, 1)), np.matmul(d2[:, None, :], d2[:, None, :].swapaxes(2, 1)))
	t3 = np.matmul(c1, c2.T) * np.matmul(d1, d2.T)
	t1 = t1.reshape(B1, 1).repeat(B2, axis=1)
	t2 = t2.reshape(1, B2).repeat(B1, axis=0)
	return t1 + t2 - 2*t3


def furthest_first(X, X_set, n):
    m = np.shape(X)[0]
    if np.shape(X_set)[0] == 0:
        min_dist = np.tile(float("inf"), m)
    else:
        dist_ctr = pairwise_distances(X, X_set)
        min_dist = np.amin(dist_ctr, axis=1)

    idxs = []

    for i in range(n):
        idx = min_dist.argmax()
        idxs.append(idx)
        dist_new_ctr = pairwise_distances(X, X[[idx], :])
        for j in range(m):
            min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

    return idxs