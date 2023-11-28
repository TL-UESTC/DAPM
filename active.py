import random
import numpy as np
import torch
import torch.utils.data as data
from diffusion_utils import *
from utils import *
from scipy.stats import ttest_rel, ttest_ind 

def get_active_func(name):
    if name == 'ttest':
        return TTest_active
    elif name == 'random':
        return RAND_active
    elif name == 'entropy':
        return ENT_active
    elif name == 'margin':
        return MAR_active
    elif name == 'badge':
        return BADGE_active
    elif name == 'coreset':
        return CORESET_active
    elif name == 'variation_ttest':
        return variation_ttest_active
    elif name == 'PIW_ttest_active':
        return PIW_ttest_active
    else:
        raise Exception("Not Implemented.")

def RAND_active(model, tgt_unlabeled_ds, tgt_selected_ds, active_ratio, totality):
    length = len(tgt_unlabeled_ds.samples)
    index = random.sample(range(length), round(totality * active_ratio))

    tgt_selected_ds.add_item(tgt_unlabeled_ds.samples[index])
    tgt_unlabeled_ds.remove_item(index)

    return index

def ENT_active(model, tgt_unlabeled_ds, tgt_selected_ds, active_ratio, totality):
    config = model.config
    
    test_loader = data.DataLoader(
        tgt_unlabeled_ds,
        batch_size=config.testing.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers)
    
    model.dif_model.eval()
    model.cond_pred_model.eval()

    entropy_by_batch_list = []

    selected_num = round(totality * active_ratio)

    with torch.no_grad():
        for step, feature_label_set in enumerate(test_loader):
            x_batch, y_labels_batch = feature_label_set
            x_batch = x_batch.to(model.device)
            # flattened features from the backbone network
            x_batch = model.backbone(x_batch)
            y_0_hat_batch_logit, z_batch  = model.compute_guiding_prediction(x_batch, True)
            y_0_hat_batch_sm = y_0_hat_batch_logit.softmax(dim=1)
            entropy_batch = Entropy(y_0_hat_batch_sm)

            if len(entropy_by_batch_list) == 0:
                entropy_by_batch_list = entropy_batch.cpu().numpy()

            else:
                entropy_by_batch_list = np.concatenate([entropy_by_batch_list, entropy_batch.cpu().numpy()], axis=0)

    ent_selected_index = entropy_by_batch_list.argsort()[-1 * selected_num:-1]


    tgt_selected_ds.add_item(tgt_unlabeled_ds.samples[ent_selected_index])
    tgt_unlabeled_ds.remove_item(ent_selected_index)

    return ent_selected_index

def MAR_active(model, tgt_unlabeled_ds, tgt_selected_ds, active_ratio, totality):
    config = model.config
    
    test_loader = data.DataLoader(
        tgt_unlabeled_ds,
        batch_size=config.testing.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers)
    
    model.dif_model.eval()
    model.cond_pred_model.eval()

    margin_by_batch_list = []

    selected_num = round(totality * active_ratio)

    with torch.no_grad():
        for step, feature_label_set in enumerate(test_loader):
            x_batch, y_labels_batch = feature_label_set
            x_batch = x_batch.to(model.device)
            # flattened features from the backbone network
            x_batch = model.backbone(x_batch)
            y_0_hat_batch_logit, z_batch  = model.compute_guiding_prediction(x_batch, True)
            y_0_hat_batch_sm = y_0_hat_batch_logit.softmax(dim=1)

            two_most_probable_classes_idx = y_0_hat_batch_sm.argsort(dim=1, descending=True)[:, :2]
            gen_y_2_class_probs = torch.gather(y_0_hat_batch_sm, dim=1,
                                               index=two_most_probable_classes_idx)  # (batch_size, 2)            

            gen_y_2_class_prob_diff = abs(gen_y_2_class_probs[:, 1] \
                                        - gen_y_2_class_probs[:, 0])  # (batch_size, )

            if len(margin_by_batch_list) == 0:
                margin_by_batch_list = gen_y_2_class_prob_diff.cpu().numpy()

            else:
                margin_by_batch_list = np.concatenate([margin_by_batch_list, gen_y_2_class_prob_diff.cpu().numpy()], axis=0)

    mar_selected_index = margin_by_batch_list.argsort()[:selected_num]

    tgt_selected_ds.add_item(tgt_unlabeled_ds.samples[mar_selected_index])
    tgt_unlabeled_ds.remove_item(mar_selected_index)

    return mar_selected_index

def BADGE_active(model, tgt_unlabeled_ds, tgt_selected_ds, active_ratio, totality):
    config = model.config
    test_loader = data.DataLoader(
        tgt_unlabeled_ds,
        batch_size=config.testing.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers)
    
    model.dif_model.eval()
    model.cond_pred_model.eval()

    tgt_pen_emb_by_batch_list = [] # embedding list
    tgt_emb_by_batch_list = [] # raw predict list
    tgt_lab_by_batch_list = [] # label list 
    tgt_preds_by_batch_list = [] # predicted label list

    selected_num = round(totality * active_ratio)

    with torch.no_grad():
        for batch_idx, feature_label_set in enumerate(test_loader):
            x_batch, y_labels_batch = feature_label_set
            x_batch = x_batch.to(model.device)
            # flattened features from the backbone network
            x_batch = model.backbone(x_batch)
            e1, e2  = model.compute_guiding_prediction(x_batch, True)
            if len(tgt_pen_emb_by_batch_list) == 0:
                tgt_pen_emb_by_batch_list = e2.cpu() 
                tgt_emb_by_batch_list = e1.cpu()
                tgt_lab_by_batch_list = y_labels_batch.cpu()
                tgt_preds_by_batch_list = e1.argmax(dim=1, keepdim=True).squeeze()
            else:
                tgt_pen_emb_by_batch_list = torch.cat((tgt_pen_emb_by_batch_list, e2.cpu()),0)
                tgt_emb_by_batch_list = torch.cat((tgt_emb_by_batch_list, e1.cpu()),0)
                tgt_lab_by_batch_list = torch.cat((tgt_lab_by_batch_list, y_labels_batch.cpu()),0)
                try:
                    tgt_preds_by_batch_list = torch.cat((tgt_preds_by_batch_list, e1.argmax(dim=1, keepdim=True).squeeze()),0)
                except:
                    tgt_preds_by_batch_list = torch.cat((tgt_preds_by_batch_list, e1.argmax(dim=1)),0)

    # Compute uncertainty gradient
    tgt_scores = nn.Softmax(dim=1)(tgt_emb_by_batch_list)
    tgt_scores_delta = torch.zeros_like(tgt_scores)
    tgt_scores_delta[torch.arange(len(tgt_scores_delta)), tgt_preds_by_batch_list.long()] = 1
    
    # Uncertainty embedding
    badge_uncertainty = (tgt_scores-tgt_scores_delta)

    # Seed with maximum uncertainty example
    max_norm = row_norms(badge_uncertainty.numpy()).argmax()

    _, q_idxs = kmeans_plus_plus_opt(badge_uncertainty.numpy(), tgt_pen_emb_by_batch_list.numpy(), selected_num, init=[max_norm])

    tgt_selected_ds.add_item(tgt_unlabeled_ds.samples[q_idxs])
    tgt_unlabeled_ds.remove_item(q_idxs)

    return q_idxs

def CORESET_active(model, tgt_unlabeled_ds, tgt_selected_ds, active_ratio, totality):
    config = model.config
    uld_tgt_loader = data.DataLoader(
        tgt_unlabeled_ds,
        batch_size=config.testing.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers)

    ld_tgt_loader = data.DataLoader(
        tgt_selected_ds,
        batch_size=config.testing.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers)

    model.dif_model.eval()
    model.cond_pred_model.eval()

    uld_embedding_by_batch_list = []
    ld_embedding_by_batch_list = []

    selected_num = round(totality * active_ratio)

    with torch.no_grad():
        for step, feature_label_set in enumerate(uld_tgt_loader):
            x_batch, y_labels_batch = feature_label_set
            x_batch = x_batch.to(model.device)
            # flattened features from the backbone network
            x_batch = model.backbone(x_batch)
            y_0_hat_batch_logit, z_batch  = model.compute_guiding_prediction(x_batch, True)
            if len(uld_embedding_by_batch_list) == 0:
                uld_embedding_by_batch_list = z_batch.cpu().numpy()

            else:
                uld_embedding_by_batch_list = np.concatenate([uld_embedding_by_batch_list, z_batch.cpu().numpy()], axis=0)
        if not tgt_selected_ds.empty:
            for step, feature_label_set in enumerate(ld_tgt_loader):
                x_batch, y_labels_batch = feature_label_set
                x_batch = x_batch.to(model.device)
                # flattened features from the backbone network
                x_batch = model.backbone(x_batch)
                y_0_hat_batch_logit, z_batch  = model.compute_guiding_prediction(x_batch, True)
                if len(ld_embedding_by_batch_list) == 0:
                    ld_embedding_by_batch_list = z_batch.cpu().numpy()

                else:
                    ld_embedding_by_batch_list = np.concatenate([ld_embedding_by_batch_list, z_batch.cpu().numpy()], axis=0)
        else:
            ld_embedding_by_batch_list = np.empty(shape=(0,1))
    chosen = furthest_first(uld_embedding_by_batch_list, ld_embedding_by_batch_list, selected_num)

    tgt_selected_ds.add_item(tgt_unlabeled_ds.samples[chosen])
    tgt_unlabeled_ds.remove_item(chosen)

    return chosen

def TTest_active(model, tgt_unlabeled_ds, tgt_selected_ds, active_ratio, totality):
    config = model.config
    
    test_loader = data.DataLoader(
        tgt_unlabeled_ds,
        batch_size=config.testing.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers)
    
    model.dif_model.eval()
    model.cond_pred_model.eval()

    true_y_label_by_batch_list = []
    majority_vote_by_batch_list = []
    CI_by_batch_list = []
    ttest_pvalue_by_batch_list = []
    gen_y_2_class_prob_diff_list = []
    most_probable_classes_var_list = []
    label_mean_probs_by_batch_list = []

    selected_num = round(totality * active_ratio)

    with torch.no_grad():
        for step, feature_label_set in enumerate(test_loader):
            x_batch, y_labels_batch = feature_label_set
            x_batch = x_batch.to(model.device)
            # flattened features from the backbone network
            x_batch = model.backbone(x_batch)
            # compute y_0_hat as the initial prediction to guide the reverse diffusion process
            y_0_hat_batch_logit, z_batch  = model.compute_guiding_prediction(x_batch, True)
            y_0_hat_batch = y_0_hat_batch_logit.softmax(dim=1)
            true_y_label_by_batch_list.append(y_labels_batch.numpy())

            y_labels_batch = y_labels_batch.reshape(-1, 1)
            batch_size = z_batch.shape[0]
            # x_batch with shape (batch_size, flattened_image_dim)
            x_tile = (z_batch.repeat(config.testing.n_samples, 1, 1).transpose(0, 1)).to(model.device).flatten(0, 1)

            y_0_hat_tile = (y_0_hat_batch.repeat(config.testing.n_samples, 1, 1).transpose(0, 1)).flatten(0, 1)
            y_T_mean_tile = y_0_hat_tile
            # generate reconstructed p(y_0|x) for the current mini-batch
            device = model.device
            z = torch.randn_like(y_T_mean_tile).to(device)  # standard Gaussian
            cur_y = z + y_T_mean_tile  # sampled y_T
            num_t = 1
            for i in reversed(range(1, model.num_timesteps)):
                y_t = cur_y
                cur_y = p_sample(model.dif_model, x_tile, y_t, y_0_hat_tile, y_T_mean_tile, i, model.alphas, model.one_minus_alphas_bar_sqrt)  # y_{t-1}
                num_t += 1
            assert num_t == model.num_timesteps
            # obtain y_0 given y_1
            y_0 = p_sample_t_1to0(model.dif_model, x_tile, cur_y, y_0_hat_tile, y_T_mean_tile, model.one_minus_alphas_bar_sqrt)
            
            gen_y_all_class_raw_probs = y_0.reshape(batch_size, config.testing.n_samples, config.data.num_classes).cpu()

            # compute softmax probabilities of all classes for each sample
            raw_prob_val = compute_val_before_softmax(gen_y_all_class_raw_probs)
            if model.args.tune_T:
                gen_y_all_class_probs = torch.softmax(raw_prob_val / model.tuned_scale_T, dim=-1)  # (batch_size, n_samples, n_classes)
            else:
                gen_y_all_class_probs = torch.softmax(raw_prob_val, dim=-1)  # (batch_size, n_samples, n_classes)
            # obtain credible interval of probability predictions in each class for the samples given the same x
            low, high = config.testing.PICP_range
            # use raw predicted probability (right before temperature scaling and Softmax) width
            # to construct prediction interval
            CI_y_pred = raw_prob_val.nanquantile(q=torch.tensor([low / 100, high / 100]),
                                                 dim=1).swapaxes(0, 1)  # (batch_size, 2, n_classes)
            CI_y_pred_diff = CI_y_pred[:,1,:] - CI_y_pred[:,0,:] # (batch_size, n_classes)

            PIW_cls_mean = CI_y_pred_diff.mean(1) # (batch_size, )
                                    
            # obtain the predicted label with the largest probability for each sample
            gen_y_labels = torch.argmax(gen_y_all_class_probs, 2, keepdim=True)  # (batch_size, n_samples, 1)
            # convert the predicted label to one-hot format
            gen_y_one_hot = torch.zeros_like(gen_y_all_class_probs).scatter_(
                dim=2, index=gen_y_labels,
                src=torch.ones_like(gen_y_labels.float()))  # (batch_size, n_samples, n_classes)
            # compute proportion of each class as the prediction given the same x
            gen_y_label_probs = gen_y_one_hot.sum(1) / config.testing.n_samples  # (batch_size, n_classes)
            gen_y_all_class_mean_prob = gen_y_all_class_probs.mean(1)  # (batch_size, n_classes)
            # obtain the class being predicted the most given the same x
            gen_y_majority_vote = torch.argmax(gen_y_label_probs, 1, keepdim=True)  # (batch_size, 1)
            # compute the proportion of predictions being the correct label for each x
            gen_y_instance_accuracy = (gen_y_labels == y_labels_batch[:, None]).float().mean(1)  # (batch_size, 1)
            # conduct paired two-sided t-test for the two most predicted classes for each instance
            two_most_probable_classes_idx = gen_y_label_probs.argsort(dim=1, descending=True)[:, :2]
            two_most_probable_classes_idx = torch.repeat_interleave(
                two_most_probable_classes_idx[:, None],
                repeats=config.testing.n_samples, dim=1)  # (batch_size, n_samples, 2)
            gen_y_2_class_probs = torch.gather(gen_y_all_class_probs, dim=2,
                                               index=two_most_probable_classes_idx)  # (batch_size, n_samples, 2)            

            gen_y_2_class_prob_diff = abs(gen_y_2_class_probs[:, :, 1] \
                                        - gen_y_2_class_probs[:, :, 0]).mean(1)  # (batch_size, )

            ttest_pvalues = (ttest_rel(gen_y_2_class_probs[:, :, 0],
                                       gen_y_2_class_probs[:, :, 1],
                                       axis=1, alternative='two-sided')).pvalue  # (batch_size, )
            
            most_probable_classes_var = gen_y_2_class_probs[:, :, 0].var(dim=1) # (batch_size, )


            if len(majority_vote_by_batch_list) == 0:
                majority_vote_by_batch_list = gen_y_majority_vote # majority vote class for each sample
                CI_by_batch_list = PIW_cls_mean # class-wise mean of PIW for each sample
                ttest_pvalue_by_batch_list = ttest_pvalues # ttest_pvalue for each sample
                gen_y_2_class_prob_diff_list = gen_y_2_class_prob_diff # prediction difference of two most voted classes for each sample 
                most_probable_classes_var_list = most_probable_classes_var
                label_mean_probs_by_batch_list  = gen_y_all_class_mean_prob

            else:
                majority_vote_by_batch_list = np.concatenate([majority_vote_by_batch_list, gen_y_majority_vote], axis=0)
                CI_by_batch_list = np.concatenate([CI_by_batch_list, PIW_cls_mean], axis=0)
                ttest_pvalue_by_batch_list = np.concatenate([ttest_pvalue_by_batch_list, ttest_pvalues], axis=0)
                gen_y_2_class_prob_diff_list = np.concatenate([gen_y_2_class_prob_diff_list, gen_y_2_class_prob_diff], axis=0)
                most_probable_classes_var_list = np.concatenate([most_probable_classes_var_list, most_probable_classes_var], axis=0)
                label_mean_probs_by_batch_list = np.concatenate([label_mean_probs_by_batch_list,
                gen_y_all_class_mean_prob], axis=0)

    ttest_pvalues_index = ttest_pvalue_by_batch_list.argsort()[-1 * selected_num:-1]

    tgt_selected_ds.add_item(tgt_unlabeled_ds.samples[ttest_pvalues_index])
    tgt_unlabeled_ds.remove_item(ttest_pvalues_index)

    return ttest_pvalues_index

def variation_ttest_active(model, tgt_unlabeled_ds, tgt_selected_ds, active_ratio, totality):
    config = model.config
    
    test_loader = data.DataLoader(
        tgt_unlabeled_ds,
        batch_size=config.testing.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    model.dif_model.eval()
    model.cond_pred_model.eval()

    true_y_label_by_batch_list = []
    ttest_pvalue_by_batch_list = []
    most_probable_classes_var_list = []

    selected_num = round(totality * active_ratio)

    with torch.no_grad():
        for step, feature_label_set in enumerate(test_loader):
            x_batch, y_labels_batch = feature_label_set
            x_batch = x_batch.to(model.device)
            # flattened features from the backbone network
            x_batch = model.backbone(x_batch)
            # compute y_0_hat as the initial prediction to guide the reverse diffusion process
            y_0_hat_batch_logit, z_batch  = model.compute_guiding_prediction(x_batch, True)
            y_0_hat_batch = y_0_hat_batch_logit.softmax(dim=1)
            true_y_label_by_batch_list.append(y_labels_batch.numpy())

            y_labels_batch = y_labels_batch.reshape(-1, 1)
            batch_size = z_batch.shape[0]
            # x_batch with shape (batch_size, flattened_image_dim)
            x_tile = (z_batch.repeat(config.testing.n_samples, 1, 1).transpose(0, 1)).to(model.device).flatten(0, 1)

            y_0_hat_tile = (y_0_hat_batch.repeat(config.testing.n_samples, 1, 1).transpose(0, 1)).flatten(0, 1)
            y_T_mean_tile = y_0_hat_tile
            # generate reconstructed p(y_0|x) for the current mini-batch
            device = model.device
            z = torch.randn_like(y_T_mean_tile).to(device)  # standard Gaussian
            cur_y = z + y_T_mean_tile  # sampled y_T
            num_t = 1
            for i in reversed(range(1, model.num_timesteps)):
                y_t = cur_y
                cur_y = p_sample(model.dif_model, x_tile, y_t, y_0_hat_tile, y_T_mean_tile, i, model.alphas, model.one_minus_alphas_bar_sqrt)  # y_{t-1}
                num_t += 1
            assert num_t == model.num_timesteps
            # obtain y_0 given y_1
            y_0 = p_sample_t_1to0(model.dif_model, x_tile, cur_y, y_0_hat_tile, y_T_mean_tile, model.one_minus_alphas_bar_sqrt)
            
            gen_y_all_class_raw_probs = y_0.reshape(batch_size, config.testing.n_samples, config.data.num_classes).cpu()

            # compute softmax probabilities of all classes for each sample
            raw_prob_val = compute_val_before_softmax(gen_y_all_class_raw_probs)

            if model.args.tune_T:
                gen_y_all_class_probs = torch.softmax(raw_prob_val / model.tuned_scale_T, dim=-1)  # (batch_size, n_samples, n_classes)
            else:
                gen_y_all_class_probs = torch.softmax(raw_prob_val, dim=-1)  # (batch_size, n_samples, n_classes)
                                    
            # obtain the predicted label with the largest probability for each sample
            gen_y_labels = torch.argmax(gen_y_all_class_probs, 2, keepdim=True)  # (batch_size, n_samples, 1)
            # convert the predicted label to one-hot format
            gen_y_one_hot = torch.zeros_like(gen_y_all_class_probs).scatter_(
                dim=2, index=gen_y_labels,
                src=torch.ones_like(gen_y_labels.float()))  # (batch_size, n_samples, n_classes)
            # compute proportion of each class as the prediction given the same x
            gen_y_label_probs = gen_y_one_hot.sum(1) / config.testing.n_samples  # (batch_size, n_classes)

            # conduct paired two-sided t-test for the two most predicted classes for each instance
            two_most_probable_classes_idx = gen_y_label_probs.argsort(dim=1, descending=True)[:, :2]
            two_most_probable_classes_idx = torch.repeat_interleave(
                two_most_probable_classes_idx[:, None],
                repeats=config.testing.n_samples, dim=1)  # (batch_size, n_samples, 2)
            gen_y_2_class_probs = torch.gather(gen_y_all_class_probs, dim=2,
                                               index=two_most_probable_classes_idx)  # (batch_size, n_samples, 2)            

            ttest_pvalues = (ttest_ind(gen_y_2_class_probs[:, :, 0],
                                       gen_y_2_class_probs[:, :, 1],
                                       axis=1, alternative='two-sided')).pvalue  # (batch_size, )
            # print(gen_y_2_class_probs[:, :, 0].shape)
            most_probable_classes_var = gen_y_2_class_probs[:, :, 0].var(dim=1) # (batch_size, )

            if len(ttest_pvalue_by_batch_list) == 0:
                ttest_pvalue_by_batch_list = ttest_pvalues # ttest_pvalue for each sample
                most_probable_classes_var_list = most_probable_classes_var

            else:
                ttest_pvalue_by_batch_list = np.concatenate([ttest_pvalue_by_batch_list, ttest_pvalues], axis=0)
                most_probable_classes_var_list = np.concatenate([most_probable_classes_var_list, most_probable_classes_var], axis=0)

    most_probable_classes_var_index = most_probable_classes_var_list.argsort()[-1 * selected_num * 20: -1]
    
    candidate_set = ttest_pvalue_by_batch_list[most_probable_classes_var_index]

    selected_index_ = candidate_set.argsort()[-1 * selected_num : -1]

    selected_index = most_probable_classes_var_index[selected_index_]

    tgt_selected_ds.add_item(tgt_unlabeled_ds.samples[selected_index])
    tgt_unlabeled_ds.remove_item(selected_index)

    return selected_index

def PIW_ttest_active(model, tgt_unlabeled_ds, tgt_selected_ds, active_ratio, totality):
    args = model.args
    config = model.config
    
    test_loader = data.DataLoader(
        tgt_unlabeled_ds,
        batch_size=config.testing.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    model.dif_model.eval()
    model.cond_pred_model.eval()

    true_y_label_by_batch_list = []
    CI_by_batch_list = []
    ttest_pvalue_by_batch_list = []
    selected_num = round(totality * active_ratio)

    with torch.no_grad():
        for step, feature_label_set in enumerate(test_loader):
            x_batch, y_labels_batch = feature_label_set
            x_batch = x_batch.to(model.device)
            # flattened features from the backbone network
            x_batch = model.backbone(x_batch) 
            # compute y_0_hat as the initial prediction to guide the reverse diffusion process
            y_0_hat_batch_logit, z_batch  = model.compute_guiding_prediction(x_batch, True)
            y_0_hat_batch = y_0_hat_batch_logit.softmax(dim=1)
            true_y_label_by_batch_list.append(y_labels_batch.numpy())

            y_labels_batch = y_labels_batch.reshape(-1, 1)
            batch_size = z_batch.shape[0]
            # x_batch with shape (batch_size, flattened_image_dim)
            x_tile = (z_batch.repeat(config.testing.n_samples, 1, 1).transpose(0, 1)).to(model.device).flatten(0, 1)

            y_0_hat_tile = (y_0_hat_batch.repeat(config.testing.n_samples, 1, 1).transpose(0, 1)).flatten(0, 1)
            y_T_mean_tile = y_0_hat_tile
            # generate reconstructed p(y_0|x) for the current mini-batch
            device = model.device
            z = torch.randn_like(y_T_mean_tile).to(device)  # standard Gaussian
            cur_y = z + y_T_mean_tile  # sampled y_T
            num_t = 1
            for i in reversed(range(1, model.num_timesteps)):
                y_t = cur_y
                cur_y = p_sample(model.dif_model, x_tile, y_t, y_0_hat_tile, y_T_mean_tile, i, model.alphas, model.one_minus_alphas_bar_sqrt)  # y_{t-1}
                num_t += 1
            assert num_t == model.num_timesteps
            # obtain y_0 given y_1
            y_0 = p_sample_t_1to0(model.dif_model, x_tile, cur_y, y_0_hat_tile, y_T_mean_tile, model.one_minus_alphas_bar_sqrt)
            
            current_t = 0

            gen_y_all_class_raw_probs = y_0.reshape(batch_size, config.testing.n_samples, config.data.num_classes).cpu()

            # compute softmax probabilities of all classes for each sample
            raw_prob_val = compute_val_before_softmax(gen_y_all_class_raw_probs)

            if model.args.tune_T:
                gen_y_all_class_probs = torch.softmax(raw_prob_val / model.tuned_scale_T, dim=-1)  # (batch_size, n_samples, n_classes)
            else:
                gen_y_all_class_probs = torch.softmax(raw_prob_val, dim=-1)  # (batch_size, n_samples, n_classes)

            # obtain credible interval of probability predictions in each class for the samples given the same x
            low, high = config.testing.PICP_range
            # use raw predicted probability (right before temperature scaling and Softmax) width
            # to construct prediction interval
            CI_y_pred = raw_prob_val.nanquantile(q=torch.tensor([low / 100, high / 100]),
                                                 dim=1).swapaxes(0, 1)  # (batch_size, 2, n_classes)
            CI_y_pred_diff = CI_y_pred[:,1,:] - CI_y_pred[:,0,:] # (batch_size, n_classes)

            PIW_cls_mean = CI_y_pred_diff.mean(1) # (batch_size, )

            # obtain the predicted label with the largest probability for each sample
            gen_y_labels = torch.argmax(gen_y_all_class_probs, 2, keepdim=True)  # (batch_size, n_samples, 1)
            # convert the predicted label to one-hot format
            gen_y_one_hot = torch.zeros_like(gen_y_all_class_probs).scatter_(
                dim=2, index=gen_y_labels,
                src=torch.ones_like(gen_y_labels.float()))  # (batch_size, n_samples, n_classes)
            # compute proportion of each class as the prediction given the same x
            gen_y_label_probs = gen_y_one_hot.sum(1) / config.testing.n_samples  # (batch_size, n_classes)

            # obtain the class being predicted the most given the same x
            gen_y_majority_vote = torch.argmax(gen_y_label_probs, 1).unsqueeze(-1)  # (batch_size, 1)

            PIW_most = torch.gather(CI_y_pred_diff, dim=1, index=gen_y_majority_vote).squeeze(-1) # (batch_size, )

            # conduct paired two-sided t-test for the two most predicted classes for each instance
            two_most_probable_classes_idx = gen_y_label_probs.argsort(dim=1, descending=True)[:, :2]
            two_most_probable_classes_idx = torch.repeat_interleave(
                two_most_probable_classes_idx[:, None],
                repeats=config.testing.n_samples, dim=1)  # (batch_size, n_samples, 2)
            gen_y_2_class_probs = torch.gather(gen_y_all_class_probs, dim=2,
                                               index=two_most_probable_classes_idx)  # (batch_size, n_samples, 2)            

            ttest_pvalues = (ttest_ind(gen_y_2_class_probs[:, :, 0],
                                       gen_y_2_class_probs[:, :, 1],
                                       axis=1, alternative='two-sided')).pvalue  # (batch_size, )

            if len(ttest_pvalue_by_batch_list) == 0:
                CI_by_batch_list = PIW_most # class-wise mean of PIW for each sample
                ttest_pvalue_by_batch_list = ttest_pvalues # ttest_pvalue for each sample

            else:
                CI_by_batch_list = np.concatenate([CI_by_batch_list, PIW_cls_mean], axis=0)
                ttest_pvalue_by_batch_list = np.concatenate([ttest_pvalue_by_batch_list, ttest_pvalues], axis=0)

    CI_by_batch_list_index = CI_by_batch_list.argsort()[-1 * selected_num * 20: -1]
    
    candidate_set = ttest_pvalue_by_batch_list[CI_by_batch_list_index]

    selected_index_ = candidate_set.argsort()[-1 * selected_num : -1]

    selected_index = CI_by_batch_list_index[selected_index_]


    tgt_selected_ds.add_item(tgt_unlabeled_ds.samples[selected_index])
    tgt_unlabeled_ds.remove_item(selected_index)

    return selected_index

def compute_val_before_softmax(gen_y_0_val):
    """
    Compute raw value before applying Softmax function to obtain prediction in probability scale.
    Corresponding to the part inside the Softmax function of Eq. (10) in paper.
    """
    # TODO: add other ways of computing such raw prob value
    raw_prob_val = -(gen_y_0_val - 1) ** 2
    return raw_prob_val