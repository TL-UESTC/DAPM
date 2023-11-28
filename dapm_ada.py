import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from ema import EMA
from model import *
from utils import *
from diffusion_utils import *
from active import *
from sklearn.metrics import accuracy_score
import ema
plt.style.use('ggplot')
from math import log

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        self.num_timesteps = config.diffusion.timesteps

        betas = make_beta_schedule(schedule=config.diffusion.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=config.diffusion.beta_start, end=config.diffusion.beta_end)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if config.diffusion.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coeff_2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # initial prediction model as guided condition
        self.cond_pred_model = MyGuidedModel(config).to(self.device)
        self.cond_pred_model_teacher = MyGuidedModel(config).to(self.device)
        self.cond_pred_model_teacher.load_state_dict(self.cond_pred_model.state_dict(), strict=True)
        self.aux_cls_function = nn.CrossEntropyLoss()
        self.aux_kl_function = nn.KLDivLoss(reduction='mean')
        self.aux_batch_kl_function = nn.KLDivLoss(reduction='batchmean')

        # scaling temperature for NLL and ECE computation
        self.tuned_scale_T = None

    # Compute guiding prediction as diffusion condition
    def compute_guiding_prediction(self, x, hidden):
        """
        Compute y_0_hat, to be used as the Gaussian mean at time step T.
        """
        y_pred = self.cond_pred_model(x, hidden)
        return y_pred

    def evaluate_guidance_model(self, dataset_loader):
        """
        Evaluate guidance model by reporting train or test set accuracy.
        """
        correct_add = 0
        size = 0
        with torch.no_grad():
            for step, feature_label_set in enumerate(dataset_loader):
                # logging.info("\nEvaluating test Minibatch {}...\n".format(step))
                # minibatch_start = time.time()
                x_batch, y_labels_batch = feature_label_set
                # y_labels_batch = y_labels_batch.reshape(-1, 1)
                x_batch, y_labels_batch = x_batch.to(self.device), y_labels_batch.to(self.device)
                x_batch = self.backbone(x_batch)
                y_pred_prob = self.compute_guiding_prediction(
                    x_batch, False).softmax(dim=1)  # (batch_size, n_classes)
                y_pred_label = y_pred_prob.data.max(1)[1] # (batch_size, )
                
                correct_add += y_pred_label.eq(y_labels_batch.data).cpu().sum()
                size += y_pred_prob.data.size()[0]

        return float(correct_add / size)

    def evaluate_guidance_model_visda(self, dataset_loader):
        """
        Evaluate guidance model by reporting train or test set accuracy on the visda dataset.
        """
        # preparation
        dset_classes = ['aeroplane',
                'bicycle',
                'bus',
                'car',
                'horse',
                'knife',
                'motorcycle',
                'person',
                'plant',
                'skateboard',
                'train',
                'truck']
        classes_acc = {}
        for i in dset_classes:
            classes_acc[i] = []
            classes_acc[i].append(0)
            classes_acc[i].append(0)

        correct_add = 0
        size = 0
        y_acc_list = []
        with torch.no_grad():
            for step, feature_label_set in enumerate(dataset_loader):
                # logging.info("\nEvaluating test Minibatch {}...\n".format(step))
                # minibatch_start = time.time()
                x_batch, y_labels_batch = feature_label_set
                # y_labels_batch = y_labels_batch.reshape(-1, 1)
                x_batch, y_labels_batch = x_batch.to(self.device), y_labels_batch.to(self.device)
                x_batch = self.backbone(x_batch)

                y_pred_prob = self.compute_guiding_prediction(
                    x_batch, False).softmax(dim=1)  # (batch_size, n_classes)
                y_pred_label = y_pred_prob.data.max(1)[1] # (batch_size, )
                
                correct_add += y_pred_label.eq(y_labels_batch.data).cpu().sum()
                size += y_pred_prob.data.size()[0]

                for i in range(y_pred_prob.data.size()[0]):
                    key_label = dset_classes[y_labels_batch.long()[i].item()]
                    key_pred = dset_classes[y_pred_label.long()[i].item()]
                    classes_acc[key_label][1] += 1
                    if key_pred == key_label:
                        classes_acc[key_pred][0] += 1  
            for i in dset_classes:
                y_acc_list.append(float(classes_acc[i][0]) / classes_acc[i][1])

        mean_acc = sum(y_acc_list)/len(y_acc_list)
        return mean_acc

    def source_model_train_one_step(self, input_s, label_s, aux_optimizer):
        """
        Pre-train a soruce model for the initialization of target model
        """
        mu_s, sigma_s = self.cond_pred_model.encoder(input_s)

        z_1, z_1_standard  = reparameterize2(mu_s, sigma_s)
        z_s = z_1.rsample()
        out_s = self.cond_pred_model.classifier(z_s)
        # classification loss
        cls_loss = self.aux_cls_function(out_s, label_s)
        # reconstruction loss 
        img_s2s = self.cond_pred_model.s_decoder(z_s)
        L1loss = nn.L1Loss()
        recon_loss_s2s = L1loss(img_s2s, input_s)
        recon_loss = recon_loss_s2s
     
        loss_KL = torch.distributions.kl.kl_divergence(z_1, z_1_standard).mean()

        loss_all = cls_loss + recon_loss + loss_KL * 0.1

        # update non-linear guidance model
        aux_optimizer.zero_grad()
        loss_all.backward()
        aux_optimizer.step()
        return loss_all.cpu().item()


    def train_guidance_model_one_step(self, input_s, input_t, label_s, input_ts, label_ts, aux_optimizer, teacher_optimizer, total_step, labels_target, max_iter):
        """
        One optimization step of the guidance model that predicts y_0_hat.
        """
        lr_scheduler(aux_optimizer, iter_num=total_step, max_iter=max_iter)

        mu_s, sigma_s = self.cond_pred_model.encoder(input_s)
        z_1, z_1_standard  = reparameterize2(mu_s, sigma_s)
        mu_t, sigma_t = self.cond_pred_model.encoder(input_t)
        z_2, z_2_standard = reparameterize2(mu_t, sigma_t)
        with torch.no_grad():
            out_t_tea = self.cond_pred_model_teacher(input_t)
        
        z_s = z_1.rsample()
        z_t = z_2.rsample()
        
        # classification loss
        out_s = self.cond_pred_model.classifier(z_s)
        out_t = self.cond_pred_model.classifier(z_t)

        z_all = torch.cat((z_s, z_t), dim=0)
        outputs_all = torch.cat((out_s, out_t), dim=0)

        softmax_t = F.softmax(out_t, dim=1)
        softmax_all = F.softmax(outputs_all, dim=1)

        # classification loss
        cls_loss = self.aux_cls_function(out_s, label_s)
        
        # add KL loss for class diversity and prediction confident
        msoftmax = softmax_t.mean(dim=0)
        kl_term_1 = torch.sum(msoftmax * torch.log(msoftmax + 1e-5)) + log(msoftmax.size(0))
        kl_term_2 = torch.mean(torch.sum(softmax_t * torch.log(softmax_t + 1e-5), dim=1)) + log(softmax_t.size(1))
        loss_kl = kl_term_1 - kl_term_2
        cls_loss += self.config.training.weight_kl * loss_kl
        # add unlabeled target samples distillation loss
        unsup_loss,_,_ = compute_distill_loss(out_t, out_t_tea, labels_target)
        
        cls_loss += self.config.training.weight_kd * unsup_loss
        
        # reconstruction loss 
        img_s2s = self.cond_pred_model.s_decoder(z_s)
        img_t2t = self.cond_pred_model.t_decoder(z_t)
        L1loss = nn.L1Loss()
        recon_loss_s2s = L1loss(img_s2s, input_s)
        recon_loss_t2t = L1loss(img_t2t, input_t)
        recon_loss = recon_loss_s2s + recon_loss_t2t

        # adversarial loss
        entropy_all = Entropy(softmax_all)
        transfer_loss = Conditional_Adversarial_Loss([z_all, softmax_all], self.ad_net, entropy_all, calc_coeff(total_step), None)        

        loss_KL = torch.distributions.kl.kl_divergence(z_1, z_1_standard).mean() + torch.distributions.kl.kl_divergence(z_2, z_2_standard).mean()

        loss_all = cls_loss + recon_loss + transfer_loss * self.config.training.weight_transfer + loss_KL * 0.1

        if (input_ts is not None) and (label_ts is not None):
            n_ts = input_ts.size(0)
            if n_ts == 1:
                # avoid bs=1, can't pass through BN layer
                input_ts = torch.cat((input_ts, input_ts), dim=0)
                label_ts = torch.cat((label_ts, label_ts), dim=0)

            mu_ts, sigma_ts = self.cond_pred_model.encoder(input_ts)

            with torch.no_grad():
               _ = self.cond_pred_model_teacher(input_ts)          

            z_3, z_3_standard = reparameterize2(mu_ts, sigma_ts)
            z_ts = z_3.rsample()

            z_4, _ = reparameterize2(mu_t[:input_ts.size(0),:], sigma_t[:input_ts.size(0),:])
            z_t_align = z_4.rsample()

            out_ts = self.cond_pred_model.classifier(z_ts)
            out_s_align = self.cond_pred_model.classifier(z_t_align)

            cls_loss_ts = self.aux_cls_function(out_ts, label_ts)
            img_ts2ts = self.cond_pred_model.t_decoder(z_ts)
            recon_loss_ts2ts = L1loss(img_ts2ts, input_ts)

            loss_KL_ts = torch.distributions.kl.kl_divergence(z_3, z_3_standard).mean()

            ## adversarial 
            z_all = torch.cat((z_ts, z_t_align), dim=0)
            outputs_all = torch.cat((out_s_align, out_ts), dim=0)
            softmax_all = F.softmax(outputs_all, dim=1)
            entropy_all = Entropy(softmax_all)
            transfer_loss_ts = Conditional_Adversarial_Loss([z_all, softmax_all], self.ad_net, entropy_all, calc_coeff(total_step), None)        
            
            loss_all += (cls_loss_ts + recon_loss_ts2ts + transfer_loss_ts * self.config.training.weight_transfer + loss_KL_ts * 0.1)

        # update non-linear guidance model
        aux_optimizer.zero_grad()
        loss_all.backward()
        aux_optimizer.step()
        teacher_optimizer.step()
        return loss_all.cpu().item()

    def train(self):
        args = self.args
        config = self.config
        src_train_dataset, _, tgt_train_dataset, test_dataset, tgt_selected_dataset = get_dataset(args, config)
        # source DL
        src_train_loader = data.DataLoader(
            src_train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.data.num_workers,
        )
        # unlabeled target DL
        unlabeled_tgt_train_loader = data.DataLoader(
            tgt_train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.data.num_workers,
        )
        # active target DL
        labeled_tgt_train_loader = data.DataLoader(
            tgt_selected_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=config.data.num_workers,
        )        
        # test DL        
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.testing.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.data.num_workers,
        )
        # total number of target samples
        totality = tgt_train_dataset.__len__()
        all_selected_images = None
        ## train        
        backbone = ResNetBackbone("resnet50")
        dif_model = ConditionalModel(config)  ### used for diffusion
        self.ad_net = AdversarialNetwork(config.diffusion.aux_cls.z_dim * config.data.num_classes, 1024) ### used for adversarial

        criterion = nn.CrossEntropyLoss()

        self.backbone = backbone
        self.dif_model = dif_model
        
        backbone.to(self.device)
        dif_model = dif_model.to(self.device)
        self.ad_net = self.ad_net.to(self.device)

        optimizer = get_optimizer(self.args, self.config.optim, dif_model.parameters())
        if self.config.training.open_backbone:
            aux_optimizer = get_optimizer(self.args, self.config.aux_optim, self.backbone.parameters_list(self.config.aux_optim.lr) + self.cond_pred_model.parameters_list(self.config.aux_optim.lr) + self.ad_net.parameters_list(self.config.aux_optim.lr)) 
        else:
            aux_optimizer = get_optimizer(self.args, self.config.aux_optim, self.cond_pred_model.parameters_list(self.config.aux_optim.lr) + self.ad_net.parameters_list(self.config.aux_optim.lr))
            for param in list(self.backbone.parameters()):
                param.requires_grad = False            
            backbone = backbone.eval()  
        aux_optimizer = op_copy(aux_optimizer)

        for param in list(self.cond_pred_model_teacher.parameters()):
            param.requires_grad = False

        teacher_optimizer = ema.MyWeightEMA(self.cond_pred_model_teacher, self.cond_pred_model, alpha=0.99)

        if self.config.model.ema:
            ema_helper = EMA(mu=self.config.model.ema_rate)
            ema_helper.register(dif_model)
        else:  
            ema_helper = None
        
        ##########################  Warm up phase  ############################################
        logging.info("\nPre-training of source model...")
        if self.config.training.open_backbone:
            self.backbone.train()
        self.cond_pred_model.train()
        pretrain_start_time = time.time()
        for epoch in range(config.diffusion.aux_cls.source_pretrain_epochs):
            len_train_source = len(src_train_loader)
            iter_per_epoch = len_train_source  
            for i in range(iter_per_epoch):
                if i % len_train_source == 0:
                    iter_source = iter(src_train_loader)
                inputs_source, labels_source = next(iter_source)
                inputs_source, labels_source = inputs_source.to(self.device), labels_source.to(self.device)
                # get ResNet processed feature as x
                inputs_source = backbone(inputs_source)     
                aux_loss = self.source_model_train_one_step(inputs_source.to(self.device), labels_source.to(self.device), aux_optimizer)
            if epoch % config.diffusion.aux_cls.logging_interval == 0:
                logging.info(
                    f"epoch: {epoch}, source model training loss: {aux_loss}"
                )
        pretrain_end_time = time.time()
        if self.config.training.open_backbone:
            self.backbone.eval()
        self.cond_pred_model.eval()
        logging.info("\nPre-training of source model took {:.4f} minutes.\n".format(
            (pretrain_end_time - pretrain_start_time) / 60))
        # save auxiliary model after pre-training
        aux_states = [
            self.cond_pred_model.state_dict(),
            aux_optimizer.state_dict(),
        ]
        torch.save(aux_states, os.path.join(self.args.log_path, "source_ckpt.pth"))
        # sync source and teacher model
        self.cond_pred_model_teacher.load_state_dict(self.cond_pred_model.state_dict(), strict=True)
        self.cond_pred_model_teacher.train()
        total_step = 0

        ########################## start training ################################
        for episode in range(20):
            logging.info("The {} active round on task {} to {}.".format(episode, config.data.source_domain, config.data.target_domain))
            if self.config.training.open_backbone:
                self.backbone.train()
            self.cond_pred_model.train()
            self.ad_net.train()
            pretrain_start_time = time.time()
            warm_up_epoch = 1 if config.data.dataset == 'visda' else config.training.warmup_epochs_adaptation
            for epoch in range(warm_up_epoch if episode == 0 else config.diffusion.aux_cls.n_pretrain_epochs):
                len_train_source = len(src_train_loader)
                len_train_target = len(unlabeled_tgt_train_loader)
                iter_per_epoch = max(len_train_source,len_train_target)
                max_iter = 10000 
                for i in range(iter_per_epoch):
                    if i % len_train_source == 0:
                        iter_source = iter(src_train_loader)
                    if i % len_train_target == 0:
                        iter_target = iter(unlabeled_tgt_train_loader)
                    if not tgt_selected_dataset.empty:
                        if i % len(labeled_tgt_train_loader) == 0:
                            iter_tgt_selected = iter(labeled_tgt_train_loader)
                    inputs_source, labels_source = next(iter_source)
                    inputs_target, labels_target = next(iter_target)
                    inputs_source, labels_source, inputs_target = inputs_source.to(self.device), labels_source.to(self.device), inputs_target.to(self.device)          
                    if not tgt_selected_dataset.empty:
                        inputs_target_sel, labels_target_sel = next(iter_tgt_selected)
                        inputs_target_sel, labels_target_sel = inputs_target_sel.to(self.device), labels_target_sel.to(self.device)
                    # get ResNet processed feature as x
                    inputs_source = backbone(inputs_source)
                    inputs_target = backbone(inputs_target)
                    if not tgt_selected_dataset.empty:
                        total_step +=1 
                        inputs_target_sel = backbone(inputs_target_sel)
                        aux_loss = self.train_guidance_model_one_step(inputs_source.to(self.device), inputs_target.to(self.device), labels_source.to(self.device), inputs_target_sel.to(self.device), labels_target_sel.to(self.device), aux_optimizer, teacher_optimizer, total_step, labels_target, max_iter)
                    else:      
                        total_step +=1      
                        aux_loss = self.train_guidance_model_one_step(inputs_source.to(self.device), inputs_target.to(self.device), labels_source.to(self.device), None, None, aux_optimizer, teacher_optimizer, total_step, labels_target, max_iter)
                if epoch % config.diffusion.aux_cls.logging_interval == 0:
                    logging.info(
                        f"epoch: {epoch}, guidance auxiliary classifier pre-training loss: {aux_loss}"
                    )
            pretrain_end_time = time.time()
            if self.config.training.open_backbone:
                self.backbone.eval()
            self.cond_pred_model.eval()
            logging.info("\nPre-training of guidance auxiliary classifier took {:.4f} minutes.\n".format(
                (pretrain_end_time - pretrain_start_time) / 60))
            # save auxiliary model after pre-training
            aux_states = [
                self.cond_pred_model.state_dict(),
                aux_optimizer.state_dict(),
            ]
            torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))

            if config.data.dataset == "visda":
                y_acc_aux_model = self.evaluate_guidance_model_visda(test_loader)
            else:
                y_acc_aux_model = self.evaluate_guidance_model(test_loader)
            logging.info("\nAfter pre-training, guidance classifier accuracy on the test set is {:.8f}.\n".format(
                y_acc_aux_model))

            max_accuracy = 0.0
            ## train the diffusion model
            warm_up_epoch = 1 if config.data.dataset == 'visda' else config.training.warmup_epochs_diffusion
            for epoch in range(warm_up_epoch if episode ==0 else self.config.training.n_epochs):
                if episode >= 5:
                    break
                step = 0
                len_train_source = len(src_train_loader)
                len_train_lbd_target = len(labeled_tgt_train_loader)
                iter_per_epoch = max(len_train_source, len_train_lbd_target)  
                dif_model.train()
                for i in range(iter_per_epoch):
                    if i % len_train_source == 0:
                        iter_source = iter(src_train_loader)
                    if not tgt_selected_dataset.empty:
                        if i % len_train_lbd_target == 0:
                            iter_tgt_selected = iter(labeled_tgt_train_loader)
                    inputs_source, labels_source = next(iter_source)
                    if not tgt_selected_dataset.empty:
                        inputs_target_sel, labels_target_sel = next(iter_tgt_selected)
                        x_batch = torch.cat((inputs_source, inputs_target_sel), dim=0).to(self.device)
                        y_batch = torch.cat((labels_source, labels_target_sel), dim=0).to(self.device)
                    else:
                        x_batch, y_batch = inputs_source.to(self.device), labels_source.to(self.device)
                    
                    # get ResNet processed feature as x
                    with torch.no_grad():
                        x_batch = backbone(x_batch)
                    step += 1
                    n = x_batch.size(0)
                    # antithetic sampling
                    t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                    t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)[:n]

                    # noise estimation loss
                    # y_0_batch = y_logits_batch.to(self.device)
                    y_one_hot_batch = cast_label_to_one_hot(y_batch, config)

                    y_0_hat_batch, z_batch = self.compute_guiding_prediction(x_batch,True)
                    y_0_hat_batch = y_0_hat_batch.softmax(dim=1)
                    z_batch = z_batch.detach()

                    y_T_mean = y_0_hat_batch

                    y_0_batch = y_one_hot_batch.to(self.device)
                    e = torch.randn_like(y_0_batch).to(y_0_batch.device)
                    y_t_batch = q_sample(y_0_batch, y_T_mean, # calculate y_t in the forward process
                                            self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=e)
                    # output = model(x_batch, y_t_batch, t, y_T_mean)
                    output = dif_model(z_batch, y_t_batch, t, y_0_hat_batch)
                    loss = (e - output).square().mean()  # use the same noise sample e during training to compute loss

                    # cross-entropy for y_0 reparameterization
                    loss0 = torch.tensor([0])
                    if args.add_ce_loss:
                        y_0_reparam_batch = y_0_reparam(dif_model, x_batch, y_t_batch, y_0_hat_batch, y_T_mean, t,
                                                        self.one_minus_alphas_bar_sqrt)
                        raw_prob_batch = -(y_0_reparam_batch - 1) ** 2
                        loss0 = criterion(raw_prob_batch, y_batch.to(self.device))
                        loss += config.training.lambda_ce * loss0

                    # optimize diffusion model that predicts eps_theta
                    optimizer.zero_grad()
                    loss.backward()
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            dif_model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    optimizer.step()
                    if self.config.model.ema:
                        ema_helper.update(dif_model)

                    # save diffusion model
                    if step % self.config.training.snapshot_freq == 0 or step == 1:
                        states = [
                            dif_model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        if self.config.model.ema:
                            states.append(ema_helper.state_dict())

                        if step > 1:  # skip saving the initial ckpt
                            torch.save(
                                states,
                                os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                            )
                        # save current states
                        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                        # save auxiliary model
                        aux_states = [
                            self.cond_pred_model.state_dict(),
                            aux_optimizer.state_dict(),
                        ]
                        if step > 1:  # skip saving the initial ckpt
                            torch.save(
                                aux_states,
                                os.path.join(self.args.log_path, "aux_ckpt_{}.pth".format(step)),
                            )
                        torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))

                # Evaluate
                if epoch % self.config.training.validation_freq == 0:
                        dif_model.eval()
                        acc_avg = 0.
                        target_all = None
                        predict_all = None
                        for test_batch_idx, (images, target) in enumerate(test_loader):
                            images = images.to(self.device)
                            target = target.to(self.device)
                            if target_all is None:
                                target_all = target.cpu()
                            else:
                                target_all = torch.cat([target_all, target.cpu()], dim = 0)
                            with torch.no_grad():
                                images = backbone(images)                         
                                # compute y_0_hat as the initial prediction to guide the reverse diffusion process
                                y_0_hat_batch_logit, z_batch  = self.compute_guiding_prediction(images, True)
                                y_0_hat_batch = y_0_hat_batch_logit.softmax(dim=1)

                                batch_size = z_batch.shape[0]
                                # x_batch with shape (batch_size, flattened_image_dim)
                                x_tile = (z_batch.repeat(config.testing.n_samples, 1, 1).transpose(0, 1)).to(self.device).flatten(0, 1)

                                y_0_hat_tile = (y_0_hat_batch.repeat(config.testing.n_samples, 1, 1).transpose(0, 1)).flatten(0, 1)
                                y_T_mean_tile = y_0_hat_tile
                                # generate reconstructed p(y_0|x) for the current mini-batch
                                z = torch.randn_like(y_T_mean_tile).to(self.device)  # standard Gaussian
                                cur_y = z + y_T_mean_tile  # sampled y_T
                                num_t = 1
                                for i in reversed(range(1, self.num_timesteps)):
                                    y_t = cur_y
                                    cur_y = p_sample(self.dif_model, x_tile, y_t, y_0_hat_tile, y_T_mean_tile, i, self.alphas, self.one_minus_alphas_bar_sqrt)  # y_{t-1}
                                    num_t += 1
                                assert num_t == self.num_timesteps
                                # obtain y_0 given y_1
                                y_0 = p_sample_t_1to0(self.dif_model, x_tile, cur_y, y_0_hat_tile, y_T_mean_tile, self.one_minus_alphas_bar_sqrt)
                                
                                gen_y_all_class_raw_probs = y_0.reshape(batch_size, config.testing.n_samples, config.data.num_classes).cpu()

                                # compute softmax probabilities of all classes for each sample
                                raw_prob_val = compute_val_before_softmax(gen_y_all_class_raw_probs)

                                gen_y_all_class_probs = torch.softmax(raw_prob_val, dim=-1)  # (batch_size, n_samples, n_classes)

                                # obtain the predicted label with the largest probability for each sample
                                gen_y_labels = torch.argmax(gen_y_all_class_probs, 2, keepdim=True)  # (batch_size, n_samples, 1)
                                # convert the predicted label to one-hot format
                                gen_y_one_hot = torch.zeros_like(gen_y_all_class_probs).scatter_(
                                    dim=2, index=gen_y_labels,
                                    src=torch.ones_like(gen_y_labels.float()))  # (batch_size, n_samples, n_classes)
                                # compute proportion of each class as the prediction given the same x
                                gen_y_label_probs = gen_y_one_hot.sum(1) / config.testing.n_samples  # (batch_size, n_classes)
                                # obtain the class being predicted the most given the same x
                                gen_y_majority_vote = torch.argmax(gen_y_label_probs, 1)  # (batch_size, )

                                if predict_all is None:
                                    predict_all = gen_y_majority_vote.cpu()
                                else:
                                    predict_all = torch.cat([predict_all, gen_y_majority_vote.cpu()], dim = 0)      
                                                              
                        if config.data.dataset == "visda":
                            dset_classes = ['aeroplane',
                                    'bicycle',
                                    'bus',
                                    'car',
                                    'horse',
                                    'knife',
                                    'motorcycle',
                                    'person',
                                    'plant',
                                    'skateboard',
                                    'train',
                                    'truck']

                            classes_acc = {}
                            for i in dset_classes:
                                classes_acc[i] = []
                                classes_acc[i].append(0)
                                classes_acc[i].append(0)

                            correct_add = 0
                            size = 0
                            y_acc_list = []

                            for i in range(predict_all.data.size()[0]):
                                key_label = dset_classes[target_all.long()[i].item()]
                                key_pred = dset_classes[predict_all.long()[i].item()]
                                classes_acc[key_label][1] += 1
                                if key_pred == key_label:
                                    classes_acc[key_pred][0] += 1  
                            for i in dset_classes:
                                y_acc_list.append(float(classes_acc[i][0]) / classes_acc[i][1])

                            acc_avg = sum(y_acc_list)/len(y_acc_list) * 100
                        else:   
                            acc_avg = accuracy_score(target_all.numpy(), predict_all.numpy()) * 100

                        if acc_avg > max_accuracy:
                            logging.info("Update best accuracy at Epoch {}.".format(epoch))
                            torch.save(states, os.path.join(self.args.log_path, "ckpt_best.pth"))
                        max_accuracy = max(max_accuracy, acc_avg)
                        logging.info(
                            (
                                    f"epoch: {epoch}, step: {step}, " +
                                    f"Accuracy: {acc_avg}, " +
                                    f"Max accuracy: {max_accuracy:.2f}%"
                            )
                        )

            # save the model after training is finished
            states = [
                dif_model.state_dict(),
                optimizer.state_dict(),
                epoch
            ]
            if self.config.model.ema:
                states.append(ema_helper.state_dict())
            torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
            
            # active selection rounds
            if episode < 5:
                logging.info('Task: {} to {}. Active Selction for Round: {} ......'.format(config.data.source_domain, config.data.target_domain, episode))
                active_samples = get_active_func(config.training.active)(model=self,
                                                tgt_unlabeled_ds=tgt_train_dataset,
                                                tgt_selected_ds=tgt_selected_dataset,
                                                active_ratio=0.01,
                                                totality=totality)
       
                # record all selected target images
                if all_selected_images is None:
                    all_selected_images = active_samples
                else:
                    all_selected_images = np.concatenate((all_selected_images, active_samples), axis=0)

