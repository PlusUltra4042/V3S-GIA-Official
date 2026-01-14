"""Mechanisms for image reconstruction from parameter gradients."""

import torch
from collections import defaultdict, OrderedDict
from inversefed.nn import MetaMonkey
from .metrics import total_variation as TV
from .metrics import InceptionScore
from .medianfilt import MedianPool2d
from copy import deepcopy
from .label_recover import label_inference
# from .Diffusion_Model_0708 import Diffusion
from diffusers import AutoencoderKL
# from imagenet_autoencoder.models.vgg import VGGAutoEncoder, get_configs
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import os
import numpy as npVAEVAE
from datetime import datetime
from torchvision.utils import make_grid
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torchvision
import torch.nn.functional as F
import math
from torch.distributions.laplace import Laplace
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
import torchvision.transforms as transforms


DEFAULT_CONFIG = dict(signed=False,
                      boxed=True,
                      stage1_cost_fn='sim',
                      stage1_indices='def',
                      stage1_weights='equal',
                      stage1_lr=0.1,
                      stage1_optim='adam',
                      restarts=1,
                      stage1_latent_iter=10,
                      stage1_tv_scale=1e-1,
                      stage1_z_r_scale=0.001,
                      stage1_bn_scale = 0.1,
                      stage1_l2_scale = 1e-6,
                      stage1_z_kl_scale = 1,
                      stage1_grad_scale = 1,
                      ######################
                      stage2_cost_fn='sim',
                      stage2_indices='def',#top10
                      stage2_weights='equal',
                      stage2_lr=0.1,
                      stage2_optim='adam',
                      stage2_iterations = 2000,
                      stage2_tv_scale=1e-1,
                      stage2_bn_scale = 0.1,
                      stage2_group_regularization = 0.01,
                      stage2_l2_scale= 1e-6,
                      stage2_contrast_scale = 1e-3,
                      stage2_grad_scale = 1,
                      stage2_alpha_noise = 0.005,
                      ###########################
                      stage3_cost_fn='sim',
                      stage3_indices='def',#top10
                      stage3_weights='equal',
                      stage3_lr = 1e-6,
                      stage3_optim='adam',
                      stage3_train_epochs=2000,
                      stage3_tv_scale = 1e-1,
                      stage3_grad_feature = 1,
                      init='randn',
                      filter='none',
                      lr_decay=True,
                      scoring_choice='loss')

def _label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def _validate_config(config):
    for key in DEFAULT_CONFIG.keys():
        if config.get(key) is None:
            config[key] = DEFAULT_CONFIG[key]
    for key in config.keys():
        if DEFAULT_CONFIG.get(key) is None:
            raise ValueError(f'Deprecated key in config dict: {key}!')
    return config

class GradientReconstructor():
    """Instantiate a reconstruction algorithm."""
    
    def __init__(self, model, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1):
        """Initialize with algorithm setup."""
        self.config = _validate_config(config)
        self.model = model
        self.setup = dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)

        self.mean_std = mean_std
        self.num_images = num_images

        if self.config['scoring_choice'] == 'inception':
            self.inception = InceptionScore(batch_size=1, setup=self.setup)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.iDLG = True

    def reconstruct(self, input_data, labels, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None):
        """Reconstruct image from gradient."""
        print("My Reconstructor!")
        start_time = time.time()
        if eval:
            self.model.eval()


        stats = defaultdict(list)
        x = self._init_images(img_shape)
        scores = torch.zeros(self.config['restarts'])

        if labels is None:
            if self.num_images == 1 and self.iDLG:
                # iDLG trick:
                last_weight_min = torch.argmin(torch.sum(input_data[-2], dim=-1), dim=-1)
                labels = last_weight_min.detach().reshape((1,)).requires_grad_(False)
                self.reconstruct_label = False
            else:
                # DLG label recovery
                # However this also improves conditioning for some LBFGS cases
                self.reconstruct_label = True

                def loss_fn(pred, labels):
                    labels = torch.nn.functional.softmax(labels, dim=-1)
                    return torch.mean(torch.sum(- labels * torch.nn.functional.log_softmax(pred, dim=-1), 1))
                self.loss_fn = loss_fn
            # fc_grad = input_data[-1].view(-1)
            # reconstructed_labels = label_inference(fc_grad, num_images, num_classes=100)
            # print("Reconstructed labels:", reconstructed_labels)
        else:
            assert labels.shape[0] == self.num_images
            self.reconstruct_label = False
            print("labels True")

        try:
            for trial in range(self.config['restarts']):
                x_trial, labels = self._run_trial(x[trial], input_data, labels, dryrun=dryrun)
                # Finalize
                scores[trial] = self._score_trial(x_trial, input_data, labels)
                x[trial] = x_trial
                if tol is not None and scores[trial] <= tol:
                    break
                if dryrun:
                    break
        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        # Choose optimal result:
        if self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            x_optimal, stats = self._average_trials(x, labels, input_data, stats)
        else:
            print('Choosing optimal result ...')
            scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
            optimal_index = torch.argmin(scores)
            print(f'Optimal result score: {scores[optimal_index]:2.4f}')
            stats['opt'] = scores[optimal_index].item()
            x_optimal = x[optimal_index]

        print(f'Total time: {time.time()-start_time}.')
        return x_optimal.detach(), stats

    def _init_images(self, img_shape):
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        else:
            raise ValueError()

    def _run_trial(self, x_trial, input_data, labels, dryrun=False):
        x_trial.requires_grad = True
        if self.reconstruct_label:
            output_test = self.model(x_trial)
            labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)

            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial, labels], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial, labels], lr=0.01, momentum=0.9, nesterov=True)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial, labels])
            else:
                raise ValueError()
        else:
            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial], lr=0.01, momentum=0.9, nesterov=True)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial])
            else:
                raise ValueError()

        max_iterations = self.config['max_iterations']
        dm, ds = self.mean_std
        if self.config['lr_decay']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[max_iterations // 2.667, max_iterations // 1.6,

                                                                         max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
        try:
            for iteration in range(max_iterations):
                closure = self._gradient_closure(optimizer, x_trial, input_data, labels)
                rec_loss = optimizer.step(closure)
                if self.config['lr_decay']:
                    scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.config['boxed']:
                        x_trial.data = torch.max(torch.min(x_trial, (1 - dm) / ds), -dm / ds)

                    if (iteration + 1 == max_iterations) or iteration % 500 == 0:
                        print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')

                    if (iteration + 1) % 500 == 0:
                        if self.config['filter'] == 'none':
                            pass
                        elif self.config['filter'] == 'median':
                            x_trial.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(x_trial)
                        else:
                            raise ValueError()

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        return x_trial.detach(), labels

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label):

        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            rec_loss = reconstruction_costs([gradient], input_gradient,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])

            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * TV(x_trial)
            rec_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            x_trial.grad = None
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            return reconstruction_costs([gradient], input_gradient,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return TV(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)
        elif self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            return 0.0
        else:
            raise ValueError()

    def _average_trials(self, x, labels, input_data, stats):
        print(f'Computing a combined result via {self.config["scoring_choice"]} ...')
        if self.config['scoring_choice'] == 'pixelmedian':
            x_optimal, _ = x.median(dim=0, keepdims=False)
        elif self.config['scoring_choice'] == 'pixelmean':
            x_optimal = x.mean(dim=0, keepdims=False)

        self.model.zero_grad()
        if self.reconstruct_label:
            labels = self.model(x_optimal).softmax(dim=1)
        loss = self.loss_fn(self.model(x_optimal), labels)
        gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
        stats['opt'] = reconstruction_costs([gradient], input_data,
                                            cost_fn=self.config['cost_fn'],
                                            indices=self.config['indices'],
                                            weights=self.config['weights'])
        print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats



class FedAvgReconstructor(GradientReconstructor):
    """Reconstruct an image from weights after n gradient descent steps."""

    def __init__(self, model, mean_std=(0.0, 1.0), local_steps=2, local_lr=1e-4,
                 config=DEFAULT_CONFIG, num_images=1, use_updates=True, batch_size=0):
        """Initialize with model, (mean, std) and config."""
        super().__init__(model, mean_std, config, num_images)
        self.local_steps = local_steps
        self.local_lr = local_lr
        self.use_updates = use_updates
        self.batch_size = batch_size

    def _gradient_closure(self, optimizer, x_trial, input_parameters, labels):
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            parameters = loss_steps(self.model, x_trial, labels, loss_fn=self.loss_fn,
                                    local_steps=self.local_steps, lr=self.local_lr,
                                    use_updates=self.use_updates,
                                    batch_size=self.batch_size)
            rec_loss = reconstruction_costs([parameters], input_parameters,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])

            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * TV(x_trial)
            rec_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, input_parameters, labels):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            parameters = loss_steps(self.model, x_trial, labels, loss_fn=self.loss_fn,
                                    local_steps=self.local_steps, lr=self.local_lr, use_updates=self.use_updates)
            return reconstruction_costs([parameters], input_parameters,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return TV(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)

def loss_steps(model, inputs, labels, loss_fn=torch.nn.CrossEntropyLoss(), lr=1e-4, local_steps=4, use_updates=True, batch_size=0):
    """Take a few gradient descent steps to fit the model to the given input."""
    patched_model = MetaMonkey(model)
    if use_updates:
        patched_model_origin = deepcopy(patched_model)
    for i in range(local_steps):
        if batch_size == 0:
            outputs = patched_model(inputs, patched_model.parameters)
            labels_ = labels
        else:
            idx = i % (inputs.shape[0] // batch_size)
            outputs = patched_model(inputs[idx * batch_size:(idx + 1) * batch_size], patched_model.parameters)
            labels_ = labels[idx * batch_size:(idx + 1) * batch_size]
        loss = loss_fn(outputs, labels_).sum()
        grad = torch.autograd.grad(loss, patched_model.parameters.values(),
                                   retain_graph=True, create_graph=True, only_inputs=True)

        patched_model.parameters = OrderedDict((name, param - lr * grad_part)
                                               for ((name, param), grad_part)
                                               in zip(patched_model.parameters.items(), grad))

    if use_updates:
        patched_model.parameters = OrderedDict((name, param - param_origin)
                                               for ((name, param), (name_origin, param_origin))
                                               in zip(patched_model.parameters.items(), patched_model_origin.parameters.items()))
    return list(patched_model.parameters.values())


def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(input_gradient))[-50:]
    else:
        raise ValueError()

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                
                # if trial_gradient[i].shape != input_gradient[i].shape:
                #     print(f"Shape mismatch at layer {i}:")
                #     print(f"  Trial gradient shape: {trial_gradient[i].shape}")
                #     print(f"  Input gradient shape: {input_gradient[i].shape}")
                
                
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                   input_gradient[i].flatten(),
                                                                   0, 1e-10) * weights[i]
        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()
            # print(f"Trial Gradient Norm: {pnorm[0].sqrt().item():.4e}, Input Gradient Norm: {pnorm[1].sqrt().item():.4e}")
        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)

def kl_divergence_loss(mu, log_var):
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return kld / mu.size(0)


def plot_images(images, title="Generated Images", save_path=None):

    if isinstance(images, torch.Tensor):
        images_np = images.to(torch.float32).detach().cpu().numpy()
    else:
        images_np = images
        
    images_np = np.transpose(images_np, (0, 2, 3, 1))
    
    min_val = images_np.min()
    max_val = images_np.max()

    if min_val < 0 or max_val > 1:
        images_np = (images_np - min_val) / (max_val - min_val + 1e-5) # 加上 1e-5 防止除以零
    images_np = np.clip(images_np, 0, 1)
    
    # 计算需要绘制的图像数量
    num_images = images_np.shape[0]
    if num_images == 0:
        print("no img")
        return
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(title, fontsize=16)
    
    if num_images == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()
    for i in range(num_images):
        ax = axes_flat[i]
        if images_np[i].shape[-1] == 1:
            ax.imshow(images_np[i].squeeze(), cmap='gray')
        else:
            ax.imshow(images_np[i])
            
        ax.axis('off')
        ax.set_title(f"Image {i+1}")
    
    for i in range(num_images, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # plt.show()
    plt.close(fig)
    
def plot(tensor):
    dm = torch.tensor((0.485, 0.456, 0.406))[:, None, None].to(device='cuda:0')
    ds = torch.tensor((0.229, 0.224, 0.225))[:, None, None].to(device='cuda:0')
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)

    if tensor.shape[0] == 1:
        return plt.imshow(tensor[0].permute(1, 2, 0).cpu())
    else:
        n = tensor.shape[0]
        cols = min(4, n) 
        rows = (n + cols - 1) // cols 

        fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*6))

        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)

        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < n:
                    axes[i][j].imshow(tensor[idx].permute(1, 2, 0).cpu())
                    axes[i][j].axis('off')
                else:
                    axes[i][j].axis('off')

        plt.tight_layout()
        plt.show()
        return fig

#############################################    STAGE 1   #############################################  
########################################################################################################    

class Latent_Space_GradientReconstructor():
    def __init__(self, model, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1, device='cuda', output_dir="results_latent_space"):
        
        self.model = model
        self.config = _validate_config(config)
        self.num_images = num_images
        self.device = device
        self.mean_std = mean_std
        self.setup = dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

        
        self.vae = AutoencoderKL.from_pretrained("./sd-vae-ft-mse").to(self.device)
        self.vae_scaling = 0.18215

        self.vae_decoder = self.vae.decoder

        self.model.eval()
        self.vae_decoder.eval()

        for param in self.vae_decoder.parameters(): param.requires_grad = False

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        

        self.bn_layers = [module for module in self.model.modules() if isinstance(module, nn.BatchNorm2d)]
        self.feature_hooks = [DeepInversionFeatureHook(layer) for layer in self.bn_layers]
        print(f"Registered {len(self.feature_hooks)} feature hooks for BN statistics.")

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    def reconstruct(self, input_data, labels, latent_shape=(4, 28, 28), dryrun=False, batch_bn_stats=None):
        print("Latent Space Reconstruction")
        final_latent_shape = (self.num_images, *latent_shape)
        self.latent_shape = final_latent_shape
        self.input_gradients = input_data
        
        self.true_bn_grads = batch_bn_stats        
        
        self.labels = labels
        
        loss_history = {"stage1": []}
        z_trial = self._init_latent(final_latent_shape)
        reconstructed_image, final_stats = self._run_trial(z_trial, dryrun=dryrun)
        loss_history["stage1"] = final_stats['loss']
        print("\n" + "="*20 + f"Stage 1 finished. Recorded {len(loss_history['stage1'])} steps of loss." + "="*20)

        for hook in self.feature_hooks:
            hook.close()
        

        return reconstructed_image.detach(), final_stats

    def _init_latent(self, latent_shape):
        if self.config['init'] == 'randn':
            return torch.randn(latent_shape, **self.setup).requires_grad_(True)
        return torch.zeros(latent_shape, **self.setup).requires_grad_(True)

    def _run_trial(self, z_trial, dryrun=False):

        stats = defaultdict(list)
        optimizer = torch.optim.AdamW([z_trial], lr=self.config['stage1_lr'], weight_decay=1e-4)
        max_iterations = self.config['stage1_latent_iter']
        scheduler = None
        dm,ds = self.mean_std
        if self.config['lr_decay']:
            
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[max_iterations // 2.667, max_iterations // 1.6,

                                                             max_iterations // 1.142], gamma=0.1)  
            

        try:
            with tqdm(range(max_iterations), desc="Latent Space DLG") as pbar:
                for iteration in pbar:
                    closure = self._gradient_closure(optimizer, z_trial, iteration)
                    rec_loss = optimizer.step(closure)
                    stats['loss'].append(rec_loss.item())

                    if scheduler:
                        scheduler.step()
                    
                    pbar.set_postfix({
                        'Loss': rec_loss.item(),
                        'LR': optimizer.param_groups[0]['lr']
                    })
                    
                    if (iteration + 1) % 100 == 0:
                        with torch.no_grad():
                            intermediate_img_r = self.vae.decode(z_trial/self.vae_scaling)
                            intermediate_img = intermediate_img_r.sample
 
                        plot_images_grid(intermediate_img, f"Latent DLG Iter {iteration+1}", 
                                         os.path.join(self.output_dir, f"iter_{iteration+1}.png"))
                    
                    if dryrun: break
        
        except KeyboardInterrupt: print("Latent Space Search interrupted.")
        
        with torch.no_grad():

            reconstructed_image_r = self.vae.decode(z_trial/ self.vae_scaling)
            reconstructed_image = reconstructed_image_r.sample

        output = reconstructed_image.detach()

        
        return output, stats
    
    
    def _gradient_closure(self, optimizer, z_trial, iteration):
        dm,ds = self.mean_std
        def closure():
            optimizer.zero_grad()
            
            feature_maps = []
            hooks = []
            for layer in self.bn_layers:
                def hook_fn(module, input, output):
                    feature_maps.append(input[0])
                hooks.append(layer.register_forward_hook(hook_fn))            
            
            

            x_dummy_r = self.vae.decode(z_trial/ self.vae_scaling)
            x_dummy = x_dummy_r.sample

            loss_ce = self.loss_fn(self.model(x_dummy), self.labels)
                
                
            for hook in hooks:
                hook.remove()

            dummy_gradient = torch.autograd.grad(loss_ce, self.model.parameters(), create_graph=True)

            grad_loss = reconstruction_costs([dummy_gradient], self.input_gradients,
                                cost_fn=self.config['stage1_cost_fn'], indices=self.config['stage1_indices'],
                                weights=self.config['stage1_weights'])
            
            ############################## z kl ##################################
            z_mean = z_trial.mean()
            z_var = z_trial.var()
            z_log_var = torch.log(z_var + 1e-8)     
            z_kl_loss = (0.5 * torch.mean(z_var + z_mean.pow(2) - 1 - z_log_var)) * self.config['stage1_z_kl_scale']
            ######################################################################
            
            
            
            ############################## z_reg  ##################################
            z_reg_loss = self.config['stage1_z_r_scale'] * torch.norm(z_trial, p=2)
            ######################################################################  
            
                        
            ############################## L2 ##################################
            l2_loss = torch.norm(x_dummy, p=2) * self.config['stage1_l2_scale']
            ######################################################################
            
            
            ############################## TV  ##################################
            tv_loss = self.config['stage1_tv_scale'] * TV(x_dummy)
            ######################################################################
            
            
            ############################## BN ##################################
            bn_loss_t = torch.tensor(0.0, device=self.device)
            for i, f_map in enumerate(feature_maps):
                bn_layer = self.bn_layers[i]
                dummy_mean = f_map.mean([0, 2, 3])
                dummy_var = f_map.var([0, 2, 3], unbiased=False)
                if self.true_bn_grads is not None:
                    target_mean, target_var = self.true_bn_grads[i]
                    # print('true_bn')
                else:
                    target_mean = bn_layer.running_mean
                    target_var = bn_layer.running_var
                    # print('None_bn')
                bn_loss_t += torch.norm(target_mean - dummy_mean, 2)
                bn_loss_t += torch.norm(target_var - dummy_var, 2)  
            bn_loss = self.config['stage1_bn_scale'] * bn_loss_t
            ######################################################################
            
            total_loss = self.config['stage1_grad_scale'] * grad_loss + tv_loss + z_reg_loss + bn_loss + z_kl_loss + l2_loss 
            
            total_loss.backward()
            
            if self.config['signed']:
                z_trial.grad.sign_()
            return total_loss
        return closure
########################################################################################################             
########################################################################################################     
        
        
        
        
#############################################    STAGE 2   #############################################  
########################################################################################################            

class Pixel_Space_GradientReconstructor():
    
    def __init__(self, model, re_img, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1, device='cuda'):
        self.config = _validate_config(config)
        self.model = model
        self.setup = dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)

        self.mean_std = mean_std
        self.num_images = num_images

        if self.config['scoring_choice'] == 'inception':
            self.inception = InceptionScore(batch_size=1, setup=self.setup)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.iDLG = True
        
        self.device = device
        self.re_img = re_img
        
        self.output_dir = "Stage2_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.bn_layers = [module for module in self.model.modules() if isinstance(module, nn.BatchNorm2d)]
        self.feature_hooks = [DeepInversionFeatureHook(layer) for layer in self.bn_layers]
        print(f"Registered {len(self.feature_hooks)} feature hooks for BN statistics.")
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    def reconstruct(self, input_data, labels, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None, batch_bn_stats=None):
        print("Pixel Space Reconstruct")
        start_time = time.time()
        if eval:
            self.model.eval()

        self.input_gradients = input_data

        self.true_bn_grads = batch_bn_stats
        
        x = self.re_img
        print("self.re_img shape",self.re_img.shape)
        print("x shape",x.shape)
        
        scores = torch.zeros(self.config['restarts'])

        loss_history = {"stage2":[]}
        
        try:
            x_trial, stats = self._run_trial(x, input_data, labels, dryrun=dryrun)
            x = x_trial
            loss_history["stage2"] = stats['loss']
            print("\n" + "="*20 + f"Stage 2 finished. Recorded {len(loss_history['stage2'])} steps of loss." + "="*20)
            for hook in self.feature_hooks:
                hook.close()
        


        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        return x_trial.detach(), stats

    def _init_images(self, img_shape):
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        else:
            raise ValueError()

    def _run_trial(self, x_trial, input_data, labels, dryrun=False):
        stats = defaultdict(list)

        x_trial.requires_grad = True

        if self.config['stage2_optim'] == 'adam':
            stage2_optimizer = torch.optim.Adam([x_trial], lr=self.config['stage2_lr'])
        elif self.config['stage2_optim'] == 'adamW': 
            stage2_optimizer = torch.optim.AdamW([x_trial], lr=self.config['stage2_lr'], weight_decay=1e-4)
        elif self.config['stage2_optim'] == 'sgd': 
            stage2_optimizer = torch.optim.SGD([x_trial], lr=0.01, momentum=0.9, nesterov=True)
        elif self.config['stage2_optim'] == 'LBFGS':
            stage2_optimizer = torch.optim.LBFGS([x_trial])
        else:
            raise ValueError()

        max_iterations = self.config['stage2_iterations']
        dm, ds = self.mean_std
        if self.config['lr_decay']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(stage2_optimizer,
                                                             milestones=[max_iterations // 2.667, max_iterations // 1.6,

                                                                         max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
        try:
            for iteration in range(max_iterations):
                closure = self._gradient_closure(stage2_optimizer, x_trial, input_data, labels, iteration)
                
                rec_loss = stage2_optimizer.step(closure)
                
                stats['loss'].append(rec_loss.item())
                
                if self.config['lr_decay']:
                    scheduler.step()
                
                alpha_n = self.config['stage2_alpha_noise']
                current_lr = stage2_optimizer.param_groups[0]['lr']

                with torch.no_grad():
    
                    if self.config['boxed']:
                        x_trial.data = torch.clamp(x_trial, min=-dm/ds, max=(1 - dm)/ds)

                    if (iteration + 1 == max_iterations) or iteration % 500 == 0:
                        print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')
                        plot_images_grid(x_trial, f"Pixel Space Search Iter {iteration}", 
                                         os.path.join(self.output_dir, f"iter_{iteration}.png"))
                    if (iteration + 1) % 500 == 0:
                        if self.config['filter'] == 'none':
                            pass
                        elif self.config['filter'] == 'median':
                            x_trial.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(x_trial)
                        else:
                            raise ValueError()
                if dryrun:
                    break

        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            display_reconstruction(x_trial, iteration)
            pass
        
        return x_trial.detach(), stats

    def _gradient_closure(self, stage2_optimizer, x_trial, input_gradient, label, iteration):

        def closure():
            stage2_optimizer.zero_grad()
            self.model.zero_grad()

            feature_maps = []
            hooks = []
            for layer in self.bn_layers:
                def hook_fn(module, input, output):
                    feature_maps.append(input[0])
                hooks.append(layer.register_forward_hook(hook_fn))


            loss = self.loss_fn(self.model(x_trial), label)

            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            rec_loss = reconstruction_costs([gradient], input_gradient,
                                            cost_fn=self.config['stage2_cost_fn'], indices=self.config['stage2_indices'],
                                            weights=self.config['stage2_weights'])
            
            


            for hook in hooks:
                hook.remove()

            ################################## BN #######################################################################     
            bn_loss_t = torch.tensor(0.0, device=self.device)
            for i, f_map in enumerate(feature_maps):
                bn_layer = self.bn_layers[i]
                # Calculate the batch-wise stats from the captured map
                dummy_mean = f_map.mean([0, 2, 3])
                dummy_var = f_map.var([0, 2, 3], unbiased=False)
                if self.true_bn_grads is not None:
                    target_mean, target_var = self.true_bn_grads[i]
                else:
                    target_mean = bn_layer.running_mean
                    target_var = bn_layer.running_var
                bn_loss_t += torch.norm(target_mean - dummy_mean, 2)
                bn_loss_t += torch.norm(target_var - dummy_var, 2)    
            bn_loss = self.config['stage2_bn_scale'] * bn_loss_t
            ################################################################################################################  
            
            ############################## TV  ##################################
            tv_loss = self.config['stage2_tv_scale'] * TV(x_trial)
            ##########################################################################
            
            current_iter = iteration

            l2_factor = max(0.0, 1.0 - current_iter / self.config['stage2_iterations'])

            constrast_factor = min(1.0, current_iter / self.config['stage2_iterations'])

            
            ############################## L2  ##################################
            l2_loss = torch.norm(x_trial, p=2) * self.config['stage2_l2_scale'] * l2_factor
            ##########################################################################
            
            ############################## contrast  ##################################
            contrast_loss = -torch.std(x_trial) * self.config['stage2_contrast_scale'] * constrast_factor
            ##########################################################################
            

            
            grad_loss_DLG=self.config['stage2_grad_scale']
            
            
            total_loss = grad_loss_DLG * rec_loss + tv_loss + bn_loss + l2_loss + contrast_loss       
            
            total_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return total_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            x_trial.grad = None
            loss = self.loss_fn(self.model(x_trial), label)
            print("loss score", loss)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            return reconstruction_costs([gradient], input_gradient,
                                        cost_fn=self.config['stage2_cost_fn'], indices=self.config['stage2_indices'],
                                        weights=self.config['stage2_weights'])
        elif self.config['scoring_choice'] == 'tv':
            return TV(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)
        elif self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            return 0.0
        else:
            raise ValueError()

    def _average_trials(self, x, labels, input_data, stats):
        print(f'Computing a combined result via {self.config["scoring_choice"]} ...')
        if self.config['scoring_choice'] == 'pixelmedian':
            x_optimal, _ = x.median(dim=0, keepdims=False)
        elif self.config['scoring_choice'] == 'pixelmean':
            x_optimal = x.mean(dim=0, keepdims=False)

        self.model.zero_grad()
        if self.reconstruct_label:
            labels = self.model(x_optimal).softmax(dim=1)
        loss = self.loss_fn(self.model(x_optimal), labels)
        gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
        stats['opt'] = reconstruction_costs([gradient], input_data,
                                            cost_fn=self.config['stage2_cost_fn'],
                                            indices=self.config['stage2_indices'],
                                            weights=self.config['stage2_weights'])
        print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats
########################################################################################################             
########################################################################################################          
        
        
        
        
        
        
        
        
#############################################    STAGE 3   #############################################  
########################################################################################################    
     
class Parameter_Space_GradientReconstructor():

    def __init__(self, model, initial_image, mean_std, config, input_gradients, labels, num_images, device='cuda'):
        self.model = model
        self.initial_image = initial_image.detach().requires_grad_(True)
        self.mean_std = mean_std
        self.config = config
        self.input_gradients = input_gradients
        self.labels = labels
        self.num_images = num_images
        self.device = device
        self.setup = dict(device=device, dtype=initial_image.dtype)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.output_dir = "Stage3_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.vae = AutoencoderKL.from_pretrained("./sd-vae-ft-mse").to(self.device)

        self.vae.encoder.train()
        self.vae.decoder.eval()

    def reconstruct(self, img_shape=(3, 224, 224)):
        print("Parameter Space Reconstruct")
        start_time = time.time()

        if self.config['stage3_optim'] == 'adam':
            stage3_optimizer = torch.optim.Adam(self.vae.encoder.parameters(), lr=self.config['stage3_lr'])
        elif self.config['stage3_optim'] == 'adamw':
            stage3_optimizer = torch.optim.AdamW(self.vae.encoder.parameters(), lr=self.config['stage3_lr'], weight_decay=1e-4)
        else:
            raise ValueError(f"Optimizer {self.config['optim']} not supported for VAERefiner.")

        train_epochs = self.config['stage3_train_epochs']
        scheduler = None
        if self.config['lr_decay']:

            scheduler = torch.optim.lr_scheduler.MultiStepLR(stage3_optimizer,
                                                 milestones=[train_epochs // 2.667, train_epochs // 1.6,
                                                             train_epochs // 1.142], gamma=0.1)
        
        loss_history={"stage3":[]}
        stats = defaultdict(list)
        try:
            with tqdm(range(train_epochs), desc="Refining Image (Stage 3)", unit="epoch") as pbar:
                
                for epoch in pbar:
                    closure = self._gradient_closure(stage3_optimizer, epoch)
                    rec_loss = stage3_optimizer.step(closure)
                    stats['loss'].append(rec_loss.item())
                    torch.nn.utils.clip_grad_norm_(self.vae.encoder.parameters(), max_norm=1.0) 
                    pbar.set_postfix({'Loss': f'{rec_loss.item():.4f}', 'LR': f'{stage3_optimizer.param_groups[0]["lr"]:.6f}'})
                    if scheduler:
                        scheduler.step()
        except KeyboardInterrupt:
            print('Refinement process manually interrupted.')
            pass

        loss_history["stage3"] = stats['loss']
        print("\n" + "="*20 + f"Stage 3 finished. Recorded {len(loss_history['stage3'])} steps of loss." + "="*20)
        
        with torch.no_grad():
            self.vae.eval()

            latent_representation = self.vae.encode(self.initial_image)
            
            latent_dist = latent_representation.latent_dist
            
            latent_codes = latent_dist.sample()
            
            vae_output = self.vae.decode(latent_codes)
            
            refined_image = vae_output.sample            
            

        plot_images(refined_image, title="Final Refined Image", save_path=os.path.join(self.output_dir, "refined_image_final.png"))
        # print(f"Refinement finished. Total time: {time.time() - start_time:.2f}s")

        return refined_image.detach(), stats

    def _gradient_closure(self, stage3_optimizer, epoch):
        def closure():
            stage3_optimizer.zero_grad()
            self.model.zero_grad()
            
            latent_representation = self.vae.encode(self.initial_image)

            latent_dist = latent_representation.latent_dist

            latent_codes = latent_dist.sample()

            vae_output = self.vae.decode(latent_codes)

            generated_images = vae_output.sample            

        ########################################################################################
            loss = self.loss_fn(self.model(generated_images), self.labels)
            
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

            grad_loss = reconstruction_costs([gradient], self.input_gradients,
                                                cost_fn=self.config['stage3_cost_fn'],
                                                indices=self.config['stage3_indices'],
                                                weights=self.config['stage3_weights'])
            grad_loss_p = self.config['stage3_grad_feature']    
            rec_loss = grad_loss * grad_loss_p


            tv_loss = self.config['stage3_tv_scale'] * TV(generated_images)
            
            total_loss = rec_loss + tv_loss
            
            total_loss.backward()
            return total_loss
        return closure
    
    



    



    
    


    
def plot_images_grid(images, title, save_path):

    if images is None: return
    try:

        images = (images / 2 + 0.5).clamp(0, 1)
        grid = torchvision.utils.make_grid(images.cpu().detach(), nrow=4)
        grid_img_plt = grid.permute(1, 2, 0)

        fig = plt.figure(figsize=(12, 12))
        plt.imshow(grid_img_plt)
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig) 

    except Exception as e:
        print(f"Could not save intermediate image. Error: {e}")


        
def extract_bn_gradients(target_model, all_gradients):
    bn_grads = {'gamma': [], 'beta': []}
    grad_dict = {name: grad for (name, _), grad in zip(target_model.named_parameters(), all_gradients)}

    for name, module in target_model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if module.weight is not None and f"{name}.weight" in grad_dict:
                bn_grads['gamma'].append(grad_dict[f"{name}.weight"])
            
            if module.bias is not None and f"{name}.bias" in grad_dict:
                bn_grads['beta'].append(grad_dict[f"{name}.bias"])
                
    return bn_grads
        
class DeepInversionFeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.r_feature = 0

    def hook_fn(self, module, input, output):

        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = (
            input[0]
            .permute(1, 0, 2, 3)
            .contiguous()
            .view([nch, -1])
            .var(1, unbiased=False)
        )
        
        mean_feature = torch.norm(module.running_mean.data - mean, 2)
        var_feature = torch.norm(module.running_var.data - var, 2)

        self.r_feature = mean_feature + var_feature

    def close(self):
        self.hook.remove()        
