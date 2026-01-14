import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import diffusers
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from inversefed.data.loss import Classification
import math
import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import inversefed
from inversefed import consts

from inversefed import Latent_Space_GradientReconstructor, Pixel_Space_GradientReconstructor, Parameter_Space_GradientReconstructor


config = dict(signed=True,
              boxed=True,
              stage1_cost_fn='sim',
              stage1_indices='def',
              stage1_weights='equal',
              stage1_lr=1e-1,
              stage1_optim='adamw',
              restarts=1,
              stage1_latent_iter=500,
              stage1_tv_scale=1e-1,
              stage1_z_r_scale=1e-3,
              stage1_bn_scale = 1e-2,
              stage1_l2_scale = 0.0001,  
              stage1_z_kl_scale = 0.01,
              stage1_grad_scale = 1,
              ########################################### 
              stage2_cost_fn='sim',
              stage2_indices='def',
              stage2_weights='equal',
              stage2_lr=1e-1,
              stage2_optim='adamW',
              stage2_iterations = 15000,               
              stage2_tv_scale=1e-1,
              stage2_bn_scale = 1e-3,
              stage2_l2_scale= 1e-4,  
              stage2_contrast_scale = 1e-1,
              stage2_grad_scale = 1,
              stage2_alpha_noise = 0.5,        
              ########################################### 
              stage3_cost_fn='sim',
              stage3_indices='def',
              stage3_weights='equal',
              stage3_lr = 1e-3,
              stage3_optim='adamw',
              stage3_train_epochs=500,
              stage3_tv_scale = 1e-3,
              stage3_grad_feature = 1,
              init='randn',
              filter='none',#median
              lr_decay=True,
              scoring_choice='loss')


trained_model = True
num_images = 8
arch = 'ResNet50'
mean_stats = [0.485, 0.456, 0.406]
std_stats = [0.229, 0.224, 0.225]


def label_inference_algorithm_2(delta_w_fc, batch_size, num_classes):

    if delta_w_fc.dim() == 1:
        total_elements = delta_w_fc.numel()
        input_features = total_elements // num_classes
        delta_W = delta_w_fc.view(input_features, num_classes).clone()
    else:
        if delta_w_fc.shape[1] != num_classes:
             delta_W = delta_w_fc.t().clone()
        else:
             delta_W = delta_w_fc.clone()

    y_hat = []
    b = 0
    
    max_iter = batch_size * 10 
    iter_count = 0

    while b < batch_size and iter_count < max_iter:
        iter_count += 1
        min_index_flat = torch.argmin(delta_W)
        m = min_index_flat // num_classes
        delta_W_m = delta_W[m] 
        
        new_labels = []
        for n in range(num_classes):
            val = delta_W_m[n]
            if val < 0:
                new_labels.append(n)

        y_hat.extend(new_labels)
        b = len(y_hat)

        if b > batch_size:
            y_hat = y_hat[:batch_size]
            b = batch_size 

        mean_val = torch.mean(delta_W[m])
        delta_W[m] = delta_W[m] - mean_val

    return y_hat


def grid_plot(tensor, infos, dm, ds, save_name="ground_truth.png"):
    print(f"Plotting images to {save_name}...")
    imgs = tensor.clone().detach()
    imgs.mul_(ds).add_(dm).clamp_(0, 1)
    
    n = len(imgs)
    rows = math.ceil(math.sqrt(n))
    cols = math.ceil(n / rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    if n > 1:
        axes = axes.flatten() 
    else:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        if i < n:
            im_display = imgs[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(im_display)
            
            info = infos[i]
            ax.set_title(f"Pos {i+1}: {info['class_name']}\n(Org ID: {info['original_index']})", fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
    print(f"Saved to {save_name}")

def main():
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')

    data_path = './demo_img'
    normalize = transforms.Normalize(mean=mean_stats, std=std_stats)

    train_transform = transforms.Compose([
            transforms.Resize(256),      
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    trainset = ImageFolder(root=f'{data_path}/train', transform=train_transform)
    validset = ImageFolder(root=f'{data_path}/train', transform=valid_transform)

    trainloader = DataLoader(trainset, batch_size=defs.batch_size, shuffle=True, num_workers=4)
    validloader = DataLoader(validset, batch_size=defs.batch_size, shuffle=False, num_workers=4)

    loss_fn = Classification()

    checkpoint_path = './moco_v2_800ep_pretrain.pth.tar'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.encoder_q.'):
                new_key = k.replace('module.encoder_q.', '')
                new_state_dict[new_key] = v
    else:
        print("Warning: Checkpoint not found, using random init.")
        new_state_dict = None
        
    model = torchvision.models.resnet50(pretrained=False)
    if new_state_dict:
        model.load_state_dict(new_state_dict, strict=False)

    model.to(**setup)
    model.eval()

    dm = torch.tensor(mean_stats, **setup)[:, None, None]
    ds = torch.tensor(std_stats, **setup)[:, None, None]

    indices_order = [7, 5, 3, 0, 2, 4, 1, 6] 
    dataset = validloader.dataset
    ground_truth_list = []
    image_infos = []

    for idx in indices_order:
        img, label = dataset[idx]
        
        ground_truth_list.append(img.to(**setup))
        
        class_name = dataset.classes[label] if hasattr(dataset, 'classes') else f"Label {label}"
        image_infos.append({
            'class_name': class_name,
            'label': label,
            'original_index': idx
        })

    ground_truth = torch.stack(ground_truth_list)
    true_labels = [889, 531, 351, 22, 275, 519, 55, 554]
    labels = torch.tensor(true_labels).to(device=setup['device'])

    print(f"Ground Truth Shape: {ground_truth.shape}")

    grid_plot(ground_truth, image_infos, dm, ds, save_name="gt_visualization.png")

    print("Calculating gradients...")
    model.zero_grad()
    target_loss, _, _ = loss_fn(model(ground_truth), labels)
    input_gradient = torch.autograd.grad(target_loss, model.parameters())
    input_gradient = [grad.detach() for grad in input_gradient]
    
    print("Gradient calculation done. Script finished.")
    
    device = 'cuda:0'
    print(device)
    
    delta_w_tensor = input_gradient[-1].view(-1)
    reconstructed_labels = label_inference_algorithm_2(delta_w_tensor, num_images, 1000)
    reconstructed_labels = torch.tensor(reconstructed_labels, dtype=torch.long).to(device)
    print("reconstructed_labels:", reconstructed_labels)

    print("\n" + "="*20 + f"Stage 1 Start." + "="*20)
    rec_machine = Latent_Space_GradientReconstructor(
        model,
        (dm, ds), 
        config, 
        num_images=num_images, 
        device=device, 
        output_dir = "Stage1_result")


    print("lables:",reconstructed_labels)
    output, stats = rec_machine.reconstruct(
        input_gradient, 
        labels=reconstructed_labels,
        batch_bn_stats=None
    )
    
    print("\n" + "="*20 + f"Stage 2 Start." + "="*20)
    rec_machine_2 = Pixel_Space_GradientReconstructor(
    model, 
    output, 
    mean_std=(dm, ds),
    config=config, 
    num_images = num_images, 
    device = device)


    output_GI, stats_GI = rec_machine_2.reconstruct(
        input_gradient, 
        labels=reconstructed_labels, 
        img_shape=(3, 224, 224), 
        batch_bn_stats=None)  

    print("\n" + "="*20 + f"Stage 3 Start." + "="*20)
    
    refiner_machine_3 = Parameter_Space_GradientReconstructor(
    model=model,
    initial_image=output_GI, 
    mean_std=(dm, ds),
    config=config,
    input_gradients=input_gradient,
    labels=reconstructed_labels,
    num_images = num_images,
    device=device
    )


    output_VAE, stats_VAE = refiner_machine_3.reconstruct(img_shape=(3, 224, 224))
    
    
if __name__ == '__main__':
    main()
