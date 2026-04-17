#!/usr/bin/env python3
"""
Training script for fluid simulation with rotation features
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set GPU 0
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import yaml
import torch
from torch.backends import cudnn
from datasets.dataset_reader_physics_rota import read_data_train, read_data_val
from collections import namedtuple
import glob
import time
from utils.deeplearningutilities.torch import Trainer, MyCheckpointManager
from evaluate_network import evaluate_torch

_k = 1000
TrainParams = namedtuple('TrainParams', ['max_iter', 'base_lr', 'batch_size'])
train_params = TrainParams(60 * _k, 0.001, 2)  # batch_size=2 (reduce to 1 if GPU memory is insufficient)

def create_model(**kwargs):
    from models.default_torch_newIAFF_with_doubleattn_weightadd_yh import MyParticleNetwork
    """Returns an instance of the network for training and evaluation"""
    model = MyParticleNetwork(**kwargs)
    return model


def compute_angular_velocity(rotation_sequence, current_time):
    """
    Compute angular velocity from rotation sequence
    Args:
        rotation_sequence: (10,) array [t0, ω0, t1, ω1, t2, ω2, t3, ω3, t4, ω4]
        current_time: scalar, current simulation time
    Returns:
        angular_velocity: scalar (rad/s)
    """
    times = rotation_sequence[::2]  # [t0, t1, t2, t3, t4]
    omegas = rotation_sequence[1::2]  # [ω0, ω1, ω2, ω3, ω4]
    
    current_time_val = current_time.item() if torch.is_tensor(current_time) else current_time
    
    # Linear interpolation
    for i in range(len(times) - 1):
        if times[i] <= current_time_val <= times[i + 1]:
            t0, t1 = times[i], times[i + 1]
            w0, w1 = omegas[i], omegas[i + 1]
            if t1 - t0 > 1e-6:
                alpha = (current_time_val - t0) / (t1 - t0)
                omega = w0 + alpha * (w1 - w0)
            else:
                omega = w0
            device = current_time.device if torch.is_tensor(current_time) else 'cuda'
            return torch.as_tensor(omega, dtype=torch.float32, device=device)
    
    # Return nearest omega if outside range
    device = current_time.device if torch.is_tensor(current_time) else 'cuda'
    if current_time_val < times[0]:
        return torch.as_tensor(omegas[0], dtype=torch.float32, device=device)
    else:
        return torch.as_tensor(omegas[-1], dtype=torch.float32, device=device)


def main():
    global min_err
    min_err = float('inf')
    
    parser = argparse.ArgumentParser(description="Optimized rotation feature training script")
    parser.add_argument("--cfg",
                        type=str,
                        default=None,
                        help="Config file path")
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/home/zh/fueltank_datasets/datasets/simple_box_rotate",
                        help="Dataset directory path")
    parser.add_argument("--gpu",
                        type=str,
                        default="0",
                        help="GPU ID to use")
    parser.add_argument("--batch_size",
                        type=int,
                        default=2,
                        help="Batch size")
    parser.add_argument("--enable_rotation",
                        action='store_true',
                        help="Enable rotation features (pass rotation_info)")
    
    args = parser.parse_args()
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Load config file
    if args.cfg is None:
        args.cfg = os.path.join(os.path.dirname(__file__), 'default_rota.yaml')
    
    print(f"Config file: {args.cfg}")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"GPU used: {args.gpu}")
    print(f"Batch size: {args.batch_size}")
    print(f"Rotation features enabled: {args.enable_rotation}")
    
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override dataset path from config
    cfg['dataset_dir'] = args.dataset_dir
    
    # Create training directory
    dataset_name = os.path.basename(args.dataset_dir)
    train_dir = f"train_network_torch_rota_optimized_{dataset_name}"
    
    # Create model weights output directory
    modelweights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    f'modelweights_rota_{dataset_name}')
    os.makedirs(modelweights_dir, exist_ok=True)
    print(f"Model weights will be saved to: {modelweights_dir}")

    # Check dataset files
    val_files = sorted(glob.glob(os.path.join(cfg['dataset_dir'], 'valid', '*.zst')))
    train_files = sorted(glob.glob(os.path.join(cfg['dataset_dir'], 'train', '*.zst')))
    
    if len(train_files) == 0:
        print(f"Error: No training data found in {cfg['dataset_dir']}/train/!")
        return 1
    
    print(f"Number of training files: {len(train_files)}")
    print(f"Number of validation files: {len(val_files)}")

    device = torch.device("cuda")

    # Load dataset
    if len(val_files) > 0:
        val_dataset = read_data_val(files=val_files, window=1, cache_data=True)
    else:
        val_dataset = None
    
    train_dataset = read_data_train(files=train_files, window=1, cache_data=True)
    
    device = torch.device("cuda")

    data_iter = iter(train_dataset)

    trainer = Trainer(train_dir)

    # Create model
    model = create_model(**cfg.get('model', {}))
    model.cuda()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Learning rate scheduler
    boundaries = [15 * _k, 25 * _k, 35 * _k, 45 * _k, 50 * _k, 55 * _k]
    lr_values = [1.5, 1, 0.5, 0.25, 0.125, 0.125 * 0.5]

    def lrfactor_fn(x):
        factor = lr_values[0]
        for b, v in zip(boundaries, lr_values[1:]):
            if x > b:
                factor = v
            else:
                break
        return factor

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=train_params.base_lr,
                                 weight_decay=0.001,
                                 eps=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrfactor_fn)

    step = torch.tensor(0)
    checkpoint_fn = lambda: {
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    manager = MyCheckpointManager(checkpoint_fn,
                                  trainer.checkpoint_dir,
                                  keep_checkpoint_steps=list(
                                      range(1 * _k, train_params.max_iter + 1, 1 * _k)))

    def euclidean_distance(a, b, epsilon=1e-9):
        return torch.sqrt(torch.sum((a - b)**2, dim=-1) + epsilon)

    def loss_fn(pr_pos, gt_pos, num_fluid_neighbors):
        gamma = 0.5
        neighbor_scale = 1 / 40
        importance = torch.exp(-neighbor_scale * num_fluid_neighbors)
        return torch.mean(importance * euclidean_distance(pr_pos, gt_pos)**gamma)

    def train(model, batch):
        optimizer.zero_grad()
        losses = []
        batch_size = args.batch_size
        
        for batch_i in range(batch_size):
            # Validate data
            if batch['pos0'][batch_i].shape[0] == 0:
                print(f"Warning: batch {batch_i} has empty pos0, skipping")
                continue
                
            if args.enable_rotation:
                # Build rotation info dict (complete version)
                rotation_info_0 = {
                    'rotation_axis': batch['rotation_axis'][batch_i],
                    'angular_velocity': compute_angular_velocity(
                        batch['rotation_sequence'][batch_i],
                        batch['sim_time0'][batch_i]
                    ),
                    'rotation_center': batch['position0'][batch_i]
                }
                
                rotation_info_1 = {
                    'rotation_axis': batch['rotation_axis'][batch_i],
                    'angular_velocity': compute_angular_velocity(
                        batch['rotation_sequence'][batch_i],
                        batch['sim_time1'][batch_i]
                    ),
                    'rotation_center': batch['position1'][batch_i]
                }
            else:
                # No rotation features
                rotation_info_0 = None
                rotation_info_1 = None
            
            # First step prediction: from time 0 to time 1
            inputs = ([
                batch['pos0'][batch_i], batch['vel0'][batch_i], None,
                batch['box_ref'][batch_i], batch['box_normals_ref'][batch_i]
            ])
            pr_pos1, pr_vel1 = model(inputs, rotation_info=rotation_info_0)
            
            # Check first prediction results
            if pr_pos1.shape[0] == 0 or model.num_fluid_neighbors.shape[0] == 0:
                print(f"Warning: First prediction resulted in 0 particles, skipping batch {batch_i}")
                continue

            l = loss_fn(pr_pos1, batch['pos1'][batch_i],
                        model.num_fluid_neighbors)
            
            # Second step prediction: use ground truth pos1 as input (to avoid num_fluid_neighbors issues)
            inputs = (batch['pos1'][batch_i], batch['vel1'][batch_i], None, 
                      batch['box_ref'][batch_i], batch['box_normals_ref'][batch_i])
            
            pr_pos2, pr_vel2 = model(inputs, rotation_info=rotation_info_1)
            
            # Check second prediction results
            if pr_pos2.shape[0] == 0 or model.num_fluid_neighbors.shape[0] == 0:
                print(f"Warning: Second prediction resulted in 0 particles, skipping batch {batch_i}")
                continue

            l += loss_fn(pr_pos2, batch['pos2'][batch_i],
                         model.num_fluid_neighbors)
            losses.append(l)
        
        if len(losses) == 0:
            print("Warning: All samples in batch were skipped")
            return torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = 128 * sum(losses) / batch_size
        total_loss.backward()
        optimizer.step()

        return total_loss

    # Restore from checkpoint
    if manager.latest_checkpoint:
        print('Restoring from checkpoint:', manager.latest_checkpoint)
        latest_checkpoint = torch.load(manager.latest_checkpoint)
        step = latest_checkpoint['step']
        model.load_state_dict(latest_checkpoint['model'])
        optimizer.load_state_dict(latest_checkpoint['optimizer'])
        scheduler.load_state_dict(latest_checkpoint['scheduler'])

    # Training loop
    display_str_list = []
    print("\nStarting training...")
    print("="*60)
    
    while trainer.keep_training(step,
                                train_params.max_iter,
                                checkpoint_manager=manager,
                                display_str_list=display_str_list):
        
        data_fetch_start = time.time()
        batch = next(data_iter)

        batch_torch = {}
        # Particle and container data
        for k in ('pos0', 'vel0', 'pos1', 'vel1', 'pos2', 'box_ref', 'box_normals_ref'):
            batch_torch[k] = []
            for x in batch[k]:
                if isinstance(x, np.ndarray):
                    # Ensure array is contiguous and has correct type
                    x_array = np.ascontiguousarray(x, dtype=np.float32)
                    batch_torch[k].append(torch.from_numpy(x_array).to(device))
                else:
                    batch_torch[k].append(torch.tensor(x, dtype=torch.float32, device=device))
        
        # Rotation-related data (if enabled)
        if args.enable_rotation:
            for k in ('rotation_axis', 'rotation_sequence', 'position0', 'position1', 
                      'sim_time0', 'sim_time1'):
                # Handle possible scalar or special arrays
                batch_torch[k] = []
                for x in batch[k]:
                    if isinstance(x, np.ndarray):
                        # Ensure array is contiguous and has correct type
                        x_array = np.ascontiguousarray(x, dtype=np.float32)
                        batch_torch[k].append(torch.from_numpy(x_array).to(device))
                    else:
                        # Scalar value converted to tensor
                        batch_torch[k].append(torch.tensor(x, dtype=torch.float32, device=device))
        
        data_fetch_latency = time.time() - data_fetch_start
        trainer.log_scalar_every_n_minutes(5, 'DataLatency', data_fetch_latency)

        current_loss = train(model, batch_torch)
        scheduler.step()
        display_str_list = ['loss', float(current_loss)]

        if trainer.current_step % 10 == 0:
            trainer.summary_writer.add_scalar('TotalLoss', current_loss,
                                              trainer.current_step)
            trainer.summary_writer.add_scalar('LearningRate',
                                              scheduler.get_last_lr()[0],
                                              trainer.current_step)

        # Evaluation (temporarily disabled, re-enable after fixing num_fluid_neighbors issue)
        if False and val_dataset is not None and (trainer.current_step) % (1.0 * _k) == 0:
            print(f"\nStep {trainer.current_step}: Starting evaluation...")
            try:
                for k, v in evaluate_torch(model,
                                     val_dataset,
                                     frame_skip=20,
                                     device=device,
                                     **cfg.get('evaluation', {})).items():
                    trainer.summary_writer.add_scalar('eval/' + k, v,
                                                      trainer.current_step)
                    if(k == "err_n1" and v < min_err):
                        min_err = v
                        best_model_path = os.path.join(modelweights_dir, 
                                                       f'{step.item()}_model_weights_best.pt')
                        torch.save({'model': model.state_dict()}, best_model_path)
                        print(f"Update best model: err_n1={v:.6f}")
            except Exception as e:
                print(f"Evaluation failed: {e}")

    # Save final model
    final_model_path = os.path.join(modelweights_dir, 'model_weights_final.pt')
    torch.save({
        'model': model.state_dict(),
        'step': trainer.current_step,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    if trainer.current_step == train_params.max_iter:
        return trainer.STATUS_TRAINING_FINISHED
    else:
        return trainer.STATUS_TRAINING_UNFINISHED


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    sys.exit(main())
