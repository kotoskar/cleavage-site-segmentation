import json
import math
import pickle
from typing import Dict, List, Tuple
import os
from torch.utils.data import DataLoader

from src.models import (
    LSTMCNNCRF,
    LSTMCNNCRFProjector,
    LSTMCNNCRFProjectorMultiScale,
    LSTMCNNCRFSplitProjector,
    SimpleLSTMCNNCRF,
    SelfAttentionCRF,
    LSTMCNNCRFGated3DiResidual,
    LSTMCNNCRFGated3DiResidualConv,
    LSTMCNNCRFGated3DiResidualConvMultiScale,
    LSTMCNNCRFTriBranchResidual,
    LSTMCNNCRFAhoEmissionFusion,
    LSTMCNNCRFAhoMidFusion,
    LSTMCNNCRFAhoStateBias,
    LSTMCNNCRFBoundaryBondLoss,
    LSTMCNNCRFGated3DiBoundary,
    LSTMCNNCRFESM2LoRA,
    LSTMCNNCRFTelescopingSegmental,
)
from src.utils.dataset import PrecomputedCSVForOverlapCRFDataset, OnlineESMCSVForOverlapCRFDataset
from src.utils.manuscript_metrics import compute_all_metrics
from src.utils.seeding import set_seed, seeded_generator, seed_worker
from torch.optim import Adam
import torch
import numpy as np
import argparse
from torch.amp import GradScaler
from typing import List, Tuple
from tqdm import tqdm

def log_metrics(metrics: dict, filepath: str, prefix: str = "", aim_run=None, epoch=None):
    """Log metrics from a dictionary to a file, one per line, sorted by key. Optionally add a prefix to each line."""
    with open(filepath, 'a') as f:
        f.write(f'Epoch: {epoch}\n')
        for key in sorted(metrics.keys()):
            sep = "" if prefix == "" else f"{prefix}/"
            f.write(f"{sep}{key}: {metrics[key]}\n")
    if aim_run:
        aim_run.track(metrics, epoch=epoch, context={'subset': prefix.lower()})
    

def get_dataloaders_loo(args: argparse.Namespace, partitions: List[int] = [0,1,2,3,4], restrict = None, leave_idxs = [], device = None)-> Tuple[DataLoader, DataLoader]:
    if args.embedding == 'precomputed':
        train_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=partitions, label_type=args.label_type, restrict=restrict, device=device)
        valid_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=partitions, label_type=args.label_type, restrict=leave_idxs, device=device)
    elif args.embedding == 'online_esm2':
        train_set = OnlineESMCSVForOverlapCRFDataset(args.data_file, args.partitioning_file, partitions=partitions, label_type=args.label_type, restrict=restrict, device=device, max_sequence_length=args.esm2_max_sequence_length)
        valid_set = OnlineESMCSVForOverlapCRFDataset(args.data_file, args.partitioning_file, partitions=partitions, label_type=args.label_type, restrict=leave_idxs, device=device, max_sequence_length=args.esm2_max_sequence_length)
    else:
        raise NotImplementedError(args.embedding)

    print('Loading LOO data...')
    train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=train_set.collate_fn,
            num_workers=8,        
            pin_memory=True,
            persistent_workers=True,  
            prefetch_factor=2,
        )
    
    valid_loader = DataLoader(
            valid_set,
            batch_size=args.batch_size,
            collate_fn=valid_set.collate_fn,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

    print(f'Loaded data. {len(train_set)} train sequences, {len(valid_set)} validation sequences.')

    return train_loader, valid_loader

def get_dataloaders(args: argparse.Namespace, train_partitions: List[int] = [0,1,2], valid_partitions: List[int] = [3], test_partitions: List[int] = [4], restrict = None, device = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if args.embedding == 'precomputed':
        train_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=train_partitions, label_type=args.label_type, restrict=restrict, device=device)
        valid_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=valid_partitions, label_type=args.label_type, restrict=restrict, device=device)
        test_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=test_partitions, label_type=args.label_type, restrict=restrict, device=device)
    elif args.embedding == 'online_esm2':
        train_set = OnlineESMCSVForOverlapCRFDataset(args.data_file, args.partitioning_file, partitions=train_partitions, label_type=args.label_type, restrict=restrict, device=device, max_sequence_length=args.esm2_max_sequence_length)
        valid_set = OnlineESMCSVForOverlapCRFDataset(args.data_file, args.partitioning_file, partitions=valid_partitions, label_type=args.label_type, restrict=restrict, device=device, max_sequence_length=args.esm2_max_sequence_length)
        test_set = OnlineESMCSVForOverlapCRFDataset(args.data_file, args.partitioning_file, partitions=test_partitions, label_type=args.label_type, restrict=restrict, device=device, max_sequence_length=args.esm2_max_sequence_length)
    else:
        raise NotImplementedError(args.embedding)

    if restrict is None:
        print('Loading data...')
    else:
        print('Loading restricted data...')
    train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=train_set.collate_fn,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
            generator=seeded_generator(getattr(args, 'seed', 42)),  # fixed shuffle order
            worker_init_fn=seed_worker,
        )
    
    valid_loader = DataLoader(
            valid_set,
            batch_size=args.batch_size,
            collate_fn=valid_set.collate_fn,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None
        )
    
    test_loader = DataLoader(test_set,
            batch_size=args.batch_size,
            collate_fn=test_set.collate_fn,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None
        )
    print(f'Loaded data. {len(train_set)} train sequences (p.{train_partitions}), {len(valid_set)} validation sequences (p.{valid_partitions}), {len(test_set)} test sequences (p.{test_partitions}).')

    return train_loader, valid_loader, test_loader

def parse_kernel_list(value: str):
    return tuple(int(x.strip()) for x in value.split(',') if x.strip())

def get_model(args: argparse.Namespace):
    if args.model == 'lstmcnncrf':
        model = LSTMCNNCRF(
            input_size = args.embedding_dim,
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            dropout_input=args.dropout,
            num_states= 101 if 'with_propeptides' in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_conv1=args.conv_dropout,
            feature_extractor=args.feature_extractor
        )
    elif args.model == 'lstmcnncrf_simple':
        model = SimpleLSTMCNNCRF(
            input_size = args.embedding_dim,
            num_labels=3 if args.label_type == 'simple_with_propeptides' else 2,
            dropout_input=args.dropout,
            num_states= 3 if args.label_type == 'simple_with_propeptides' else 2,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_conv1=args.conv_dropout,
        )

    # NOTE just use already existing CLI args with names that don't really match. Works.
    elif args.model == 'selfattentioncrf':
        model = SelfAttentionCRF(
            input_size = args.embedding_dim,
            hidden_size= args.hidden_size,
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            dropout_input=args.dropout,
            num_states= 121 if 'with_propeptides' in args.label_type else 61,
            n_heads=args.num_filters,
            attn_dropout=args.conv_dropout,
        )
    elif args.model == 'lstmcnncrf_projector':
        model = LSTMCNNCRFProjector(
            input_size=args.embedding_dim,
            # projector:
            proj_size=256,                 # D_proj (adjustable)
            dropout_projector=0.4, # or fix at 0.2
            # rest is the same as LSTMCNNCRF:
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            num_states=101 if 'with_propeptides' in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_input=args.dropout,
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == 'lstmcnncrf_projector_multiscale':
        model = LSTMCNNCRFProjectorMultiScale(
            input_size=args.embedding_dim,
            proj_size=args.seq_proj_size,
            dropout_projector=args.projector_dropout,
            multiscale_kernels=parse_kernel_list(args.multiscale_kernels),
            multiscale_dropout=args.multiscale_dropout,
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            num_states=101 if 'with_propeptides' in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_input=args.dropout,
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == 'lstmcnncrf_projector_split':
        model = LSTMCNNCRFSplitProjector(
            input_size=args.embedding_dim,
            seq_input_size=args.seq_input_size,
            struct_input_size=args.struct_input_size,
            seq_proj_size=args.seq_proj_size,
            struct_proj_size=args.struct_proj_size,
            dropout_projector=args.projector_dropout,
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            num_states=101 if 'with_propeptides' in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_input=args.dropout,
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == 'lstmcnncrf_gated3diresidual':
        model = LSTMCNNCRFGated3DiResidual(
            input_size=args.embedding_dim,
            seq_input_size=args.seq_input_size,
            struct_input_size=args.struct_input_size,
            seq_proj_size=args.seq_proj_size,
            struct_proj_size=args.struct_proj_size,
            dropout_projector=args.projector_dropout,
            residual_scale=args.gated_residual_scale,
            struct_branch_dropout=args.struct_branch_dropout,
            gate_bias=args.gated_gate_bias,
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            num_states=101 if 'with_propeptides' in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_input=args.dropout,
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == 'lstmcnncrf_gated3diresidual_conv':
        model = LSTMCNNCRFGated3DiResidualConv(
            input_size=args.embedding_dim,
            seq_input_size=getattr(args, 'seq_input_size', 1280),
            struct_input_size=getattr(args, 'struct_input_size', 20),
            seq_proj_size=getattr(args, 'seq_proj_size', 256),
            struct_proj_size=getattr(args, 'struct_proj_size', 16),
            dropout_projector=getattr(args, 'projector_dropout', 0.4),
            residual_scale=getattr(args, 'gated_residual_scale', 0.1),
            struct_branch_dropout=getattr(args, 'struct_branch_dropout', 0.5),
            gate_bias=getattr(args, 'gated_gate_bias', -2.5),
            struct_conv_kernel=getattr(args, 'struct_conv_kernel', 5),
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            num_states=101 if 'with_propeptides' in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_input=args.dropout,
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == 'lstmcnncrf_gated3diresidual_conv_multiscale':
        model = LSTMCNNCRFGated3DiResidualConvMultiScale(
            input_size=args.embedding_dim,
            seq_input_size=getattr(args, 'seq_input_size', 1280),
            struct_input_size=getattr(args, 'struct_input_size', 20),
            seq_proj_size=getattr(args, 'seq_proj_size', 256),
            struct_proj_size=getattr(args, 'struct_proj_size', 16),
            dropout_projector=getattr(args, 'projector_dropout', 0.4),
            multiscale_kernels=parse_kernel_list(args.multiscale_kernels),
            multiscale_dropout=args.multiscale_dropout,
            residual_scale=getattr(args, 'gated_residual_scale', 0.1),
            struct_branch_dropout=getattr(args, 'struct_branch_dropout', 0.5),
            gate_bias=getattr(args, 'gated_gate_bias', -2.5),
            struct_conv_kernel=getattr(args, 'struct_conv_kernel', 5),
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            num_states=101 if 'with_propeptides' in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_input=args.dropout,
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == 'lstmcnncrf_aho_emission_fusion':
        model = LSTMCNNCRFAhoEmissionFusion(
            input_size=args.embedding_dim,
            seq_input_size=args.seq_input_size,
            aho_input_size=args.residue_input_size,
            aho_hidden_size=args.aho_hidden_size,
            aho_dropout=args.aho_dropout,
            aho_scale=args.aho_scale,
            aho_branch_dropout=args.aho_branch_dropout,
            aho_zero_init=not args.aho_no_zero_init,
            aho_none_scale=args.aho_none_scale,
            aho_pep_scale=args.aho_pep_scale,
            aho_propep_scale=args.aho_propep_scale,
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            num_states=101 if 'with_propeptides' in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_input=args.dropout,
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == 'lstmcnncrf_aho_mid_fusion':
        model = LSTMCNNCRFAhoMidFusion(
            input_size=args.embedding_dim,
            seq_input_size=args.seq_input_size,
            aho_input_size=args.residue_input_size,
            aho_hidden_size=args.aho_hidden_size,
            aho_mid_hidden_size=args.aho_mid_hidden_size,
            aho_dropout=args.aho_dropout,
            aho_scale=args.aho_scale,
            aho_branch_dropout=args.aho_branch_dropout,
            aho_zero_init=not args.aho_no_zero_init,
            aho_none_scale=args.aho_none_scale,
            aho_pep_scale=args.aho_pep_scale,
            aho_propep_scale=args.aho_propep_scale,
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            num_states=101 if 'with_propeptides' in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_input=args.dropout,
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == 'lstmcnncrf_aho_state_bias':
        model = LSTMCNNCRFAhoStateBias(
            input_size=args.embedding_dim,
            seq_input_size=args.seq_input_size,
            aho_input_size=args.residue_input_size,
            aho_feature_names_file=args.aho_feature_names_file,
            aho_state_boundary_feature=args.aho_state_boundary_feature,
            aho_state_scale=args.aho_state_scale,
            aho_state_branch_dropout=args.aho_state_branch_dropout,
            aho_state_bias_trainable=args.aho_state_bias_trainable,
            aho_state_pep_inside_bias=args.aho_state_pep_inside_bias,
            aho_state_pep_start_bias=args.aho_state_pep_start_bias,
            aho_state_pep_end_bias=args.aho_state_pep_end_bias,
            aho_state_propep_inside_bias=args.aho_state_propep_inside_bias,
            aho_state_propep_start_bias=args.aho_state_propep_start_bias,
            aho_state_propep_end_bias=args.aho_state_propep_end_bias,
            aho_state_pep_to_propep_inside_bias=args.aho_state_pep_to_propep_inside_bias,
            aho_state_pep_to_propep_start_bias=args.aho_state_pep_to_propep_start_bias,
            aho_state_pep_to_propep_end_bias=args.aho_state_pep_to_propep_end_bias,
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            num_states=101 if 'with_propeptides' in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_input=args.dropout,
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == 'lstmcnncrf_boundary_bond_loss':
        model = LSTMCNNCRFBoundaryBondLoss(
            input_size=args.embedding_dim,
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            num_states=101 if 'with_propeptides' in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_input=args.dropout,
            dropout_conv1=args.conv_dropout,
            feature_extractor=args.feature_extractor,
            boundary_state_hidden_size=args.boundary_state_hidden_size,
            boundary_state_dropout=args.boundary_state_dropout,
            boundary_state_scale=args.boundary_state_scale,
            boundary_state_zero_init=not args.boundary_state_no_zero_init,
            boundary_window=getattr(args, 'boundary_window', 1),
            bond_loss_lambda=args.bond_loss_lambda,
            bond_soft_window=args.bond_soft_window,
            bond_soft_tau=args.bond_soft_tau,
            bond_soft_mode=args.bond_soft_mode,
            bond_positive_weight=args.bond_positive_weight,
            bond_hidden_size=args.bond_hidden_size,
            bond_dropout=args.bond_dropout,
            bond_zero_init=args.bond_zero_init,
        )
    elif args.model == 'lstmcnncrf_gated3di_boundary':
        model = LSTMCNNCRFGated3DiBoundary(
            input_size=args.embedding_dim,
            seq_input_size=getattr(args, 'seq_input_size', 2560),
            struct_input_size=getattr(args, 'struct_input_size', 20),
            seq_proj_size=getattr(args, 'seq_proj_size', 256),
            struct_proj_size=getattr(args, 'struct_proj_size', 32),
            dropout_projector=getattr(args, 'projector_dropout', 0.4),
            residual_scale=getattr(args, 'gated_residual_scale', 0.1),
            struct_branch_dropout=getattr(args, 'struct_branch_dropout', 0.3),
            gate_bias=getattr(args, 'gated_gate_bias', -2.5),
            struct_conv_kernel=getattr(args, 'struct_conv_kernel', 5),
            boundary_state_hidden_size=args.boundary_state_hidden_size,
            boundary_state_dropout=args.boundary_state_dropout,
            boundary_state_scale=args.boundary_state_scale,
            boundary_state_zero_init=not args.boundary_state_no_zero_init,
            boundary_window=getattr(args, 'boundary_window', 1),
            seq_only=getattr(args, 'gated_seq_only', False),
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            num_states=101 if 'with_propeptides' in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_input=args.dropout,
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == 'lstmcnncrf_esm2_lora':
        model = LSTMCNNCRFESM2LoRA(
            esm2_model_name=args.esm2_model_name,
            esm2_repr_layer=args.esm2_repr_layer,
            esm2_max_sequence_length=args.esm2_max_sequence_length,
            lora_rank=args.esm2_lora_rank,
            lora_alpha=args.esm2_lora_alpha,
            lora_dropout=args.esm2_lora_dropout,
            lora_target_modules=args.esm2_lora_target_modules,
            lora_layers=args.esm2_lora_layers,
            train_esm_layer_norm=args.esm2_lora_train_layer_norm,
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            num_states=101 if 'with_propeptides' in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_input=args.dropout,
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == 'lstmcnncrf_tribranchresidual':
        model = LSTMCNNCRFTriBranchResidual(
            input_size=args.embedding_dim,
            seq_input_size=args.seq_input_size,
            residue_input_size=args.residue_input_size,
            struct_input_size=args.struct_input_size,
            seq_proj_size=args.seq_proj_size,
            residue_proj_size=args.residue_proj_size,
            struct_proj_size=args.struct_proj_size,
            dropout_projector=args.projector_dropout,
            residue_residual_scale=args.residue_residual_scale,
            struct_residual_scale=args.struct_residual_scale,
            residue_branch_dropout=args.residue_branch_dropout,
            struct_branch_dropout=args.struct_branch_dropout,
            residue_gate_bias=args.residue_gate_bias,
            struct_gate_bias=args.struct_gate_bias,
            struct_conv_kernel=args.struct_conv_kernel,
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            num_states=101 if 'with_propeptides' in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_input=args.dropout,
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == 'lstmcnncrf_telescoping_segmental':
        model = LSTMCNNCRFTelescopingSegmental(
            input_size=args.embedding_dim,
            dropout_input=args.dropout,
            n_filters=args.num_filters,
            filter_size=args.kernel_size,
            dropout_conv1=args.conv_dropout,
            hidden_size=args.hidden_size,
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            feature_extractor=args.feature_extractor,
            segmental_max_len=getattr(args, 'segmental_max_len', 50),
            segmental_min_len=getattr(args, 'segmental_min_len', 1),
            position_score_mode=getattr(args, 'position_score_mode', 'neg_abs'),
            position_score_tau=getattr(args, 'position_score_tau', 0.25),
            position_score_scale=getattr(args, 'position_score_scale', 0.25),
            relative_position_loss_lambda=getattr(args, 'relative_position_loss_lambda', 0.0),
            allow_same_label_segments=not getattr(args, 'disallow_same_label_segments', False),
        )
    else:
        raise NotImplementedError(args.model)

    print('trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model


def train_homo_loo(args, run = None):
    '''
    DEPRECATED
    '''
    # Portability fix for the anonymized release: fall back to CPU instead of
    # hard-requiring CUDA (mirrors infer.py's resolve_device()).
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    
    homo_file = os.path.normpath(os.path.join(args.embeddings_dir, '../../protein_id_homo.txt'))
    if os.path.exists(homo_file):
        with open(homo_file, 'r') as f:
            homo_ids = [line.strip() for line in f if line.strip()]
    else:
        homo_ids = []
        
    run['hparams'] = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'feature_extractor': args.feature_extractor,
        'homo_only': args.homo_only
    }
    
    for i in tqdm(range(0, len(homo_ids), args.K), desc='LOO partitions'):
        train_loader, valid_loader = get_dataloaders_loo(args, list(range(5)), restrict=homo_ids[:i] + homo_ids[min(i+args.K, len(homo_ids)):], leave_idxs=homo_ids[i:i+args.K], device=device)
        
        model = get_model(args).to(device)

        if hasattr(model, 'feature_extractor') and hasattr(model.feature_extractor, 'biLSTM'):
            model.feature_extractor.biLSTM.flatten_parameters()

        optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr = args.lr)
        use_amp = getattr(args, "amp", False)

        amp_dtype = torch.bfloat16  # default
        if getattr(args, "amp_dtype", "bf16") == "fp16":
            amp_dtype = torch.float16

        scaler = GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
        
        for epoch in tqdm(range(args.epochs), desc='loo epochs', dynamic_ncols=True):
            train_loss, train_probs, train_preds, train_peptides, train_labels = run_dataloader(
                train_loader, model, optimizer, do_train=True, device=device, scaler=scaler,
                use_amp=use_amp, amp_dtype=amp_dtype, collect_outputs=False,
                desc=f'Train LOO epoch {epoch + 1}/{args.epochs}'
            )
            
            # Log valid metrics
            valid_loss, valid_probs, valid_preds, valid_peptides, valid_labels = run_dataloader(
                valid_loader, model, optimizer, do_train=False, device=device, scaler=scaler,
                use_amp=use_amp, amp_dtype=amp_dtype, desc=f'Valid LOO epoch {epoch + 1}/{args.epochs}'
            )
            valid_metrics = compute_all_metrics(valid_probs, valid_preds, valid_labels, valid_loader.dataset.names, valid_loader.dataset.data, windows = [3])[0]
            log_metrics(valid_metrics, os.path.join(args.out_dir, 'all_metrics.txt'), prefix=f'Part {i//args.K} of LLO with K={args.K} (Homo)', aim_run=run, epoch=epoch)     
        
    return None

def train(args, train_partitions: List[int] = [0,1,2], valid_partitions: List[int] = [3], test_partitions: List[int] = [4], run = None):

    # Reproducibility: seed everything before any model / dataloader is built.
    set_seed(getattr(args, 'seed', 42))

    # FOR 4060TI 16GB
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Portability fix for the anonymized release: the original code hard-required
    # CUDA (`assert torch.cuda.is_available()`), which makes this entry point
    # impossible to run or smoke-test on a CPU-only machine. Fall back to CPU
    # instead, mirroring the CPU fallback infer.py's resolve_device() already uses.
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    homo_file = os.path.normpath(os.path.join(args.embeddings_dir, '../../protein_id_homo.txt'))
    if os.path.exists(homo_file):
        with open(homo_file, 'r') as f:
            homo_ids = [line.strip() for line in f if line.strip()]
    else:
        homo_ids = []
        
    if args.homo_only:
        # return train_homo_loo(args, run)
        train_loader, valid_loader, test_loader = get_dataloaders(args, train_partitions, valid_partitions, test_partitions, device=device, restrict=homo_ids)
    else:
        train_loader, valid_loader, test_loader = get_dataloaders(args, train_partitions, valid_partitions, test_partitions, device=device)
        homo_train_loader, homo_valid_loader, homo_test_loader = get_dataloaders(args, train_partitions, valid_partitions, test_partitions, restrict=homo_ids, device=device)
    
    
    if not os.path.exists(args.checkpoints_dir):
        os.mkdir(args.checkpoints_dir)

    model = get_model(args).to(device)

    if hasattr(model, 'feature_extractor') and hasattr(model.feature_extractor, 'biLSTM'):
        model.feature_extractor.biLSTM.flatten_parameters()

    optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr = args.lr)
    use_amp = getattr(args, "amp", False)

    amp_dtype = torch.bfloat16  # default
    if getattr(args, "amp_dtype", "bf16") == "fp16":
        amp_dtype = torch.float16

    scaler = GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    previous_best = -100000000000
    best_val_metrics = None
    start_epoch = 0

    # Optional resume: opt-in via args.resume (default off, so existing callers/scripts
    # are unaffected). Granularity = the periodic checkpoint cadence below (every 10
    # epochs, plus epoch 0) -- no per-batch/per-epoch state is kept, so a resume can
    # redo up to ~9 epochs of the interrupted run.
    if getattr(args, 'resume', False) and os.path.isdir(args.checkpoints_dir):
        ckpt_files = [f for f in os.listdir(args.checkpoints_dir) if f.startswith('model_') and f.endswith('.pth')]
        if ckpt_files:
            latest_epoch = max(int(f[len('model_'):-len('.pth')]) for f in ckpt_files)
            ckpt = torch.load(f'{args.checkpoints_dir}/model_{latest_epoch}.pth', map_location=device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                if 'optimizer_state_dict' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            else:
                # backward-compat: older checkpoints are a raw model state_dict (no optimizer state)
                model.load_state_dict(ckpt)
            start_epoch = latest_epoch + 1
            best_metrics_path = os.path.join(args.out_dir, 'valid_metrics.json')
            if os.path.exists(best_metrics_path):
                best_val_metrics = json.load(open(best_metrics_path))
                previous_best = (best_val_metrics['f1 peptides'] + best_val_metrics['f1 propeptides']) / 2
            print(f"[resume] loaded {args.checkpoints_dir}/model_{latest_epoch}.pth, "
                  f"resuming at epoch {start_epoch} (previous_best={previous_best})", flush=True)

    run['hparams'] = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'feature_extractor': args.feature_extractor,
        'homo_only': args.homo_only
    }

    epoch_bar = tqdm(range(start_epoch, args.epochs), initial=start_epoch, total=args.epochs,
                      desc='epochs', dynamic_ncols=True)
    for epoch in epoch_bar:
        train_loss, train_probs, train_preds, train_peptides, train_labels = run_dataloader(
            train_loader,
            model,
            optimizer,
            do_train=True,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            collect_outputs=False,
            desc=f'Train epoch {epoch + 1}/{args.epochs}',
        )
        
        # Log valid metrics
        valid_loss, valid_probs, valid_preds, valid_peptides, valid_labels = run_dataloader(
            valid_loader,
            model,
            optimizer,
            do_train=False,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            desc=f'Valid epoch {epoch + 1}/{args.epochs}',
        )
        valid_metrics = compute_all_metrics(valid_probs, valid_preds, valid_labels, valid_loader.dataset.names, valid_loader.dataset.data, windows = [3])[0]
        log_metrics(valid_metrics, os.path.join(args.out_dir, 'all_metrics.txt'), prefix='Valid', aim_run=run, epoch=epoch)
        epoch_bar.set_postfix({
            'train_loss': f'{train_loss:.3f}',
            'valid_loss': f'{valid_loss:.3f}',
            'f1_all': f"{valid_metrics.get('f1 all', 0.0):.3f}",
            'f1_pep': f"{valid_metrics.get('f1 peptides', 0.0):.3f}",
            'f1_pro': f"{valid_metrics.get('f1 propeptides', 0.0):.3f}",
        })
        tqdm.write(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} | "
            f"f1_all={valid_metrics.get('f1 all', 0.0):.4f} "
            f"pep={valid_metrics.get('f1 peptides', 0.0):.4f} "
            f"propep={valid_metrics.get('f1 propeptides', 0.0):.4f}"
        )
        if ((epoch + 1) % 10 == 0) or epoch == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }, f'{args.checkpoints_dir}/model_{epoch}.pth')

        # Best checkpoint selection
        stopping_metric = (valid_metrics['f1 peptides'] + valid_metrics['f1 propeptides'])/2
        if stopping_metric > previous_best:
            previous_best = stopping_metric
            best_val_metrics = valid_metrics
            # pickle.dump((valid_probs, valid_preds, valid_labels, valid_loader.dataset.names), open(os.path.join(args.out_dir, 'valid_outputs.pickle'), 'wb'))
            valid_metrics['epoch'] = epoch # keep track of best early stopping.
            json.dump(valid_metrics, open(os.path.join(args.out_dir, 'valid_metrics.json'), 'w'), indent=2)
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pt'))
            
        # Log metrics on homo subset
        if not args.homo_only:
            homo_valid_loss, homo_valid_probs, homo_valid_preds, homo_valid_peptides, homo_valid_labels = run_dataloader(
                homo_valid_loader, model, optimizer, do_train=False, device=device, scaler=scaler,
                use_amp=use_amp, amp_dtype=amp_dtype, desc=f'Valid homo epoch {epoch + 1}/{args.epochs}'
            )
            homo_valid_metrics = compute_all_metrics(homo_valid_probs, homo_valid_preds, homo_valid_labels, homo_valid_loader.dataset.names, homo_valid_loader.dataset.data, windows = [3])[0]
            log_metrics(homo_valid_metrics, os.path.join(args.out_dir, 'all_metrics.txt'), prefix='Valid (Homo)', aim_run=run, epoch=epoch)
            tqdm.write(f"Homo valid | loss={homo_valid_loss:.4f} | f1_all={homo_valid_metrics.get('f1 all', 0.0):.4f} pep={homo_valid_metrics.get('f1 peptides', 0.0):.4f} propep={homo_valid_metrics.get('f1 propeptides', 0.0):.4f}")

    # Log final test_metrics on best checkpoint
    model.load_state_dict(torch.load(os.path.join(args.out_dir, 'model.pt')))
    
    test_loss, test_probs, test_preds, test_peptides, test_labels = run_dataloader(test_loader, model, optimizer, do_train=False, device=device, desc='Test')
    test_metrics = compute_all_metrics(test_probs, test_preds, test_labels, test_loader.dataset.names, test_loader.dataset.data, windows = [3])[0]
    log_metrics(test_metrics, os.path.join(args.out_dir, 'all_metrics.txt'), prefix='Test', aim_run=run, epoch=epoch)
    # pickle.dump((test_probs, test_preds, test_labels, test_loader.dataset.names), open(os.path.join(args.out_dir, 'test_outputs.pickle'), 'wb'))
    json.dump(test_metrics, open(os.path.join(args.out_dir, 'test_metrics.json'), 'w'), indent=2)
    print('Test complete.')
    
    if not args.homo_only:
        homo_test_loss, homo_test_probs, homo_test_preds, homo_test_peptides, homo_test_labels = run_dataloader(homo_test_loader, model, optimizer, do_train=False, device=device, desc='Test homo')
        homo_test_metrics = compute_all_metrics(homo_test_probs, homo_test_preds, homo_test_labels, homo_test_loader.dataset.names, homo_test_loader.dataset.data, windows = [3])[0]
        log_metrics(homo_test_metrics, os.path.join(args.out_dir, 'all_metrics.txt'), prefix='Test (Homo)', aim_run=run, epoch=epoch)
        # pickle.dump((homo_test_probs, homo_test_preds, homo_test_labels, homo_test_loader.dataset.names), open(os.path.join(args.out_dir, 'homo_test_outputs.pickle'), 'wb'))
        json.dump(homo_test_metrics, open(os.path.join(args.out_dir, 'homo_test_metrics.json'), 'w'), indent=2)
        print('Test (Homo) complete.')
    
    return best_val_metrics, test_metrics


def run_dataloader(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    do_train: bool = True,
    device: str = "cpu",
    scaler=None,                 
    use_amp: bool = False,
    amp_dtype=torch.bfloat16,
    collect_outputs: bool = True,
    desc: str | None = None,
    show_progress: bool = True,
) -> Tuple[float, List[np.ndarray], List[List[int]], List[np.ndarray], List[np.ndarray]]:
    """
    Run a dataloader through the model. Collect predicted probabilities and
    true labels. Can be used both for training and prediction.
    """

    true = []
    labels = []
    probs = []
    preds = []
    epoch_loss = []

    model.train() if do_train else model.eval()

    n = 0
    # fp16 requires a scaler, bf16 does not
    use_scaler = bool(scaler is not None and getattr(scaler, "is_enabled", lambda: False)())
    if amp_dtype != torch.float16:
        use_scaler = False

    if desc is None:
        desc = 'Train' if do_train else 'Eval'

    iterator = enumerate(loader)
    progress = None
    if show_progress:
        progress = tqdm(
            iterator,
            total=len(loader),
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            smoothing=0.05,
        )
        iterator = progress

    for idx, batch in iterator:
        embeddings, mask, label, peptides = batch

        if torch.is_tensor(embeddings):
            embeddings = embeddings.to(device, non_blocking=True)
        # For --embedding online_esm2, embeddings is a list of raw sequence strings.
        mask = mask.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True).long()   # <-- dtype fix is required

        if do_train:
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                pos_probs, pos_preds, loss = model(embeddings, mask, label, skip_marginals=True, decode=False, return_probs=False)

            if not torch.isfinite(loss):
                # Non-finite loss (nan/inf): skip backward+step entirely so it can't
                # poison the weights. Model stays at last good state, training continues.
                optimizer.zero_grad(set_to_none=True)
                tqdm.write(f"[nan-guard] non-finite loss at batch {idx}, skipping optimizer step")
            else:
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if use_scaler:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                if not torch.isfinite(grad_norm):
                    # Loss was finite but gradients blew up (e.g. one bad batch under AMP).
                    # Same treatment: drop the update, keep the model, keep training.
                    optimizer.zero_grad(set_to_none=True)
                    tqdm.write(f"[nan-guard] non-finite grad norm at batch {idx}, skipping optimizer step")
                elif use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

        else:
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    pos_probs, pos_preds, loss = model(embeddings, mask, label)

        if collect_outputs:
            true.extend(peptides)
            probs.append(pos_probs.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
            preds.extend(pos_preds)
        batch_loss = float(loss.item())
        if math.isfinite(batch_loss):
            epoch_loss.append(batch_loss)
        running_loss = sum(epoch_loss) / max(1, len(epoch_loss))

        if progress is not None:
            progress.set_postfix(loss=f'{running_loss:.4f}', batch=f'{batch_loss:.4f}')

        n += 1

    if progress is not None:
        progress.close()

    epoch_loss = sum(epoch_loss) / max(1, len(epoch_loss))
    return epoch_loss, probs, preds, true, labels



def parse_arguments():
    '''Parse arguments, prepare output directory and dump run configuration.'''
    p = argparse.ArgumentParser()

    p.add_argument('--embeddings_dir', type=str, help='Directory of precomputed per-sequence embeddings (see src/utils/embedding_generators/).', default = 'data/embeddings/')
    p.add_argument('--checkpoints_dir', type=str, help='Dir to save checkpoints', default = './checkpoints')
    p.add_argument('--data_file', '-df', type=str, help='Sequences with Graph-Part headers', default = 'data/labeled_sequences.csv')
    p.add_argument('--partitioning_file', '-pf', type=str, help='Graph-Part output. Assume train-val-test split.', default = 'data/graphpart_assignments.csv')
    p.add_argument('--embedding', '-em', type=str, help='Sequence embedding strategy.', default='precomputed')
    p.add_argument('--embedding_dim', '-ed', type=int, help='Sequence embedding dimension.', default=1280)
    p.add_argument('--esm2_model_name', type=str, default='esm2_t33_650M_UR50D', help='fair-esm pretrained ESM2 loader used by --embedding online_esm2.')
    p.add_argument('--esm2_repr_layer', type=int, default=-1, help='ESM2 representation layer. -1 means final layer.')
    p.add_argument('--esm2_max_sequence_length', type=int, default=1022, help='Maximum residue length for online ESM2 batches. Longer proteins raise a clear error.')
    p.add_argument('--esm2_lora_rank', type=int, default=8, help='LoRA rank inside ESM2.')
    p.add_argument('--esm2_lora_alpha', type=float, default=16.0, help='LoRA alpha inside ESM2.')
    p.add_argument('--esm2_lora_dropout', type=float, default=0.05, help='Dropout on LoRA branch inside ESM2.')
    p.add_argument('--esm2_lora_target_modules', type=str, default='q_proj,k_proj,v_proj,out_proj,fc1,fc2', help='Comma-separated substrings of ESM2 Linear module names to wrap with LoRA.')
    p.add_argument('--esm2_lora_layers', type=str, default='all', help='Which ESM2 transformer layers receive LoRA: all, last:N, lastN, or comma-separated indices, e.g. 30,31,32.')
    p.add_argument('--esm2_lora_train_layer_norm', action='store_true', help='Also train ESM2 LayerNorm parameters.')
    p.add_argument('--seq_input_size', type=int, default=1280, help='Left part of concatenated embedding [esm | structure].')
    p.add_argument('--struct_input_size', type=int, default=20, help='Right part of concatenated embedding [esm | structure].')
    p.add_argument('--seq_proj_size', type=int, default=256, help='Projection size for the sequence branch.')
    p.add_argument('--residue_proj_size', type=int, default=16)
    p.add_argument('--struct_proj_size', type=int, default=64, help='Projection size for the structural branch.')
    p.add_argument('--projector_dropout', type=float, default=0.4, help='Dropout used inside split projector adapters.')
    p.add_argument('--struct_conv_kernel', type=int, default=5, help='Kernel size for Conv1d over the 3Di branch before gated residual fusion.')
    p.add_argument('--multiscale_kernels', type=str, default='3,7,15', help='Comma-separated kernel sizes for the multi-scale Conv1d front-end.')
    p.add_argument('--multiscale_dropout', type=float, default=0.1, help='Dropout inside the multi-scale Conv1d front-end.')
    p.add_argument('--residue_input_size', type=int, default=10)
    p.add_argument('--aho_hidden_size', type=int, default=0, help='Hidden size for Aho emission head. 0 = linear head.')
    p.add_argument('--aho_mid_hidden_size', type=int, default=64, help='Hidden size for Aho mid-fusion residual head. 0 = linear residual head.')
    p.add_argument('--aho_dropout', type=float, default=0.1, help='Dropout inside Aho emission head.')
    p.add_argument('--aho_scale', type=float, default=1.0, help='Scale for additive Aho emission logits.')
    p.add_argument('--aho_branch_dropout', type=float, default=0.0, help='Drop the whole Aho branch per sample during training.')
    p.add_argument('--aho_no_zero_init', action='store_true', help='Do not zero-initialize the final Aho emission layer.')
    p.add_argument('--aho_none_scale', type=float, default=1.0, help='Scale for Aho correction to None emission. Use 0 for pep-only Aho masking.')
    p.add_argument('--aho_pep_scale', type=float, default=1.0, help='Scale for Aho correction to Peptide emission.')
    p.add_argument('--aho_propep_scale', type=float, default=1.0, help='Scale for Aho correction to Propeptide emission. Ignored for 2-label models.')
    p.add_argument('--aho_feature_names_file', type=str, default=None, help='Path to Aho feature_names.json for state-bias model.')
    p.add_argument('--aho_state_boundary_feature', type=str, default='binary', choices=['binary', 'decay', 'window'], help='Aho boundary features to use for state bias.')
    p.add_argument('--aho_state_scale', type=float, default=1.0, help='Global scale for Aho state-level boundary bias.')
    p.add_argument('--aho_state_branch_dropout', type=float, default=0.0, help='Drop the whole Aho state-bias branch per sample during training.')
    p.add_argument('--aho_state_bias_trainable', action='store_true', help='Make the six Aho state-bias coefficients trainable.')
    p.add_argument('--aho_state_pep_inside_bias', type=float, default=0.0)
    p.add_argument('--aho_state_pep_start_bias', type=float, default=0.0)
    p.add_argument('--aho_state_pep_end_bias', type=float, default=0.0)
    p.add_argument('--aho_state_propep_inside_bias', type=float, default=0.0)
    p.add_argument('--aho_state_propep_start_bias', type=float, default=0.0)
    p.add_argument('--aho_state_propep_end_bias', type=float, default=0.0)
    p.add_argument('--aho_state_pep_to_propep_inside_bias', type=float, default=0.0, help='Optional cross-bias: pep Aho features support propep states.')
    p.add_argument('--aho_state_pep_to_propep_start_bias', type=float, default=0.0, help='Optional cross-bias: pep start feature supports propep start state.')
    p.add_argument('--aho_state_pep_to_propep_end_bias', type=float, default=0.0, help='Optional cross-bias: pep end feature supports propep end state.')
    
    p.add_argument('--boundary_state_hidden_size', type=int, default=64, help='Hidden size for learned start/inside/end state-emission bias head. 0 = linear head.')
    p.add_argument('--boundary_state_dropout', type=float, default=0.1, help='Dropout in learned boundary state-emission head.')
    p.add_argument('--boundary_state_scale', type=float, default=1.0, help='Global scale for learned boundary state-emission logits.')
    p.add_argument('--boundary_state_no_zero_init', action='store_true', help='Do not zero-initialize the final boundary state-emission layer.')
    p.add_argument('--boundary_window', type=int, default=1, help='Boundary state-emission window W: per branch emit start_1..W, inside, end_1..W mapped to states {1..W} and {max_len..max_len-W+1}. 1 = original start/inside/end head.')
    p.add_argument('--bond_loss_lambda', type=float, default=0.02, help='Weight for auxiliary soft bond boundary loss.')
    p.add_argument('--bond_soft_window', type=int, default=5, help='Window around true boundaries for soft bond targets. Use -1 for no cutoff.')
    p.add_argument('--bond_soft_tau', type=float, default=1.5, help='Temperature/sigma for soft bond targets.')
    p.add_argument('--bond_soft_mode', type=str, default='exp', choices=['exp', 'gaussian'], help='Soft target shape for bond loss.')
    p.add_argument('--bond_positive_weight', type=float, default=10.0, help='Extra BCE weight multiplier: weight = 1 + alpha * y_soft.')
    p.add_argument('--bond_hidden_size', type=int, default=64, help='Hidden size for auxiliary bond head. 0 = linear head.')
    p.add_argument('--bond_dropout', type=float, default=0.1, help='Dropout inside auxiliary bond head.')
    p.add_argument('--bond_zero_init', action='store_true', help='Zero-initialize final bond-head layer.')
    
    p.add_argument('--gated_residual_scale', type=float, default=0.2, help='Scale of the gated 3Di residual added to the ESM2 branch.')
    p.add_argument('--residue_residual_scale', type=float, default=0.05)
    p.add_argument('--struct_residual_scale', type=float, default=0.10)
    
    p.add_argument('--residue_branch_dropout', type=float, default=0.2)
    p.add_argument('--struct_branch_dropout', type=float, default=0.3, help='Drop the whole structural branch per sample during training with this probability.')
    
    p.add_argument('--residue_gate_bias', type=float, default=-2.5)
    p.add_argument('--struct_gate_bias', type=float, default=-2.5)
    p.add_argument('--gated_seq_only', action='store_true', help='Gated-3Di projector: use ONLY the sequence adapter (LayerNorm+reproject), no struct/3Di branch. Isolates the adapter for ablations.')

    p.add_argument('--model', '-m', type=str, default='lstmcnncrf')

    p.add_argument('--out_dir', '-od', type=str, help='name that will be added to the runs folder output', default='runs/train_run')
    p.add_argument('--epochs', type=int, default=100, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', '-bs', type=int, default=48, help='samples that will be processed in parallel')

    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--conv_dropout', type=float, default=0.1)
    p.add_argument('--kernel_size', type=int, default=3)
    p.add_argument('--num_filters', type=int, default=32)
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--port', type=int, default=12355)
    p.add_argument('--feature_extractor', type=str, default='LSTMCNN')
    p.add_argument('--homo_only', action='store_true')
    p.add_argument('--K', type=int, default=10)
    p.add_argument('--amp', action='store_true', help='Enable mixed precision (AMP)')
    p.add_argument('--amp_dtype', type=str, default='bf16', choices=['fp16','bf16'])

    p.add_argument('--label_type', type=str, default='multistate_with_propeptides')
    p.add_argument('--seed', type=int, default=42, help='Global RNG seed for reproducibility (python/numpy/torch + deterministic algorithms).')

    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_dir, 'config.json'), 'w'), indent=3)

    return args
