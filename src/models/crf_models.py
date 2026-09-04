'''
The CRF state space models. Many parameters are hardcoded due to the complexity of the CRF configuration.
'''
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from .multi_tag_crf import CRF
from .lstm_cnn import LSTMCNN
from .position_scores import compute_position_score

class CustomTransformerWrapper(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        from transformers import AutoModelForMaskedLM
        
        transformer = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_small', trust_remote_code=True)
        self.input_dim = input_dim
        self.hidden_dim = 960
        self.encoder = transformer.transformer.blocks

        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, output_dim) if output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim] - input embeddings
        Returns:
            [batch_size, seq_len, output_dim or hidden_dim]
        """
        x = self.input_proj(x)  # Project to 960
        for block in self.encoder:
            x, _ = block(x)
        x = self.output_proj(x)
        return x



def build_aho_label_scale_tensor(
    num_labels: int,
    none_scale: float = 1.0,
    pep_scale: float = 1.0,
    propep_scale: float = 1.0,
) -> torch.Tensor:
    """Return [1, 1, num_labels] scales for additive Aho emission corrections.

    Label convention used by the existing models:
      2 labels: [None, Peptide]
      3 labels: [None, Peptide, Propeptide]
    """
    if num_labels == 2:
        values = [none_scale, pep_scale]
    elif num_labels == 3:
        values = [none_scale, pep_scale, propep_scale]
    else:
        raise ValueError(f"Unsupported num_labels={num_labels}; expected 2 or 3")
    return torch.tensor(values, dtype=torch.float32).view(1, 1, num_labels)


class CRFBaseModel(nn.Module):
    '''Extend this model by defining a feature_extractor.'''
    def __init__(
        self,
        num_labels: int = 2, #logits (=emissions) to produce by the NN
        num_states = 61 # total number of states in the state space model
        ) -> None:


        super().__init__()
        self.max_len = 50
        self.min_len = 5
        self.feature_extractor = None
        self.features_to_emissions = nn.Linear(64, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(self.max_len, self.min_len, n_branches=2 if num_labels==3 else 1)
        self.allowed_transitions = allowed_transitions
        self.crf = CRF(num_states, batch_first=True, allowed_transitions=allowed_transitions, allowed_start=allowed_start, allowed_end=allowed_end)

    @staticmethod
    def get_crf_constraints(max_len: int = 60, min_len: int = 5, n_branches: int = 1):
        '''Build the peptide state space model.
        Each peptide starts as state 1 and goes through 2, 3.
        From 3, it can either go to 4 or skip ahead to any other state up to 59.
        From 59, go to 60. 
        This enforces a minimum peptide length of 5. Each peptide is forced to end in 60,
        so this state can learn peptide end properties.
        '''
        allowed_starts = [0,1]
        allowed_ends = [0, max_len]

        allowed_state_transitions = []
        allowed_state_transitions.append((0,0)) # None to None
        allowed_state_transitions.append((0,1)) # None to Peptide_0
        allowed_state_transitions.append((max_len,1)) # Peptide_50 to Peptide_0, no need to have 1 AA gap
        allowed_state_transitions.append((max_len,0)) # Peptide_50 (peptide end position) to None

        for i in range(1, max_len): 
            to_next = (i, i+1)
            allowed_state_transitions.append(to_next)

            if i >min_len-1: #make skip forward connections
                skip_to_i = (min_len-2,i) #3
                allowed_state_transitions.append(skip_to_i) 

        allowed_state_transitions.append((max_len-1,max_len)) # peptide end position -1 to peptide end position
        # logic of this state space model is that the end state is the same for all peptides, regardless their length.

        # branch 1 + no state: 0-50
        # branch 2: 51-101
        if n_branches == 2:
            start = 1 + max_len
            end = 2*max_len

            allowed_starts.append(start)
            allowed_ends.append(end)
            allowed_state_transitions.append((0,start))
            allowed_state_transitions.append((end,start)) 
            allowed_state_transitions.append((end,0))

            # can go directly from end of peptide to start of propeptide and vice versa.
            allowed_state_transitions.append((end,1))
            allowed_state_transitions.append((start-1, start))

            for i in range(start, end): 
                to_next = (i, i+1)
                allowed_state_transitions.append(to_next)

                if i >min_len-1: #make skip forward connections
                    skip_to_i = (start+min_len-3, i)#((min_len-2,i))
                    allowed_state_transitions.append(skip_to_i) 

        return allowed_state_transitions, allowed_starts, allowed_ends

    def _debug_crf(self, targets):
        '''Check label sequences for incompatibilities with the defined state grammar.'''
        for i in range(targets.shape[0]):

            for j in range(1, targets.shape[1]):
                l = int(targets[i,j].item())
                l_prev = int(targets[i,j-1].item())

                if (l_prev, l) not in self.allowed_transitions:
                    print(f'Found invalid transition from {l_prev} to {l}.')


    
    def _repeat_emissions(self, emissions):
        '''Turn a (batch_size, seq_len, 2) tensor into (batch_size, seq_len, num_states) by repeating the emissions at position 1.'''

        if emissions.shape[-1] == 2:
            emissions_out = torch.zeros(emissions.shape[0], emissions.shape[1], self.num_states, dtype=emissions.dtype, device=emissions.device)    
            emissions_out[:,:,0] = emissions[:,:,0]
            emissions_out[:,:, 1:(self.max_len+1)] = emissions[:,:,1].unsqueeze(-1)
        elif emissions.shape[-1] == 3:
            emissions_out = torch.zeros(emissions.shape[0], emissions.shape[1], self.num_states, dtype=emissions.dtype, device=emissions.device)
            emissions_out[:,:,0] = emissions[:,:,0]
            emissions_out[:,:, 1:] = emissions[:,:,1].unsqueeze(-1)
            emissions_out[:,:, (self.max_len+1):] = emissions[:,:,2].unsqueeze(-1)
        else:
            raise NotImplementedError()
        
        return emissions_out


    def forward(
        self,
        embeddings,
        mask,
        targets=None,
        skip_marginals: bool = False,
        top_k: int = 1,
        decode: bool = True,          # <-- NEW: allows disabling decode during train
        return_probs: bool = True,    # <-- NEW: allows skipping probs computation entirely
    ):
        # embeddings: [B, C, L] (same layout as LSTMCNN expects)
        # mask:       [B, L] (0/1)
        mask = mask.bool()

        if isinstance(self.feature_extractor, CustomTransformerWrapper):
            # this wrapper expects [B, L, C]
            feats_in = embeddings.transpose(1, 2)  # [B, L, C]
            features = self.feature_extractor(feats_in)  # [B, L, F]
        else:
            # LSTMCNN expects [B, C, L]
            features = self.feature_extractor(embeddings, mask)  # [B, L, F]

        emissions = self.features_to_emissions(features)        # [B, L, num_labels]
        emissions = self._repeat_emissions(emissions)           # [B, L, num_states]

        loss = None
        if targets is not None:
            # targets are expected as [B, L] long
            targets = targets.long()
            loss = self.crf(emissions=emissions, tags=targets, mask=mask, reduction="mean") * -1
            if loss.item() > 10000:
                self._debug_crf(targets)

        viterbi_paths = None
        path_probs = None
        if decode:
            viterbi_paths, path_probs = self.crf.decode(emissions=emissions, mask=mask, top_k=top_k)

        probs = None
        if return_probs:
            # If skip_marginals=True, use softmax as a surrogate (matches prior behavior)
            probs = (
                self.crf.compute_marginal_probabilities(emissions, mask)
                if not skip_marginals
                else torch.softmax(emissions, dim=-1)
            )

        if targets is not None:
            return probs, viterbi_paths, loss
        else:
            return probs, viterbi_paths, path_probs


    @staticmethod
    def _esm_embed(sequence:str, device: torch.device, repr_layers: int=33) -> torch.Tensor:


        from esm import pretrained
        esm_model, esm_alphabet = pretrained.load_model_and_alphabet('esm1b_t33_650M_UR50S')
        batch_converter = esm_alphabet.get_batch_converter()
        esm_model.to(device)


        data = [
            ("protein1", sequence),
        ]
        labels, strs, toks = batch_converter(data)

        repr_layers_list = [
            (i + esm_model.num_layers + 1) % (esm_model.num_layers + 1) for i in range(repr_layers)
        ]

        out = None

        toks = toks.to(device)

        minibatch_max_length = toks.size(1)

        tokens_list = []
        end = 0
        while end <= minibatch_max_length:
            start = end
            end = start + 1022
            if end <= minibatch_max_length:
                # we are not on the last one, so make this shorter
                end = end - 300
            tokens = esm_model(toks[:, start:end], repr_layers=repr_layers_list, return_contacts=False)["representations"][repr_layers - 1]
            tokens_list.append(tokens)

        out = torch.cat(tokens_list, dim=1).cpu()

        # set nan to zeros
        out[out!=out] = 0.0

        res = out.transpose(0,1)[1:-1] 
        seq_embedding = res[:,0]

        return seq_embedding

    def predict_from_sequence(self, sequence: str, top_k: int = 5):
        self.eval()
        with torch.no_grad():
            device =  next(self.parameters()).device
            embedding = self._esm_embed(sequence, device)
            embedding = torch.unsqueeze(embedding.permute(1,0), 0)
            mask = torch.unsqueeze(torch.ones(embedding.shape[2]),0)
        

            pos_probs, pos_preds, path_probs = self(embedding, mask, top_k=top_k)

            return pos_probs.squeeze(), pos_preds[0], path_probs[0]

    @staticmethod
    def _make_tag_bitmap(length, start, end, start_state=1, min_len=5, max_len=50):
        '''Make a multi-tag bitmap for the given peptide positions where all other positions are flexible.'''
        with torch.no_grad():
            label = torch.zeros((length, max_len*2+1))

            peptide_length = end-start +1 # inclusive.
            peptide_label = torch.concat(
                [ 
                torch.arange(start_state, start_state+min_len-2),#np.arange(1, 4), # from start to first position with skip connections
                # (end_state -1) - (peptide_length - min_len)
                torch.arange((start_state+max_len-2 - (peptide_length - min_len)), start_state+max_len) #np.arange( 59-(peptide_length-5) ,61) 
                ]
            )

            # set the positions in the matrix to true.
            label[torch.arange(start-1,end), peptide_label] = 1
            label[:start-1,:] = 1
            label[end:, :] = 1

        return label

    def predict_peptide_probability(self, sequence:str, start: int, stop: int):
        '''Computes probability of a peptide given all possible paths.'''
        self.eval()
        with torch.no_grad():
            device =  next(self.parameters()).device
            embedding = self._esm_embed(sequence, device)
            embedding = torch.unsqueeze(embedding.permute(1,0), 0)
            mask = torch.unsqueeze(torch.ones(embedding.shape[2]),0)
        
            features = self.feature_extractor(embedding, mask) # (batch_size, seq_len, feature_dim)
            emissions = self.features_to_emissions(features) # (batch_size, seq_len, num_labels)
            emissions = self._repeat_emissions(emissions) # (batch_size, seq_len, num_states)

            targets = self._make_tag_bitmap(len(sequence), start, stop, start_state=1)
            targets = torch.unsqueeze(targets,0)
            llh_pep= self.crf(emissions = emissions, tag_bitmap=targets.long(), mask = mask.bool(), reduction='none')
            
            targets = self._make_tag_bitmap(len(sequence), start, stop, start_state=51)
            targets = torch.unsqueeze(targets,0)
            llh_pro= self.crf(emissions = emissions, tag_bitmap=targets.long(), mask = mask.bool(), reduction='none')

            return torch.exp(llh_pep[0]).item(), torch.exp(llh_pro[0]).item()


class EmbeddingProjector(nn.Module):
    """Small trainable adapter that projects precomputed residue embeddings before the feature extractor.

    Expects embeddings shaped [B, C, L] (channel-first), and returns [B, C_proj, L].
    """
    def __init__(self, input_size: int, proj_size: int = 512, dropout: float = 0.2):
        super().__init__()
        self.ln = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(input_size, proj_size)
        self.act = nn.GELU()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # [B, C, L] -> [B, L, C]
        x = embeddings.transpose(1, 2)
        x = self.ln(x)
        x = self.dropout(x)
        x = self.proj(x)
        x = self.act(x)
        x = self.dropout(x)
        # [B, L, C_proj] -> [B, C_proj, L]
        return x.transpose(1, 2)


class MultiScaleConv1DBlock(nn.Module):
    """Residual multi-scale Conv1d block for channel-first residue features [B, C, L]."""

    def __init__(
        self,
        channels: int,
        kernels=(3, 7, 15),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.kernels = tuple(int(k) for k in kernels)
        self.branches = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=k, padding=k // 2)
            for k in self.kernels
        ])
        self.merge = nn.Conv1d(channels * len(self.kernels), channels, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        outs = [self.act(conv(x)) for conv in self.branches]
        y = torch.cat(outs, dim=1)     # [B, C*num_scales, L]
        y = self.merge(y)              # [B, C, L]
        y = self.act(y)
        y = self.dropout(y)
        y = y + x                      # residual
        y = self.out_ln(y.transpose(1, 2)).transpose(1, 2)
        return y


class MultiScaleProjectedLSTMCNN(nn.Module):
    """EmbeddingProjector -> MultiScaleConv1DBlock -> existing LSTMCNN backbone."""
    def __init__(
        self,
        input_size: int,
        proj_size: int = 256,
        dropout_projector: float = 0.2,
        multiscale_kernels=(3, 7, 15),
        multiscale_dropout: float = 0.1,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
    ):
        super().__init__()
        self.projector = EmbeddingProjector(
            input_size=input_size,
            proj_size=proj_size,
            dropout=dropout_projector,
        )
        self.multiscale = MultiScaleConv1DBlock(
            channels=proj_size,
            kernels=multiscale_kernels,
            dropout=multiscale_dropout,
        )
        self.backbone = LSTMCNN(
            input_size=proj_size,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        self.biLSTM = self.backbone.biLSTM

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.projector(embeddings)
        x = self.multiscale(x)
        return self.backbone(x, mask)


class LSTMCNNCRFProjectorMultiScale(CRFBaseModel):
    """LSTM-CNN + CRF with trainable projector and multi-scale Conv1d front-end."""
    def __init__(
        self,
        input_size: int = 1280,
        proj_size: int = 256,
        dropout_projector: float = 0.4,
        multiscale_kernels=(3, 7, 15),
        multiscale_dropout: float = 0.1,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
    ) -> None:
        super().__init__(num_labels, num_states)

        self.feature_extractor = MultiScaleProjectedLSTMCNN(
            input_size=input_size,
            proj_size=proj_size,
            dropout_projector=dropout_projector,
            multiscale_kernels=multiscale_kernels,
            multiscale_dropout=multiscale_dropout,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
        )
        self.features_to_emissions = nn.Linear(n_filters * 2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )


class GatedResidualConvMultiScaleProjectedLSTMCNN(nn.Module):
    """GatedResidualConvSplitProjector -> MultiScaleConv1DBlock -> existing LSTMCNN backbone."""
    def __init__(
        self,
        input_size: int,
        seq_input_size: int,
        struct_input_size: int,
        seq_proj_size: int = 256,
        struct_proj_size: int = 32,
        dropout_projector: float = 0.2,
        multiscale_kernels=(3, 7, 15),
        multiscale_dropout: float = 0.1,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        residual_scale: float = 0.1,
        struct_branch_dropout: float = 0.3,
        gate_bias: float = -2.5,
        struct_conv_kernel: int = 5,
    ):
        super().__init__()
        if input_size != seq_input_size + struct_input_size:
            raise ValueError(
                f"input_size ({input_size}) must equal seq_input_size + struct_input_size ({seq_input_size + struct_input_size})"
            )

        self.projector = GatedResidualConvSplitProjector(
            seq_input_size=seq_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            struct_proj_size=struct_proj_size,
            dropout=dropout_projector,
            residual_scale=residual_scale,
            struct_branch_dropout=struct_branch_dropout,
            gate_bias=gate_bias,
            struct_conv_kernel=struct_conv_kernel,
        )
        self.multiscale = MultiScaleConv1DBlock(
            channels=seq_proj_size,
            kernels=multiscale_kernels,
            dropout=multiscale_dropout,
        )
        self.backbone = LSTMCNN(
            input_size=seq_proj_size,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        self.biLSTM = self.backbone.biLSTM

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.projector(embeddings)
        x = self.multiscale(x)
        return self.backbone(x, mask)


class LSTMCNNCRFGated3DiResidualConvMultiScale(CRFBaseModel):
    """LSTM-CNN + CRF with ESM2 main branch, gated 3Di conv residual, and multi-scale Conv1d on the fused main branch."""
    def __init__(
        self,
        input_size: int = 1300,
        seq_input_size: int = 1280,
        struct_input_size: int = 20,
        seq_proj_size: int = 256,
        struct_proj_size: int = 32,
        dropout_projector: float = 0.4,
        multiscale_kernels=(3, 7, 15),
        multiscale_dropout: float = 0.1,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
        residual_scale: float = 0.1,
        struct_branch_dropout: float = 0.3,
        gate_bias: float = -2.5,
        struct_conv_kernel: int = 5,
    ) -> None:
        super().__init__(num_labels, num_states)

        self.feature_extractor = GatedResidualConvMultiScaleProjectedLSTMCNN(
            input_size=input_size,
            seq_input_size=seq_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            struct_proj_size=struct_proj_size,
            dropout_projector=dropout_projector,
            multiscale_kernels=multiscale_kernels,
            multiscale_dropout=multiscale_dropout,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            dropout_conv1=dropout_conv1,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            residual_scale=residual_scale,
            struct_branch_dropout=struct_branch_dropout,
            gate_bias=gate_bias,
            struct_conv_kernel=struct_conv_kernel,
        )
        self.features_to_emissions = nn.Linear(n_filters * 2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )


class TriBranchResidualProjector(nn.Module):
    """
    ESM2 main branch + residue10 residual + optional 3Di residual with Conv1d.

    Input layout:
      [seq | residue | struct]
    where:
      seq     = ESM2
      residue = additional residue-level params
      struct  = optional 3Di channels

    Shapes:
      embeddings: [B, C_total, L]
      output:     [B, seq_proj_size, L]
    """
    def __init__(
        self,
        seq_input_size: int,
        residue_input_size: int,
        struct_input_size: int = 0,
        seq_proj_size: int = 256,
        residue_proj_size: int = 16,
        struct_proj_size: int = 16,
        dropout: float = 0.2,
        residue_residual_scale: float = 0.05,
        struct_residual_scale: float = 0.10,
        residue_branch_dropout: float = 0.2,
        struct_branch_dropout: float = 0.3,
        residue_gate_bias: float = -2.5,
        struct_gate_bias: float = -2.5,
        struct_conv_kernel: int = 5,
    ):
        super().__init__()
        self.seq_input_size = seq_input_size
        self.residue_input_size = residue_input_size
        self.struct_input_size = struct_input_size

        self.residue_residual_scale = residue_residual_scale
        self.struct_residual_scale = struct_residual_scale
        self.residue_branch_dropout = residue_branch_dropout
        self.struct_branch_dropout = struct_branch_dropout

        self.seq_projector = EmbeddingProjector(
            input_size=seq_input_size,
            proj_size=seq_proj_size,
            dropout=dropout,
        )
        self.residue_projector = EmbeddingProjector(
            input_size=residue_input_size,
            proj_size=residue_proj_size,
            dropout=dropout,
        )

        self.residue_to_seq = nn.Conv1d(residue_proj_size, seq_proj_size, kernel_size=1)
        self.residue_gate_ln = nn.LayerNorm(seq_proj_size + residue_proj_size)
        self.residue_gate_dropout = nn.Dropout(dropout)
        self.residue_gate = nn.Linear(seq_proj_size + residue_proj_size, seq_proj_size)

        self.has_struct = struct_input_size > 0
        if self.has_struct:
            pad = struct_conv_kernel // 2
            self.struct_projector = EmbeddingProjector(
                input_size=struct_input_size,
                proj_size=struct_proj_size,
                dropout=dropout,
            )
            self.struct_conv = nn.Conv1d(
                struct_proj_size,
                struct_proj_size,
                kernel_size=struct_conv_kernel,
                padding=pad,
            )
            self.struct_conv_act = nn.GELU()
            self.struct_conv_dropout = nn.Dropout(dropout)

            self.struct_to_seq = nn.Conv1d(struct_proj_size, seq_proj_size, kernel_size=1)
            self.struct_gate_ln = nn.LayerNorm(seq_proj_size + struct_proj_size)
            self.struct_gate_dropout = nn.Dropout(dropout)
            self.struct_gate = nn.Linear(seq_proj_size + struct_proj_size, seq_proj_size)

            nn.init.zeros_(self.struct_gate.weight)
            nn.init.constant_(self.struct_gate.bias, struct_gate_bias)

        self.out_ln = nn.LayerNorm(seq_proj_size)

        nn.init.zeros_(self.residue_gate.weight)
        nn.init.constant_(self.residue_gate.bias, residue_gate_bias)

    def _drop_branch(self, x: torch.Tensor, p: float) -> torch.Tensor:
        if self.training and p > 0:
            keep = (torch.rand(x.size(0), 1, 1, device=x.device) >= p).to(x.dtype)
            x = x * keep
        return x

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        expected = self.seq_input_size + self.residue_input_size + self.struct_input_size
        if embeddings.size(1) != expected:
            raise ValueError(
                f"Expected concatenated embeddings with C={expected} "
                f"(= {self.seq_input_size}+{self.residue_input_size}+{self.struct_input_size}), "
                f"got {embeddings.size(1)}"
            )

        s0 = 0
        s1 = self.seq_input_size
        s2 = s1 + self.residue_input_size
        s3 = s2 + self.struct_input_size

        seq = embeddings[:, s0:s1, :]
        residue = embeddings[:, s1:s2, :]
        struct = embeddings[:, s2:s3, :] if self.has_struct else None

        seq = self.seq_projector(seq)              # [B, D_seq, L]
        residue = self.residue_projector(residue)  # [B, D_res, L]
        residue = self._drop_branch(residue, self.residue_branch_dropout)

        seq_t = seq.transpose(1, 2)                                # [B, L, D_seq]
        residue_t = residue.transpose(1, 2)                        # [B, L, D_res]
        residue_up_t = self.residue_to_seq(residue).transpose(1, 2)  # [B, L, D_seq]

        residue_gate_in = torch.cat([seq_t, residue_t], dim=-1)
        residue_gate = torch.sigmoid(
            self.residue_gate(
                self.residue_gate_dropout(
                    self.residue_gate_ln(residue_gate_in)
                )
            )
        )

        fused_t = seq_t + self.residue_residual_scale * residue_gate * residue_up_t

        if self.has_struct:
            struct = self.struct_projector(struct)     # [B, D_str, L]
            struct = self.struct_conv(struct)
            struct = self.struct_conv_act(struct)
            struct = self.struct_conv_dropout(struct)
            struct = self._drop_branch(struct, self.struct_branch_dropout)

            struct_t = struct.transpose(1, 2)                          # [B, L, D_str]
            struct_up_t = self.struct_to_seq(struct).transpose(1, 2)  # [B, L, D_seq]

            struct_gate_in = torch.cat([fused_t, struct_t], dim=-1)
            struct_gate = torch.sigmoid(
                self.struct_gate(
                    self.struct_gate_dropout(
                        self.struct_gate_ln(struct_gate_in)
                    )
                )
            )

            fused_t = fused_t + self.struct_residual_scale * struct_gate * struct_up_t

        fused_t = self.out_ln(fused_t)
        return fused_t.transpose(1, 2)  # [B, D_seq, L]


class TriBranchProjectedLSTMCNN(nn.Module):
    """TriBranchResidualProjector + existing LSTMCNN backbone."""
    def __init__(
        self,
        input_size: int,
        seq_input_size: int,
        residue_input_size: int,
        struct_input_size: int = 0,
        seq_proj_size: int = 256,
        residue_proj_size: int = 16,
        struct_proj_size: int = 16,
        dropout_projector: float = 0.2,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        residue_residual_scale: float = 0.05,
        struct_residual_scale: float = 0.10,
        residue_branch_dropout: float = 0.2,
        struct_branch_dropout: float = 0.3,
        residue_gate_bias: float = -2.5,
        struct_gate_bias: float = -2.5,
        struct_conv_kernel: int = 5,
    ):
        super().__init__()
        expected = seq_input_size + residue_input_size + struct_input_size
        if input_size != expected:
            raise ValueError(
                f"input_size ({input_size}) must equal "
                f"seq_input_size + residue_input_size + struct_input_size ({expected})"
            )

        self.projector = TriBranchResidualProjector(
            seq_input_size=seq_input_size,
            residue_input_size=residue_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            residue_proj_size=residue_proj_size,
            struct_proj_size=struct_proj_size,
            dropout=dropout_projector,
            residue_residual_scale=residue_residual_scale,
            struct_residual_scale=struct_residual_scale,
            residue_branch_dropout=residue_branch_dropout,
            struct_branch_dropout=struct_branch_dropout,
            residue_gate_bias=residue_gate_bias,
            struct_gate_bias=struct_gate_bias,
            struct_conv_kernel=struct_conv_kernel,
        )

        self.backbone = LSTMCNN(
            input_size=seq_proj_size,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        self.biLSTM = self.backbone.biLSTM

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.projector(embeddings)
        return self.backbone(x, mask)


class LSTMCNNCRFTriBranchResidual(CRFBaseModel):
    """
    LSTM-CNN + CRF with:
      - ESM2 main branch
      - residue10 gated residual branch
      - optional 3Di gated residual branch with Conv1d
    """
    def __init__(
        self,
        input_size: int = 1290,
        seq_input_size: int = 1280,
        residue_input_size: int = 10,
        struct_input_size: int = 0,
        seq_proj_size: int = 256,
        residue_proj_size: int = 16,
        struct_proj_size: int = 16,
        dropout_projector: float = 0.4,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
        residue_residual_scale: float = 0.05,
        struct_residual_scale: float = 0.10,
        residue_branch_dropout: float = 0.2,
        struct_branch_dropout: float = 0.3,
        residue_gate_bias: float = -2.5,
        struct_gate_bias: float = -2.5,
        struct_conv_kernel: int = 5,
    ) -> None:
        super().__init__(num_labels, num_states)

        self.feature_extractor = TriBranchProjectedLSTMCNN(
            input_size=input_size,
            seq_input_size=seq_input_size,
            residue_input_size=residue_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            residue_proj_size=residue_proj_size,
            struct_proj_size=struct_proj_size,
            dropout_projector=dropout_projector,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            dropout_conv1=dropout_conv1,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            residue_residual_scale=residue_residual_scale,
            struct_residual_scale=struct_residual_scale,
            residue_branch_dropout=residue_branch_dropout,
            struct_branch_dropout=struct_branch_dropout,
            residue_gate_bias=residue_gate_bias,
            struct_gate_bias=struct_gate_bias,
            struct_conv_kernel=struct_conv_kernel,
        )
        self.features_to_emissions = nn.Linear(n_filters * 2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )


class ProjectedLSTMCNN(nn.Module):
    """EmbeddingProjector + existing LSTMCNN feature extractor.

    Keeps the same forward signature as LSTMCNN: (embeddings [B,C,L], mask [B,L]) -> features [B,L,F].
    """
    def __init__(
        self,
        input_size: int,
        proj_size: int = 512,
        dropout_projector: float = 0.2,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
    ):
        super().__init__()
        self.projector = EmbeddingProjector(input_size=input_size, proj_size=proj_size, dropout=dropout_projector)
        self.backbone = LSTMCNN(
            input_size=proj_size,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        # Expose biLSTM so existing training code can still call flatten_parameters()
        self.biLSTM = self.backbone.biLSTM

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.projector(embeddings)
        return self.backbone(x, mask)



class SplitEmbeddingProjector(nn.Module):
    """Project sequence and structural channels separately, then concatenate them back.

    Expects concatenated embeddings shaped [B, C_total, L] where
    C_total = seq_input_size + struct_input_size and the layout is [seq | struct].
    """
    def __init__(
        self,
        seq_input_size: int,
        struct_input_size: int,
        seq_proj_size: int = 256,
        struct_proj_size: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_input_size = seq_input_size
        self.struct_input_size = struct_input_size
        self.seq_projector = EmbeddingProjector(
            input_size=seq_input_size,
            proj_size=seq_proj_size,
            dropout=dropout,
        )
        self.struct_projector = EmbeddingProjector(
            input_size=struct_input_size,
            proj_size=struct_proj_size,
            dropout=dropout,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        expected = self.seq_input_size + self.struct_input_size
        if embeddings.size(1) != expected:
            raise ValueError(
                f"Expected concatenated embeddings with C={expected} (= {self.seq_input_size}+{self.struct_input_size}), got {embeddings.size(1)}"
            )

        seq = embeddings[:, : self.seq_input_size, :]
        struct = embeddings[:, self.seq_input_size : expected, :]

        seq = self.seq_projector(seq)
        struct = self.struct_projector(struct)
        return torch.cat([seq, struct], dim=1)


class SplitProjectedLSTMCNN(nn.Module):
    """SplitEmbeddingProjector + existing LSTMCNN feature extractor.

    Use this when you store one concatenated per-residue tensor [ESM | 3Di-onehot]
    and want to keep the dataset unchanged.
    """
    def __init__(
        self,
        input_size: int,
        seq_input_size: int,
        struct_input_size: int,
        seq_proj_size: int = 256,
        struct_proj_size: int = 64,
        dropout_projector: float = 0.2,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
    ):
        super().__init__()
        if input_size != seq_input_size + struct_input_size:
            raise ValueError(
                f"input_size ({input_size}) must equal seq_input_size + struct_input_size ({seq_input_size + struct_input_size})"
            )

        self.projector = SplitEmbeddingProjector(
            seq_input_size=seq_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            struct_proj_size=struct_proj_size,
            dropout=dropout_projector,
        )
        self.backbone = LSTMCNN(
            input_size=seq_proj_size + struct_proj_size,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        self.biLSTM = self.backbone.biLSTM

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.projector(embeddings)
        return self.backbone(x, mask)


class LSTMCNNCRFSplitProjector(CRFBaseModel):
    """LSTM-CNN + CRF with separate trainable projectors for sequence and structural channels.

    Input layout must be a single concatenated tensor [seq | struct] with shape [B, C, L].
    This keeps the dataset unchanged while allowing different compression for each branch.
    """
    def __init__(
        self,
        input_size: int = 1300,
        seq_input_size: int = 1280,
        struct_input_size: int = 20,
        seq_proj_size: int = 256,
        struct_proj_size: int = 64,
        dropout_projector: float = 0.2,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
    ) -> None:
        super().__init__(num_labels, num_states)

        self.feature_extractor = SplitProjectedLSTMCNN(
            input_size=input_size,
            seq_input_size=seq_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            struct_proj_size=struct_proj_size,
            dropout_projector=dropout_projector,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            dropout_conv1=dropout_conv1,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
        )
        self.features_to_emissions = nn.Linear(n_filters * 2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )


class GatedResidualSplitProjector(nn.Module):
    """Project sequence and structural channels separately, then add a gated structural residual to the sequence branch.

    Input layout is one concatenated tensor [seq | struct] with shape [B, C_total, L].
    Output shape is [B, seq_proj_size, L], so ESM2 remains the main branch and
    3Di only injects a controlled residual signal.
    """
    def __init__(
        self,
        seq_input_size: int,
        struct_input_size: int,
        seq_proj_size: int = 256,
        struct_proj_size: int = 32,
        dropout: float = 0.2,
        residual_scale: float = 0.2,
        struct_branch_dropout: float = 0.0,
        gate_bias: float = -2.0,
    ):
        super().__init__()
        self.seq_input_size = seq_input_size
        self.struct_input_size = struct_input_size
        self.residual_scale = residual_scale
        self.struct_branch_dropout = struct_branch_dropout

        self.seq_projector = EmbeddingProjector(
            input_size=seq_input_size,
            proj_size=seq_proj_size,
            dropout=dropout,
        )
        self.struct_projector = EmbeddingProjector(
            input_size=struct_input_size,
            proj_size=struct_proj_size,
            dropout=dropout,
        )

        self.struct_to_seq = nn.Conv1d(struct_proj_size, seq_proj_size, kernel_size=1)
        self.gate_ln = nn.LayerNorm(seq_proj_size + struct_proj_size)
        self.gate_dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(seq_proj_size + struct_proj_size, seq_proj_size)
        self.out_ln = nn.LayerNorm(seq_proj_size)

        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        expected = self.seq_input_size + self.struct_input_size
        if embeddings.size(1) != expected:
            raise ValueError(
                f"Expected concatenated embeddings with C={expected} (= {self.seq_input_size}+{self.struct_input_size}), got {embeddings.size(1)}"
            )

        seq = embeddings[:, : self.seq_input_size, :]
        struct = embeddings[:, self.seq_input_size : expected, :]

        seq = self.seq_projector(seq)
        struct = self.struct_projector(struct)

        if self.training and self.struct_branch_dropout > 0:
            keep = (torch.rand(struct.size(0), 1, 1, device=struct.device) >= self.struct_branch_dropout).to(struct.dtype)
            struct = struct * keep

        seq_t = seq.transpose(1, 2)
        struct_t = struct.transpose(1, 2)
        struct_up_t = self.struct_to_seq(struct).transpose(1, 2)

        gate_in = torch.cat([seq_t, struct_t], dim=-1)
        gate = torch.sigmoid(self.gate(self.gate_dropout(self.gate_ln(gate_in))))

        fused = self.out_ln(seq_t + self.residual_scale * gate * struct_up_t)
        return fused.transpose(1, 2)


class GatedResidualConvSplitProjector(nn.Module):
    """Project ESM2 and 3Di separately, run a small Conv1d over the structural branch,
    then add a gated structural residual to the ESM2 branch.

    Input layout: [B, C_total, L] where C_total = seq_input_size + struct_input_size
    and channels are concatenated as [seq | struct].
    """
    def __init__(
        self,
        seq_input_size: int,
        struct_input_size: int,
        seq_proj_size: int = 256,
        struct_proj_size: int = 32,
        dropout: float = 0.2,
        residual_scale: float = 0.1,
        struct_branch_dropout: float = 0.3,
        gate_bias: float = -2.5,
        struct_conv_kernel: int = 5,
        seq_only: bool = False,
    ):
        super().__init__()
        self.seq_input_size = seq_input_size
        self.struct_input_size = struct_input_size
        self.residual_scale = residual_scale
        self.struct_branch_dropout = struct_branch_dropout
        self.seq_only = seq_only

        self.seq_projector = EmbeddingProjector(
            input_size=seq_input_size,
            proj_size=seq_proj_size,
            dropout=dropout,
        )
        self.out_ln = nn.LayerNorm(seq_proj_size)

        # Structural branch — built and used only when fusing 3Di. seq_only=True
        # isolates the sequence adapter (LayerNorm+reproject) with no struct path.
        if not seq_only:
            self.struct_projector = EmbeddingProjector(
                input_size=struct_input_size,
                proj_size=struct_proj_size,
                dropout=dropout,
            )

            pad = struct_conv_kernel // 2
            self.struct_conv = nn.Conv1d(
                struct_proj_size,
                struct_proj_size,
                kernel_size=struct_conv_kernel,
                padding=pad,
            )
            self.struct_conv_act = nn.GELU()
            self.struct_conv_dropout = nn.Dropout(dropout)

            self.struct_to_seq = nn.Conv1d(struct_proj_size, seq_proj_size, kernel_size=1)

            self.gate_ln = nn.LayerNorm(seq_proj_size + struct_proj_size)
            self.gate_dropout = nn.Dropout(dropout)
            self.gate = nn.Linear(seq_proj_size + struct_proj_size, seq_proj_size)

            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.seq_only:
            # No structural branch exists in this mode (never built in __init__), so
            # accept plain, unconcatenated sequence embeddings -- no need to carry a
            # (disabled) structural channel through the dataset just to satisfy a
            # shape check.
            if embeddings.size(1) != self.seq_input_size:
                raise ValueError(
                    f"Expected sequence-only embeddings with C={self.seq_input_size}, got {embeddings.size(1)}"
                )
            seq = self.seq_projector(embeddings)
            seq_t = seq.transpose(1, 2)              # [B, L, D_seq]
            return self.out_ln(seq_t).transpose(1, 2)  # [B, D_seq, L]

        expected = self.seq_input_size + self.struct_input_size
        if embeddings.size(1) != expected:
            raise ValueError(
                f"Expected concatenated embeddings with C={expected} (= {self.seq_input_size}+{self.struct_input_size}), got {embeddings.size(1)}"
            )

        seq = embeddings[:, : self.seq_input_size, :]
        seq = self.seq_projector(seq)

        struct = embeddings[:, self.seq_input_size : expected, :]
        struct = self.struct_projector(struct)

        # local structural patterns before gating
        struct = self.struct_conv(struct)
        struct = self.struct_conv_act(struct)
        struct = self.struct_conv_dropout(struct)

        if self.training and self.struct_branch_dropout > 0:
            keep = (
                torch.rand(struct.size(0), 1, 1, device=struct.device) >= self.struct_branch_dropout
            ).to(struct.dtype)
            struct = struct * keep

        seq_t = seq.transpose(1, 2)                       # [B, L, D_seq]
        struct_t = struct.transpose(1, 2)                # [B, L, D_str]
        struct_up_t = self.struct_to_seq(struct).transpose(1, 2)  # [B, L, D_seq]

        gate_in = torch.cat([seq_t, struct_t], dim=-1)
        gate = torch.sigmoid(self.gate(self.gate_dropout(self.gate_ln(gate_in))))

        fused = self.out_ln(seq_t + self.residual_scale * gate * struct_up_t)
        return fused.transpose(1, 2)                     # [B, D_seq, L]


class GatedResidualConvProjectedLSTMCNN(nn.Module):
    """Conv-enhanced gated residual projector + existing LSTMCNN backbone."""
    def __init__(
        self,
        input_size: int,
        seq_input_size: int,
        struct_input_size: int,
        seq_proj_size: int = 256,
        struct_proj_size: int = 32,
        dropout_projector: float = 0.2,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        residual_scale: float = 0.1,
        struct_branch_dropout: float = 0.3,
        gate_bias: float = -2.5,
        struct_conv_kernel: int = 5,
        seq_only: bool = False,
    ):
        super().__init__()
        if seq_only:
            if input_size != seq_input_size:
                raise ValueError(
                    f"input_size ({input_size}) must equal seq_input_size ({seq_input_size}) when seq_only=True "
                    "(no structural channel is expected in the input)"
                )
        elif input_size != seq_input_size + struct_input_size:
            raise ValueError(
                f"input_size ({input_size}) must equal seq_input_size + struct_input_size ({seq_input_size + struct_input_size})"
            )

        self.projector = GatedResidualConvSplitProjector(
            seq_input_size=seq_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            struct_proj_size=struct_proj_size,
            dropout=dropout_projector,
            residual_scale=residual_scale,
            struct_branch_dropout=struct_branch_dropout,
            gate_bias=gate_bias,
            struct_conv_kernel=struct_conv_kernel,
            seq_only=seq_only,
        )

        self.backbone = LSTMCNN(
            input_size=seq_proj_size,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        self.biLSTM = self.backbone.biLSTM

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.projector(embeddings)
        return self.backbone(x, mask)


class LSTMCNNCRFGated3DiResidualConv(CRFBaseModel):
    """LSTM-CNN + CRF with ESM2 backbone and gated 3Di residual fusion,
    plus a small Conv1d on the structural branch before gating.
    """
    def __init__(
        self,
        input_size: int = 1300,
        seq_input_size: int = 1280,
        struct_input_size: int = 20,
        seq_proj_size: int = 256,
        struct_proj_size: int = 32,
        dropout_projector: float = 0.4,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
        residual_scale: float = 0.1,
        struct_branch_dropout: float = 0.3,
        gate_bias: float = -2.5,
        struct_conv_kernel: int = 5,
    ) -> None:
        super().__init__(num_labels, num_states)

        self.feature_extractor = GatedResidualConvProjectedLSTMCNN(
            input_size=input_size,
            seq_input_size=seq_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            struct_proj_size=struct_proj_size,
            dropout_projector=dropout_projector,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            dropout_conv1=dropout_conv1,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            residual_scale=residual_scale,
            struct_branch_dropout=struct_branch_dropout,
            gate_bias=gate_bias,
            struct_conv_kernel=struct_conv_kernel,
        )
        self.features_to_emissions = nn.Linear(n_filters * 2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )
        

class GatedResidualProjectedLSTMCNN(nn.Module):
    """GatedResidualSplitProjector + existing LSTMCNN feature extractor.

    Keeps ESM2 as the main branch and injects 3Di as a small gated residual before the backbone.
    """
    def __init__(
        self,
        input_size: int,
        seq_input_size: int,
        struct_input_size: int,
        seq_proj_size: int = 256,
        struct_proj_size: int = 32,
        dropout_projector: float = 0.2,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        residual_scale: float = 0.2,
        struct_branch_dropout: float = 0.0,
        gate_bias: float = -2.0,
    ):
        super().__init__()
        if input_size != seq_input_size + struct_input_size:
            raise ValueError(
                f"input_size ({input_size}) must equal seq_input_size + struct_input_size ({seq_input_size + struct_input_size})"
            )

        self.projector = GatedResidualSplitProjector(
            seq_input_size=seq_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            struct_proj_size=struct_proj_size,
            dropout=dropout_projector,
            residual_scale=residual_scale,
            struct_branch_dropout=struct_branch_dropout,
            gate_bias=gate_bias,
        )
        self.backbone = LSTMCNN(
            input_size=seq_proj_size,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        self.biLSTM = self.backbone.biLSTM

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.projector(embeddings)
        return self.backbone(x, mask)


class LSTMCNNCRFGated3DiResidual(CRFBaseModel):
    """LSTM-CNN + CRF with ESM2 backbone and gated 3Di residual fusion.

    Input layout must be a single concatenated tensor [seq | struct] with shape [B, C, L].
    The model projects ESM2 and 3Di separately, then adds a learned gated structural residual
    to the projected ESM2 branch before the existing LSTMCNN feature extractor.
    """
    def __init__(
        self,
        input_size: int = 1300,
        seq_input_size: int = 1280,
        struct_input_size: int = 20,
        seq_proj_size: int = 256,
        struct_proj_size: int = 32,
        dropout_projector: float = 0.2,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
        residual_scale: float = 0.2,
        struct_branch_dropout: float = 0.0,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__(num_labels, num_states)

        self.feature_extractor = GatedResidualProjectedLSTMCNN(
            input_size=input_size,
            seq_input_size=seq_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            struct_proj_size=struct_proj_size,
            dropout_projector=dropout_projector,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            dropout_conv1=dropout_conv1,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            residual_scale=residual_scale,
            struct_branch_dropout=struct_branch_dropout,
            gate_bias=gate_bias,
        )
        self.features_to_emissions = nn.Linear(n_filters * 2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )


class AhoEmissionHead(nn.Module):
    """Small late-fusion head that maps sparse Aho features to per-label emission biases.

    Input:  aho features [B, C_aho, L]
    Output: emission bias [B, L, num_labels]

    The final linear layer can be zero-initialized so the model starts as the plain
    ESM2/LSTMCNN/CRF model and learns to use Aho only if it helps.
    """
    def __init__(
        self,
        aho_input_size: int,
        num_labels: int,
        hidden_size: int = 0,
        dropout: float = 0.1,
        zero_init: bool = True,
    ):
        super().__init__()
        self.aho_input_size = aho_input_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size

        if aho_input_size <= 0:
            raise ValueError(f"aho_input_size must be positive, got {aho_input_size}")

        if hidden_size and hidden_size > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(aho_input_size),
                nn.Dropout(dropout),
                nn.Linear(aho_input_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_labels),
            )
            final = self.net[-1]
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(aho_input_size),
                nn.Dropout(dropout),
                nn.Linear(aho_input_size, num_labels),
            )
            final = self.net[-1]

        if zero_init:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, aho: torch.Tensor) -> torch.Tensor:
        # [B, C_aho, L] -> [B, L, C_aho]
        aho_t = aho.transpose(1, 2)
        return self.net(aho_t)


class LSTMCNNCRFAhoEmissionFusion(CRFBaseModel):
    """ESM2 LSTM-CNN-CRF with late Aho-to-emission fusion.

    Input layout is a single concatenated tensor [seq | aho] with shape [B, C_total, L].
    The sequence branch is the original LSTMCNN backbone over ESM2 channels only.
    The Aho branch produces additive per-label emission biases before the multistate CRF.

    This keeps the ESM2 representation path intact and prevents sparse Aho features from
    being mixed into the dense ESM2 features before the CNN/LSTM backbone.
    """
    def __init__(
        self,
        input_size: int = 1356,
        seq_input_size: int = 1280,
        aho_input_size: int = 76,
        aho_hidden_size: int = 0,
        aho_dropout: float = 0.1,
        aho_scale: float = 1.0,
        aho_branch_dropout: float = 0.0,
        aho_zero_init: bool = True,
        aho_none_scale: float = 1.0,
        aho_pep_scale: float = 1.0,
        aho_propep_scale: float = 1.0,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
    ) -> None:
        super().__init__(num_labels, num_states)

        if input_size != seq_input_size + aho_input_size:
            raise ValueError(
                f"input_size ({input_size}) must equal seq_input_size + aho_input_size "
                f"({seq_input_size + aho_input_size})"
            )

        self.seq_input_size = seq_input_size
        self.aho_input_size = aho_input_size
        self.aho_scale = float(aho_scale)
        self.aho_branch_dropout = float(aho_branch_dropout)

        # Keep this attribute name for compatibility with train_loop_crf.py flatten_parameters().
        self.feature_extractor = LSTMCNN(
            input_size=seq_input_size,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        self.features_to_emissions = nn.Linear(n_filters * 2, num_labels)
        self.aho_to_emissions = AhoEmissionHead(
            aho_input_size=aho_input_size,
            num_labels=num_labels,
            hidden_size=aho_hidden_size,
            dropout=aho_dropout,
            zero_init=aho_zero_init,
        )
        self.register_buffer(
            "aho_label_scales",
            build_aho_label_scale_tensor(
                num_labels=num_labels,
                none_scale=aho_none_scale,
                pep_scale=aho_pep_scale,
                propep_scale=aho_propep_scale,
            ),
        )
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.allowed_transitions = allowed_transitions
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )

    def forward(
        self,
        embeddings,
        mask,
        targets=None,
        skip_marginals: bool = False,
        top_k: int = 1,
        decode: bool = True,
        return_probs: bool = True,
    ):
        # embeddings: [B, C_total, L], layout [ESM2 | Aho]
        mask = mask.bool()
        expected = self.seq_input_size + self.aho_input_size
        if embeddings.size(1) != expected:
            raise ValueError(
                f"Expected embeddings with C={expected} (= {self.seq_input_size}+{self.aho_input_size}), "
                f"got {embeddings.size(1)}"
            )

        seq = embeddings[:, : self.seq_input_size, :]
        aho = embeddings[:, self.seq_input_size : expected, :]

        features = self.feature_extractor(seq, mask)                 # [B, L, F]
        base_emissions = self.features_to_emissions(features)        # [B, L, num_labels]

        if self.training and self.aho_branch_dropout > 0:
            keep = (torch.rand(aho.size(0), 1, 1, device=aho.device) >= self.aho_branch_dropout).to(aho.dtype)
            aho = aho * keep

        aho_emissions = self.aho_to_emissions(aho)                   # [B, L, num_labels]
        aho_emissions = aho_emissions * self.aho_label_scales.to(dtype=aho_emissions.dtype)
        emissions = base_emissions + self.aho_scale * aho_emissions
        emissions = self._repeat_emissions(emissions)                # [B, L, num_states]

        loss = None
        if targets is not None:
            targets = targets.long()
            loss = self.crf(emissions=emissions, tags=targets, mask=mask, reduction="mean") * -1
            if loss.item() > 10000:
                self._debug_crf(targets)

        viterbi_paths = None
        path_probs = None
        if decode:
            viterbi_paths, path_probs = self.crf.decode(emissions=emissions, mask=mask, top_k=top_k)

        probs = None
        if return_probs:
            probs = (
                self.crf.compute_marginal_probabilities(emissions, mask)
                if not skip_marginals
                else torch.softmax(emissions, dim=-1)
            )

        if targets is not None:
            return probs, viterbi_paths, loss
        else:
            return probs, viterbi_paths, path_probs


class AhoFeatureEncoder(nn.Module):
    """Encode sparse Aho features per residue before fusing with neural context.

    Input:  [B, C_aho, L]
    Output: [B, L, C_out]

    If hidden_size <= 0, this is only LayerNorm + Dropout over raw Aho features.
    If hidden_size > 0, it projects Aho features to hidden_size with GELU.
    """
    def __init__(self, aho_input_size: int, hidden_size: int = 0, dropout: float = 0.1):
        super().__init__()
        self.aho_input_size = int(aho_input_size)
        self.hidden_size = int(hidden_size)
        if self.aho_input_size <= 0:
            raise ValueError(f"aho_input_size must be positive, got {aho_input_size}")

        if self.hidden_size > 0:
            self.out_size = self.hidden_size
            self.net = nn.Sequential(
                nn.LayerNorm(self.aho_input_size),
                nn.Dropout(dropout),
                nn.Linear(self.aho_input_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.out_size = self.aho_input_size
            self.net = nn.Sequential(
                nn.LayerNorm(self.aho_input_size),
                nn.Dropout(dropout),
            )

    def forward(self, aho: torch.Tensor) -> torch.Tensor:
        # [B, C_aho, L] -> [B, L, C_aho]
        return self.net(aho.transpose(1, 2))


class AhoMidFusionHead(nn.Module):
    """Residual emission head conditioned on both LSTMCNN features and Aho features.

    It returns an additive emission correction. The final layer is zero-initialized by
    default, so the model starts as the plain ESM2/LSTMCNN/CRF path and learns the
    Aho-conditioned correction only if useful.
    """
    def __init__(
        self,
        feature_size: int,
        aho_context_size: int,
        num_labels: int,
        hidden_size: int = 0,
        dropout: float = 0.1,
        zero_init: bool = True,
    ):
        super().__init__()
        self.input_size = int(feature_size) + int(aho_context_size)
        self.num_labels = int(num_labels)
        self.hidden_size = int(hidden_size)

        if self.hidden_size > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(self.input_size),
                nn.Dropout(dropout),
                nn.Linear(self.input_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, self.num_labels),
            )
            final = self.net[-1]
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(self.input_size),
                nn.Dropout(dropout),
                nn.Linear(self.input_size, self.num_labels),
            )
            final = self.net[-1]

        if zero_init:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor, aho_context: torch.Tensor) -> torch.Tensor:
        # features: [B, L, F], aho_context: [B, L, A]
        return self.net(torch.cat([features, aho_context], dim=-1))


class LSTMCNNCRFAhoMidFusion(CRFBaseModel):
    """ESM2 LSTM-CNN-CRF with Aho mid-fusion after neural context extraction.

    Input layout: [ESM2 | Aho] as [B, C_total, L].

    Sequence branch:
        ESM2 -> LSTMCNN -> contextual features h_i

    Aho branch:
        Aho sparse features -> AhoFeatureEncoder -> a_i

    Fusion:
        base_emissions = Linear(h_i)
        mid_bias = MLP([h_i ; a_i])
        emissions = base_emissions + aho_scale * mid_bias

    This lets the Aho correction depend on both the dictionary hit features and the
    neural context, unlike pure late emission fusion where the Aho head only sees Aho.
    """
    def __init__(
        self,
        input_size: int = 1356,
        seq_input_size: int = 1280,
        aho_input_size: int = 76,
        aho_hidden_size: int = 32,
        aho_mid_hidden_size: int = 64,
        aho_dropout: float = 0.1,
        aho_scale: float = 1.0,
        aho_branch_dropout: float = 0.0,
        aho_zero_init: bool = True,
        aho_none_scale: float = 1.0,
        aho_pep_scale: float = 1.0,
        aho_propep_scale: float = 1.0,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
    ) -> None:
        super().__init__(num_labels, num_states)

        if input_size != seq_input_size + aho_input_size:
            raise ValueError(
                f"input_size ({input_size}) must equal seq_input_size + aho_input_size "
                f"({seq_input_size + aho_input_size})"
            )

        self.seq_input_size = int(seq_input_size)
        self.aho_input_size = int(aho_input_size)
        self.aho_scale = float(aho_scale)
        self.aho_branch_dropout = float(aho_branch_dropout)
        self.feature_size = int(n_filters) * 2

        # Keep this attribute name for compatibility with train_loop_crf.py flatten_parameters().
        self.feature_extractor = LSTMCNN(
            input_size=seq_input_size,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        self.features_to_emissions = nn.Linear(self.feature_size, num_labels)

        self.aho_encoder = AhoFeatureEncoder(
            aho_input_size=aho_input_size,
            hidden_size=aho_hidden_size,
            dropout=aho_dropout,
        )
        self.aho_mid_fusion = AhoMidFusionHead(
            feature_size=self.feature_size,
            aho_context_size=self.aho_encoder.out_size,
            num_labels=num_labels,
            hidden_size=aho_mid_hidden_size,
            dropout=aho_dropout,
            zero_init=aho_zero_init,
        )
        self.register_buffer(
            "aho_label_scales",
            build_aho_label_scale_tensor(
                num_labels=num_labels,
                none_scale=aho_none_scale,
                pep_scale=aho_pep_scale,
                propep_scale=aho_propep_scale,
            ),
        )
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.allowed_transitions = allowed_transitions
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )

    def forward(
        self,
        embeddings,
        mask,
        targets=None,
        skip_marginals: bool = False,
        top_k: int = 1,
        decode: bool = True,
        return_probs: bool = True,
    ):
        mask = mask.bool()
        expected = self.seq_input_size + self.aho_input_size
        if embeddings.size(1) != expected:
            raise ValueError(
                f"Expected embeddings with C={expected} (= {self.seq_input_size}+{self.aho_input_size}), "
                f"got {embeddings.size(1)}"
            )

        seq = embeddings[:, : self.seq_input_size, :]
        aho = embeddings[:, self.seq_input_size : expected, :]

        features = self.feature_extractor(seq, mask)           # [B, L, F]
        base_emissions = self.features_to_emissions(features)  # [B, L, num_labels]

        if self.training and self.aho_branch_dropout > 0:
            keep = (torch.rand(aho.size(0), 1, 1, device=aho.device) >= self.aho_branch_dropout).to(aho.dtype)
            aho = aho * keep

        aho_context = self.aho_encoder(aho)                    # [B, L, A]
        aho_bias = self.aho_mid_fusion(features, aho_context)  # [B, L, num_labels]
        aho_bias = aho_bias * self.aho_label_scales.to(dtype=aho_bias.dtype)
        emissions = base_emissions + self.aho_scale * aho_bias
        emissions = self._repeat_emissions(emissions)          # [B, L, num_states]

        loss = None
        if targets is not None:
            targets = targets.long()
            loss = self.crf(emissions=emissions, tags=targets, mask=mask, reduction="mean") * -1
            if loss.item() > 10000:
                self._debug_crf(targets)

        viterbi_paths = None
        path_probs = None
        if decode:
            viterbi_paths, path_probs = self.crf.decode(emissions=emissions, mask=mask, top_k=top_k)

        probs = None
        if return_probs:
            probs = (
                self.crf.compute_marginal_probabilities(emissions, mask)
                if not skip_marginals
                else torch.softmax(emissions, dim=-1)
            )

        if targets is not None:
            return probs, viterbi_paths, loss
        else:
            return probs, viterbi_paths, path_probs


class AhoStateBias(CRFBaseModel):
    pass

class LSTMCNNCRFAhoStateBias(CRFBaseModel):
    """ESM2 LSTM-CNN-CRF with simple Aho state-level boundary bias.

    This is *not* a pairwise transition bias. It leaves the CRF implementation untouched.
    It adds Aho-derived scores after coarse emissions are repeated into multistate CRF
    emissions:

      peptide start feature  -> state 1
      peptide inside feature -> states 1..max_len
      peptide end feature    -> state max_len

    and, for 3-label models:

      propep start feature  -> state max_len+1
      propep inside feature -> states max_len+1..2*max_len
      propep end feature    -> state 2*max_len

    Input layout is [ESM2 | Aho] with shape [B, C_total, L].
    """

    def __init__(
        self,
        input_size: int = 1356,
        seq_input_size: int = 1280,
        aho_input_size: int = 76,
        aho_feature_names_file: str = None,
        aho_state_boundary_feature: str = "binary",
        aho_state_scale: float = 1.0,
        aho_state_branch_dropout: float = 0.0,
        aho_state_bias_trainable: bool = False,
        aho_state_pep_inside_bias: float = 0.0,
        aho_state_pep_start_bias: float = 0.0,
        aho_state_pep_end_bias: float = 0.0,
        aho_state_propep_inside_bias: float = 0.0,
        aho_state_propep_start_bias: float = 0.0,
        aho_state_propep_end_bias: float = 0.0,
        aho_state_pep_to_propep_inside_bias: float = 0.0,
        aho_state_pep_to_propep_start_bias: float = 0.0,
        aho_state_pep_to_propep_end_bias: float = 0.0,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
    ) -> None:
        super().__init__(num_labels, num_states)
        import json

        if input_size != seq_input_size + aho_input_size:
            raise ValueError(
                f"input_size ({input_size}) must equal seq_input_size + aho_input_size "
                f"({seq_input_size + aho_input_size})"
            )
        if aho_feature_names_file is None:
            raise ValueError("aho_feature_names_file is required for lstmcnncrf_aho_state_bias")

        self.seq_input_size = int(seq_input_size)
        self.aho_input_size = int(aho_input_size)
        self.aho_state_scale = float(aho_state_scale)
        self.aho_state_branch_dropout = float(aho_state_branch_dropout)
        self.aho_state_boundary_feature = str(aho_state_boundary_feature)

        with open(aho_feature_names_file) as f:
            feature_names = json.load(f)
        self.aho_feature_names = list(feature_names)
        if len(self.aho_feature_names) != self.aho_input_size:
            raise ValueError(
                f"feature_names length ({len(self.aho_feature_names)}) != aho_input_size ({self.aho_input_size})"
            )
        name_to_idx = {name: i for i, name in enumerate(self.aho_feature_names)}

        def idx(name: str) -> int:
            return int(name_to_idx[name]) if name in name_to_idx else -1

        def boundary_names(label: str):
            if self.aho_state_boundary_feature == "binary":
                return f"{label}.start", f"{label}.inside", f"{label}.end"
            if self.aho_state_boundary_feature == "decay":
                return f"{label}.start_decay", f"{label}.inside", f"{label}.end_decay"
            if self.aho_state_boundary_feature == "window":
                return f"{label}.start_window3", f"{label}.inside", f"{label}.end_window3"
            raise ValueError(
                f"Unknown aho_state_boundary_feature={self.aho_state_boundary_feature}; "
                "expected one of binary, decay, window"
            )

        pep_start, pep_inside, pep_end = boundary_names("pep")
        pro_start, pro_inside, pro_end = boundary_names("propep")
        indices = torch.tensor([
            idx(pep_start), idx(pep_inside), idx(pep_end),
            idx(pro_start), idx(pro_inside), idx(pro_end),
        ], dtype=torch.long)
        self.register_buffer("aho_state_feature_indices", indices)

        init_biases = torch.tensor([
            aho_state_pep_start_bias,
            aho_state_pep_inside_bias,
            aho_state_pep_end_bias,
            aho_state_propep_start_bias,
            aho_state_propep_inside_bias,
            aho_state_propep_end_bias,
        ], dtype=torch.float32)
        if aho_state_bias_trainable:
            self.aho_state_biases = nn.Parameter(init_biases)
        else:
            self.register_buffer("aho_state_biases", init_biases)

        cross_biases = torch.tensor([
            aho_state_pep_to_propep_start_bias,
            aho_state_pep_to_propep_inside_bias,
            aho_state_pep_to_propep_end_bias,
        ], dtype=torch.float32)
        if aho_state_bias_trainable:
            self.aho_state_cross_biases = nn.Parameter(cross_biases)
        else:
            self.register_buffer("aho_state_cross_biases", cross_biases)

        self.feature_extractor = LSTMCNN(
            input_size=seq_input_size,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        self.features_to_emissions = nn.Linear(n_filters * 2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.allowed_transitions = allowed_transitions
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )

    def _gather_aho_state_features(self, aho: torch.Tensor) -> torch.Tensor:
        # aho: [B, C, L]. Return [B, L, 6] in order:
        # pep_start, pep_inside, pep_end, propep_start, propep_inside, propep_end.
        B, C, L = aho.shape
        feats = []
        for raw_idx in self.aho_state_feature_indices.tolist():
            if raw_idx < 0:
                feats.append(torch.zeros(B, L, dtype=aho.dtype, device=aho.device))
            else:
                feats.append(aho[:, raw_idx, :])
        return torch.stack(feats, dim=-1)

    def _make_state_bias(self, aho: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # Return [B, L, num_states].
        if self.training and self.aho_state_branch_dropout > 0:
            keep = (torch.rand(aho.size(0), 1, 1, device=aho.device) >= self.aho_state_branch_dropout).to(aho.dtype)
            aho = aho * keep

        raw_values = self._gather_aho_state_features(aho).to(dtype=dtype)  # [B, L, 6]
        weights = self.aho_state_biases.to(device=aho.device, dtype=dtype).view(1, 1, 6)
        values = raw_values * weights * self.aho_state_scale

        B, L, _ = values.shape
        state_bias = torch.zeros(B, L, self.num_states, dtype=dtype, device=aho.device)

        # Peptide branch: state 1..max_len.
        pep_start = values[:, :, 0]
        pep_inside = values[:, :, 1]
        pep_end = values[:, :, 2]
        if self.max_len < self.num_states:
            state_bias[:, :, 1:(self.max_len + 1)] += pep_inside.unsqueeze(-1)
            state_bias[:, :, 1] += pep_start
            state_bias[:, :, self.max_len] += pep_end

        # Propeptide branch only exists for 3-label / 101-state setup.
        if self.num_states > self.max_len + 1:
            pro_start_state = self.max_len + 1
            pro_end_state = min(2 * self.max_len, self.num_states - 1)
            pro_start = values[:, :, 3]
            pro_inside = values[:, :, 4]
            pro_end = values[:, :, 5]
            state_bias[:, :, pro_start_state:(pro_end_state + 1)] += pro_inside.unsqueeze(-1)
            state_bias[:, :, pro_start_state] += pro_start
            state_bias[:, :, pro_end_state] += pro_end

            # Optional cross-bias: mature peptide hit can also be weak evidence that
            # this area lies inside a broader propeptide. Defaults are zero.
            cross_w = self.aho_state_cross_biases.to(device=aho.device, dtype=dtype).view(1, 1, 3)
            cross = raw_values[:, :, 0:3] * cross_w * self.aho_state_scale
            cross_start = cross[:, :, 0]
            cross_inside = cross[:, :, 1]
            cross_end = cross[:, :, 2]
            state_bias[:, :, pro_start_state:(pro_end_state + 1)] += cross_inside.unsqueeze(-1)
            state_bias[:, :, pro_start_state] += cross_start
            state_bias[:, :, pro_end_state] += cross_end

        return state_bias

    def forward(
        self,
        embeddings,
        mask,
        targets=None,
        skip_marginals: bool = False,
        top_k: int = 1,
        decode: bool = True,
        return_probs: bool = True,
    ):
        mask = mask.bool()
        expected = self.seq_input_size + self.aho_input_size
        if embeddings.size(1) != expected:
            raise ValueError(
                f"Expected embeddings with C={expected} (= {self.seq_input_size}+{self.aho_input_size}), "
                f"got {embeddings.size(1)}"
            )

        seq = embeddings[:, : self.seq_input_size, :]
        aho = embeddings[:, self.seq_input_size : expected, :]

        features = self.feature_extractor(seq, mask)
        emissions = self.features_to_emissions(features)
        emissions = self._repeat_emissions(emissions)
        emissions = emissions + self._make_state_bias(aho, dtype=emissions.dtype)

        loss = None
        if targets is not None:
            targets = targets.long()
            loss = self.crf(emissions=emissions, tags=targets, mask=mask, reduction="mean") * -1
            if loss.item() > 10000:
                self._debug_crf(targets)

        viterbi_paths = None
        path_probs = None
        if decode:
            viterbi_paths, path_probs = self.crf.decode(emissions=emissions, mask=mask, top_k=top_k)

        probs = None
        if return_probs:
            probs = (
                self.crf.compute_marginal_probabilities(emissions, mask)
                if not skip_marginals
                else torch.softmax(emissions, dim=-1)
            )

        if targets is not None:
            return probs, viterbi_paths, loss
        else:
            return probs, viterbi_paths, path_probs


class BoundaryStateEmissionHead(nn.Module):
    """Predict state-level boundary emission biases from contextual residue features.

    Per positive branch (peptide, and propeptide for 3-label) the head emits a window
    of ``2*window+1`` channels, ordered:
      [start_1 .. start_W, inside, end_1 .. end_W]
    where start_k targets the k-th state from the segment start (states 1,2,..,W) and
    end_k the k-th state from the segment end (states max_len, max_len-1, ..). These
    near-end states are exactly the residues anchored by the telescoping grammar
    (first W and last W residues = the cleavage-motif window). window=1 reproduces the
    original {start, inside, end} head exactly (out_size 3 / 6), so old checkpoints load.

    The final layer is zero-initialized by default, so the model starts as the plain
    coarse-emission CRF and learns boundary specialization only if it helps.
    """
    def __init__(
        self,
        feature_size: int,
        num_labels: int,
        hidden_size: int = 64,
        dropout: float = 0.1,
        zero_init: bool = True,
        window: int = 1,
    ):
        super().__init__()
        if num_labels not in (2, 3):
            raise ValueError(f"Unsupported num_labels={num_labels}; expected 2 or 3")
        if int(window) < 1:
            raise ValueError(f"boundary window must be >= 1, got {window}")
        self.num_labels = int(num_labels)
        self.window = int(window)
        self.per_branch = 2 * self.window + 1
        n_branches = 1 if num_labels == 2 else 2
        self.out_size = self.per_branch * n_branches
        hidden_size = int(hidden_size)

        if hidden_size > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(feature_size),
                nn.Dropout(dropout),
                nn.Linear(feature_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, self.out_size),
            )
            final = self.net[-1]
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(feature_size),
                nn.Dropout(dropout),
                nn.Linear(feature_size, self.out_size),
            )
            final = self.net[-1]

        if zero_init:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, L, F] -> [B, L, 3 or 6]
        return self.net(features)


class BondBoundaryHead(nn.Module):
    """Predict one logit per residue bond from adjacent contextual features.

    For bond i between residues i and i+1 the input is:
      [h_i, h_{i+1}, |h_{i+1} - h_i|]
    """
    def __init__(
        self,
        feature_size: int,
        hidden_size: int = 64,
        dropout: float = 0.1,
        zero_init: bool = False,
    ):
        super().__init__()
        hidden_size = int(hidden_size)
        in_size = int(feature_size) * 3
        if hidden_size > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(in_size),
                nn.Dropout(dropout),
                nn.Linear(in_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
            final = self.net[-1]
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(in_size),
                nn.Dropout(dropout),
                nn.Linear(in_size, 1),
            )
            final = self.net[-1]

        if zero_init:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, L, F] -> logits [B, L-1]
        left = features[:, :-1, :]
        right = features[:, 1:, :]
        x = torch.cat([left, right, torch.abs(right - left)], dim=-1)
        return self.net(x).squeeze(-1)


class LSTMCNNCRFBoundaryBondLoss(CRFBaseModel):
    """LSTM-CNN + multistate CRF with learned boundary-aware state emissions.

    Compared with plain LSTMCNNCRF, the model keeps the coarse emissions
    None / Peptide / Propeptide, repeats them into CRF states, and then adds
    a learned state bias:

      pep_start  -> peptide start state
      pep_inside -> all peptide states
      pep_end    -> peptide end state

    and similarly for propeptide in 3-label models. An auxiliary soft bond loss
    can be added on top of the same contextual features. The bond head is not
    used directly at inference; the boundary state head is.
    """
    def __init__(
        self,
        input_size: int = 1280,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
        boundary_state_hidden_size: int = 64,
        boundary_state_dropout: float = 0.1,
        boundary_state_scale: float = 1.0,
        boundary_state_zero_init: bool = True,
        boundary_window: int = 1,
        bond_loss_lambda: float = 0.02,
        bond_soft_window: int = 5,
        bond_soft_tau: float = 1.5,
        bond_soft_mode: str = "exp",
        bond_positive_weight: float = 10.0,
        bond_hidden_size: int = 64,
        bond_dropout: float = 0.1,
        bond_zero_init: bool = False,
        feature_extractor: str = "LSTMCNN",
    ) -> None:
        super().__init__(num_labels, num_states)

        if feature_extractor != "LSTMCNN":
            raise ValueError("LSTMCNNCRFBoundaryBondLoss currently supports only feature_extractor='LSTMCNN'")
        if bond_soft_mode not in ("exp", "gaussian"):
            raise ValueError("bond_soft_mode must be 'exp' or 'gaussian'")

        self.feature_extractor = LSTMCNN(
            input_size=input_size,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        self.feature_size = int(n_filters) * 2
        self.boundary_window = int(boundary_window)
        self.features_to_emissions = nn.Linear(self.feature_size, num_labels)
        self.boundary_to_state = BoundaryStateEmissionHead(
            feature_size=self.feature_size,
            num_labels=num_labels,
            hidden_size=boundary_state_hidden_size,
            dropout=boundary_state_dropout,
            zero_init=boundary_state_zero_init,
            window=self.boundary_window,
        )
        self.bond_head = BondBoundaryHead(
            feature_size=self.feature_size,
            hidden_size=bond_hidden_size,
            dropout=bond_dropout,
            zero_init=bond_zero_init,
        )

        self.boundary_state_scale = float(boundary_state_scale)
        self.bond_loss_lambda = float(bond_loss_lambda)
        self.bond_soft_window = int(bond_soft_window)
        self.bond_soft_tau = float(bond_soft_tau)
        self.bond_soft_mode = str(bond_soft_mode)
        self.bond_positive_weight = float(bond_positive_weight)
        self.num_labels = int(num_labels)
        self.num_states = int(num_states)

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.allowed_transitions = allowed_transitions
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )

        # For optional debugging/logging from outside.
        self.last_crf_loss = None
        self.last_bond_loss = None

    def _add_boundary_state_emissions(self, emissions: torch.Tensor, boundary_logits: torch.Tensor) -> torch.Tensor:
        # emissions: [B, L, num_states], boundary_logits: [B, L, per_branch * n_branches]
        if self.boundary_state_scale == 0:
            return emissions

        b = boundary_logits.to(dtype=emissions.dtype) * self.boundary_state_scale
        W = self.boundary_window
        per = 2 * W + 1  # [start_1..W, inside, end_1..W]

        # (start_state, end_state, channel_offset) per positive branch.
        branches = [(1, self.max_len, 0)]
        if self.num_labels == 3 and self.num_states > self.max_len + 1:
            pro_start_state = self.max_len + 1
            pro_end_state = min(2 * self.max_len, self.num_states - 1)
            branches.append((pro_start_state, pro_end_state, per))

        for s_state, e_state, off in branches:
            inside = b[:, :, off + W]
            emissions[:, :, s_state:(e_state + 1)] = emissions[:, :, s_state:(e_state + 1)] + inside.unsqueeze(-1)
            for k in range(W):
                st = s_state + k                 # k-th state from start: 1,2,..,W
                if st <= e_state:
                    emissions[:, :, st] = emissions[:, :, st] + b[:, :, off + k]
                en = e_state - k                 # k-th state from end: max_len, max_len-1, ..
                if en >= s_state:
                    emissions[:, :, en] = emissions[:, :, en] + b[:, :, off + W + 1 + k]

        return emissions

    def _state_branch_ids(self, tags: torch.Tensor) -> torch.Tensor:
        # tags: [B, L] multistate CRF labels.
        # Return 0=None, 1=Peptide branch, 2=Propeptide branch.
        branch = torch.zeros_like(tags, dtype=torch.long)
        branch[(tags >= 1) & (tags <= self.max_len)] = 1
        if self.num_labels == 3:
            branch[(tags >= self.max_len + 1) & (tags <= 2 * self.max_len)] = 2
        return branch

    def _exact_boundary_targets_from_tags(self, tags: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Return hard targets [B, L-1] and valid bond mask [B, L-1].
        # A boundary is any class change or a reset inside the same positive branch
        # (e.g. peptide_end -> peptide_start for adjacent segments of the same type).
        tags = tags.long()
        valid_bonds = mask[:, :-1].bool() & mask[:, 1:].bool()

        left = tags[:, :-1]
        right = tags[:, 1:]
        left_branch = self._state_branch_ids(left)
        right_branch = self._state_branch_ids(right)

        branch_change = left_branch != right_branch
        same_positive_branch = (left_branch == right_branch) & (left_branch > 0)
        state_reset = right <= left
        boundary = (branch_change | (same_positive_branch & state_reset)) & valid_bonds
        return boundary.to(dtype=torch.float32), valid_bonds

    def _make_soft_bond_targets(self, tags: torch.Tensor, mask: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        # Return soft targets [B, L-1] and valid bond mask [B, L-1].
        hard, valid_bonds = self._exact_boundary_targets_from_tags(tags, mask)
        B, N = hard.shape
        device = hard.device
        soft = torch.zeros(B, N, dtype=dtype, device=device)
        if N == 0:
            return soft, valid_bonds

        tau = max(float(self.bond_soft_tau), 1e-6)
        window = int(self.bond_soft_window)
        positions = torch.arange(N, device=device, dtype=torch.float32)

        for bidx in range(B):
            true_pos = torch.nonzero(hard[bidx] > 0.5, as_tuple=False).flatten()
            if true_pos.numel() == 0:
                continue
            dist = torch.abs(positions[:, None] - true_pos.to(dtype=torch.float32)[None, :]).min(dim=1).values
            if self.bond_soft_mode == "gaussian":
                vals = torch.exp(-(dist * dist) / (2.0 * tau * tau))
            else:
                vals = torch.exp(-dist / tau)
            if window >= 0:
                vals = torch.where(dist <= window, vals, torch.zeros_like(vals))
            soft[bidx] = vals.to(dtype=dtype)

        soft = soft * valid_bonds.to(dtype=dtype)
        return soft, valid_bonds

    def _compute_bond_loss(self, features: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # features: [B, L, F]. targets/mask: [B, L].
        if features.size(1) < 2:
            return features.new_zeros(())
        bond_logits = self.bond_head(features)  # [B, L-1]
        soft_targets, valid_bonds = self._make_soft_bond_targets(targets, mask, dtype=bond_logits.dtype)
        valid = valid_bonds.to(dtype=bond_logits.dtype)
        if valid.sum() <= 0:
            return bond_logits.new_zeros(())

        weights = 1.0 + self.bond_positive_weight * soft_targets
        raw = F.binary_cross_entropy_with_logits(
            bond_logits,
            soft_targets,
            weight=weights,
            reduction="none",
        )
        return (raw * valid).sum() / valid.sum().clamp_min(1.0)

    def forward(
        self,
        embeddings,
        mask,
        targets=None,
        skip_marginals: bool = False,
        top_k: int = 1,
        decode: bool = True,
        return_probs: bool = True,
    ):
        mask = mask.bool()
        features = self.feature_extractor(embeddings, mask)          # [B, L, F]
        coarse_emissions = self.features_to_emissions(features)      # [B, L, num_labels]
        emissions = self._repeat_emissions(coarse_emissions)         # [B, L, num_states]
        boundary_logits = self.boundary_to_state(features)           # [B, L, 3 or 6]
        emissions = self._add_boundary_state_emissions(emissions, boundary_logits)

        loss = None
        if targets is not None:
            targets = targets.long()
            crf_loss = self.crf(emissions=emissions, tags=targets, mask=mask, reduction="mean") * -1
            bond_loss = self._compute_bond_loss(features, targets, mask)
            loss = crf_loss + self.bond_loss_lambda * bond_loss
            self.last_crf_loss = float(crf_loss.detach().item())
            self.last_bond_loss = float(bond_loss.detach().item())
            if crf_loss.item() > 10000:
                self._debug_crf(targets)

        viterbi_paths = None
        path_probs = None
        if decode:
            viterbi_paths, path_probs = self.crf.decode(emissions=emissions, mask=mask, top_k=top_k)

        probs = None
        if return_probs:
            probs = (
                self.crf.compute_marginal_probabilities(emissions, mask)
                if not skip_marginals
                else torch.softmax(emissions, dim=-1)
            )

        if targets is not None:
            return probs, viterbi_paths, loss
        else:
            return probs, viterbi_paths, path_probs


class LSTMCNNCRFGated3DiBoundary(CRFBaseModel):
    """Gated 3Di residual fusion + the winner's boundary-aware state-emission head.

    Combines the proven gated structural fusion (GatedResidualConvProjectedLSTMCNN, the
    feature extractor of `lstmcnncrf_gated3diresidual_conv`: a sequence branch + a gated
    3Di residual branch with a small Conv1d) with `BoundaryStateEmissionHead` from the
    project-best boundary model (start_1..W / inside / end_1..W state biases). No bond
    loss (shown to be dead weight). Intended for ESM-C 6B (2560) ⊕ 3Di (20) = 2580 input,
    layout [B, C, L] with channels [seq | struct].
    """
    def __init__(
        self,
        input_size: int = 2580,
        seq_input_size: int = 2560,
        struct_input_size: int = 20,
        seq_proj_size: int = 256,
        struct_proj_size: int = 32,
        dropout_projector: float = 0.4,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
        residual_scale: float = 0.1,
        struct_branch_dropout: float = 0.3,
        gate_bias: float = -2.5,
        struct_conv_kernel: int = 5,
        boundary_state_hidden_size: int = 64,
        boundary_state_dropout: float = 0.1,
        boundary_state_scale: float = 1.0,
        boundary_state_zero_init: bool = True,
        boundary_window: int = 1,
        seq_only: bool = False,
    ) -> None:
        super().__init__(num_labels, num_states)

        self.feature_extractor = GatedResidualConvProjectedLSTMCNN(
            input_size=input_size,
            seq_input_size=seq_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            struct_proj_size=struct_proj_size,
            dropout_projector=dropout_projector,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            dropout_conv1=dropout_conv1,
            seq_only=seq_only,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            residual_scale=residual_scale,
            struct_branch_dropout=struct_branch_dropout,
            gate_bias=gate_bias,
            struct_conv_kernel=struct_conv_kernel,
        )
        self.feature_size = int(n_filters) * 2
        self.boundary_window = int(boundary_window)
        self.features_to_emissions = nn.Linear(self.feature_size, num_labels)
        self.boundary_to_state = BoundaryStateEmissionHead(
            feature_size=self.feature_size,
            num_labels=num_labels,
            hidden_size=boundary_state_hidden_size,
            dropout=boundary_state_dropout,
            zero_init=boundary_state_zero_init,
            window=self.boundary_window,
        )
        self.boundary_state_scale = float(boundary_state_scale)
        self.num_labels = int(num_labels)
        self.num_states = int(num_states)

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )

    # Identical state-bias mapping as the boundary model (start/inside/end window W).
    _add_boundary_state_emissions = LSTMCNNCRFBoundaryBondLoss._add_boundary_state_emissions

    def forward(
        self,
        embeddings,
        mask,
        targets=None,
        skip_marginals: bool = False,
        top_k: int = 1,
        decode: bool = True,
        return_probs: bool = True,
    ):
        mask = mask.bool()
        features = self.feature_extractor(embeddings, mask)          # [B, L, F]
        coarse_emissions = self.features_to_emissions(features)      # [B, L, num_labels]
        emissions = self._repeat_emissions(coarse_emissions)         # [B, L, num_states]
        boundary_logits = self.boundary_to_state(features)
        emissions = self._add_boundary_state_emissions(emissions, boundary_logits)

        loss = None
        if targets is not None:
            targets = targets.long()
            loss = self.crf(emissions=emissions, tags=targets, mask=mask, reduction="mean") * -1

        viterbi_paths = path_probs = None
        if decode:
            viterbi_paths, path_probs = self.crf.decode(emissions=emissions, mask=mask, top_k=top_k)

        probs = None
        if return_probs:
            probs = (
                self.crf.compute_marginal_probabilities(emissions, mask)
                if not skip_marginals
                else torch.softmax(emissions, dim=-1)
            )

        if targets is not None:
            return probs, viterbi_paths, loss
        return probs, viterbi_paths, path_probs


class LSTMCNNCRFTelescopingSegmental(nn.Module):
    """LSTM-CNN + CRF with length-aware telescoping segment emissions.

    This model keeps the ordinary multistate CRF forward/backward implementation, but
    replaces copied PEPTIDE/PROPEP emissions with delta emissions:

        emission(i, y_k) = SpanScore_y(i-k+1, i) - SpanScore_y(i-k+1, i-1)

    for strict length/progress states y_1..y_K. The accumulated emission score along
    y_1 -> ... -> y_m is therefore SpanScore_y(s, e).

    Existing DeepPeptide labels are converted internally from the legacy compressed
    state encoding to strict length states for training. Decoded strict paths are
    converted back to legacy-like paths before returning so the existing metrics keep
    working unchanged.
    """

    NONE = 0
    LEGACY_PEP_START = 1
    LEGACY_PEP_END = 50
    LEGACY_PRO_START = 51
    LEGACY_PRO_END = 100

    def __init__(
        self,
        input_size: int = 1280,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 3,
        feature_extractor: str = "LSTMCNN",
        segmental_max_len: int = 50,
        segmental_min_len: int = 1,
        position_score_mode: str = "neg_abs",
        position_score_tau: float = 0.25,
        position_score_scale: float = 0.25,
        relative_position_loss_lambda: float = 0.0,
        allow_same_label_segments: bool = True,
    ) -> None:
        super().__init__()
        if num_labels not in (2, 3):
            raise ValueError(f"num_labels must be 2 or 3, got {num_labels}")
        if segmental_max_len <= 0:
            raise ValueError(f"segmental_max_len must be positive, got {segmental_max_len}")
        if segmental_min_len <= 0 or segmental_min_len > segmental_max_len:
            raise ValueError(
                f"segmental_min_len must be in [1, segmental_max_len], got {segmental_min_len}"
            )

        self.max_len = int(segmental_max_len)
        self.min_len = int(segmental_min_len)
        self.num_labels = int(num_labels)
        self.n_segment_labels = 2 if num_labels == 3 else 1
        self.num_states = 1 + self.n_segment_labels * self.max_len
        self.position_score_mode = str(position_score_mode)
        self.position_score_tau = float(position_score_tau)
        self.position_score_scale = float(position_score_scale)
        self.relative_position_loss_lambda = float(relative_position_loss_lambda)
        self.allow_same_label_segments = bool(allow_same_label_segments)
        self.feature_size = int(n_filters) * 2

        if feature_extractor == "LSTMCNN":
            self.feature_extractor = LSTMCNN(
                input_size=input_size,
                dropout_input=dropout_input,
                n_filters=n_filters,
                filter_size=filter_size,
                hidden_size=hidden_size,
                num_lstm_layers=num_lstm_layers,
                dropout_conv1=dropout_conv1,
                n_tissues=0,
            )
        else:
            self.feature_extractor = CustomTransformerWrapper(input_dim=input_size, output_dim=self.feature_size)

        self.features_to_emissions = nn.Linear(self.feature_size, num_labels)
        self.position_head = nn.Sequential(
            nn.LayerNorm(self.feature_size),
            nn.Dropout(dropout_input),
            nn.Linear(self.feature_size, self.n_segment_labels),
        )
        # Start with r_pred ~= 0.5, i.e. a neutral middle-position prior.
        final = self.position_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

        allowed_transitions, allowed_start, allowed_end = self.get_telescoping_crf_constraints(
            max_len=self.max_len,
            min_len=self.min_len,
            n_segment_labels=self.n_segment_labels,
            allow_same_label_segments=self.allow_same_label_segments,
        )
        self.allowed_transitions = allowed_transitions
        self.crf = CRF(
            self.num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )

        self.last_crf_loss = None
        self.last_position_loss = None

    @staticmethod
    def get_telescoping_crf_constraints(
        max_len: int = 50,
        min_len: int = 1,
        n_segment_labels: int = 2,
        allow_same_label_segments: bool = True,
    ):
        """Strict length-state grammar for telescoping emissions.

        State layout:
            0 = NONE
            1..K = PEPTIDE strict length/progress states
            K+1..2K = PROPEP strict length/progress states, if present
        """
        allowed_starts = [0]
        allowed_ends = [0]
        transitions = [(0, 0)]

        starts = []
        ends = []
        for branch in range(n_segment_labels):
            start = 1 + branch * max_len
            end = start + max_len - 1
            starts.append(start)
            ends.append(end)
            allowed_starts.append(start)

            transitions.append((0, start))
            for state in range(start, end):
                transitions.append((state, state + 1))

            for state in range(start + min_len - 1, end + 1):
                allowed_ends.append(state)
                transitions.append((state, 0))

        # Cross-label adjacent segments.
        for b_from, start_from in enumerate(starts):
            end_from = ends[b_from]
            for state in range(start_from + min_len - 1, end_from + 1):
                for b_to, start_to in enumerate(starts):
                    if b_to == b_from and not allow_same_label_segments:
                        continue
                    transitions.append((state, start_to))

        # Deduplicate while preserving order.
        seen = set()
        uniq = []
        for tr in transitions:
            if tr not in seen:
                uniq.append(tr)
                seen.add(tr)
        return uniq, allowed_starts, allowed_ends

    def _state_label(self, state: int) -> int:
        if state == 0:
            return 0
        if 1 <= state <= self.LEGACY_PEP_END:
            return 1
        if self.num_labels == 3 and self.LEGACY_PRO_START <= state <= self.LEGACY_PRO_END:
            return 2
        raise ValueError(f"Unsupported legacy state {state}; expected 0, 1..50, or 51..100")

    def _legacy_path_to_spans(self, path, length: int):
        """Convert a legacy compressed multistate path to non-overlapping spans.

        A new span starts when the coarse label changes or when a legacy start state
        appears immediately after another segment of the same label. This preserves
        adjacent same-label sampled targets from the overlap-aware dataset.
        """
        spans = []
        cur_label = 0
        start = None
        for pos in range(length):
            state = int(path[pos])
            label = self._state_label(state)
            is_start_state = state == self.LEGACY_PEP_START or (
                self.num_labels == 3 and state == self.LEGACY_PRO_START
            )

            if label == 0:
                if cur_label != 0:
                    spans.append((start, pos - 1, cur_label))
                    cur_label = 0
                    start = None
                continue

            if cur_label == 0:
                cur_label = label
                start = pos
            elif label != cur_label or is_start_state:
                spans.append((start, pos - 1, cur_label))
                cur_label = label
                start = pos

        if cur_label != 0:
            spans.append((start, length - 1, cur_label))
        return spans

    def _legacy_targets_to_strict(self, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Convert legacy DeepPeptide target states to strict length states for this CRF."""
        B, L = targets.shape
        out = torch.zeros_like(targets, dtype=torch.long)
        lengths = mask.long().sum(dim=1).tolist()
        for b in range(B):
            length = int(lengths[b])
            spans = self._legacy_path_to_spans(targets[b].detach().cpu().tolist(), length)
            for start, end, label in spans:
                m = end - start + 1
                if m > self.max_len:
                    raise ValueError(
                        f"Gold segment length {m} exceeds segmental_max_len={self.max_len}. "
                        f"Span: batch={b}, start={start}, end={end}, label={label}"
                    )
                if m < self.min_len:
                    raise ValueError(
                        f"Gold segment length {m} is below segmental_min_len={self.min_len}. "
                        f"Use --segmental_min_len 1 if short segments should be allowed."
                    )
                if label == 1:
                    base = 1
                elif label == 2 and self.num_labels == 3:
                    base = 1 + self.max_len
                else:
                    raise ValueError(f"Unsupported segment label {label}")
                out[b, start : end + 1] = torch.arange(
                    base, base + m, device=targets.device, dtype=torch.long
                )
        return out

    def _make_legacy_segment_states(self, label: int, length: int):
        """Make a metrics-compatible legacy path for one predicted segment.

        Existing manuscript metrics only require the start state (1/51) and end state
        (50/100) to recover borders. For length >= 5 we mimic the old compressed
        DeepPeptide encoding; for rare shorter spans we still mark first and last.
        """
        if label == 1:
            start_state = self.LEGACY_PEP_START
            end_state = self.LEGACY_PEP_END
        elif label == 2:
            start_state = self.LEGACY_PRO_START
            end_state = self.LEGACY_PRO_END
        else:
            return [0] * length

        if length <= 0:
            return []
        if length == 1:
            # Cannot encode start and end in one scalar state; this should be rare if min_len>=2/5.
            return [start_state]
        if length < 5:
            states = [start_state] * length
            states[-1] = end_state
            return states

        # Old compressed grammar: first min_len-2 states, then suffix ending in end_state.
        prefix_len = 3
        prefix = list(range(start_state, start_state + prefix_len))
        suffix_len = length - prefix_len
        suffix_start = end_state - suffix_len + 1
        suffix = list(range(suffix_start, end_state + 1))
        return prefix + suffix

    def _strict_path_to_legacy(self, path):
        out = [0] * len(path)
        cur_label = 0
        start = None
        spans = []

        for pos, state in enumerate(path):
            state = int(state)
            if state == 0:
                label = 0
                is_start = False
            elif 1 <= state <= self.max_len:
                label = 1
                is_start = state == 1
            elif self.num_labels == 3 and self.max_len < state <= 2 * self.max_len:
                label = 2
                is_start = state == self.max_len + 1
            else:
                raise ValueError(f"Invalid strict decoded state {state}")

            if label == 0:
                if cur_label != 0:
                    spans.append((start, pos - 1, cur_label))
                    cur_label = 0
                    start = None
                continue

            if cur_label == 0:
                cur_label = label
                start = pos
            elif label != cur_label or is_start:
                spans.append((start, pos - 1, cur_label))
                cur_label = label
                start = pos

        if cur_label != 0:
            spans.append((start, len(path) - 1, cur_label))

        for s, e, label in spans:
            states = self._make_legacy_segment_states(label, e - s + 1)
            out[s : e + 1] = states
        return out

    def _strict_paths_to_legacy(self, paths):
        if paths is None:
            return None
        return [self._strict_path_to_legacy(path) for path in paths]

    def _build_span_scores(self, coarse_logits: torch.Tensor, r_pred: torch.Tensor) -> torch.Tensor:
        """Compute SpanScore_y(start, end) indexed as [B, end, y, length-1]."""
        B, L, _ = coarse_logits.shape
        K = self.max_len
        n = self.n_segment_labels
        neg = torch.tensor(-1e4, dtype=coarse_logits.dtype, device=coarse_logits.device)
        span_scores = torch.full((B, L, n, K), neg, dtype=coarse_logits.dtype, device=coarse_logits.device)

        for branch in range(n):
            coarse_channel = 1 + branch
            coarse_y = coarse_logits[:, :, coarse_channel]  # [B,L]
            r_y = r_pred[:, :, branch]                      # [B,L]
            for k in range(1, K + 1):
                if k > L:
                    break
                windows_c = coarse_y.unfold(dimension=1, size=k, step=1)  # [B,L-k+1,k]
                windows_r = r_y.unfold(dimension=1, size=k, step=1)       # [B,L-k+1,k]
                if k == 1:
                    t_rel = torch.full((1, 1, 1), 0.5, dtype=coarse_logits.dtype, device=coarse_logits.device)
                else:
                    t_rel = torch.linspace(0.0, 1.0, steps=k, dtype=coarse_logits.dtype, device=coarse_logits.device).view(1, 1, k)
                pos_score = compute_position_score(
                    windows_r,
                    t_rel,
                    mode=self.position_score_mode,
                    tau=self.position_score_tau,
                )
                score = windows_c.sum(dim=-1) + self.position_score_scale * pos_score.sum(dim=-1)
                # windows are indexed by start; the corresponding end positions are k-1..L-1.
                span_scores[:, k - 1 :, branch, k - 1] = score
        return span_scores

    def _span_scores_to_delta_emissions(self, coarse_logits: torch.Tensor, span_scores: torch.Tensor) -> torch.Tensor:
        B, L, _ = coarse_logits.shape
        K = self.max_len
        n = self.n_segment_labels
        neg = torch.tensor(-1e4, dtype=coarse_logits.dtype, device=coarse_logits.device)
        state_emissions = torch.full((B, L, self.num_states), neg, dtype=coarse_logits.dtype, device=coarse_logits.device)
        state_emissions[:, :, 0] = coarse_logits[:, :, 0]

        for branch in range(n):
            state_start = 1 + branch * K
            # k = 1: delta = SpanScore(i,i)
            state_emissions[:, :, state_start] = span_scores[:, :, branch, 0]
            # k > 1: delta(end=i,k) = span(end=i,k) - span(end=i-1,k-1)
            for k in range(2, K + 1):
                if k > L:
                    break
                curr = span_scores[:, k - 1 :, branch, k - 1]
                prev = span_scores[:, k - 2 : L - 1, branch, k - 2]
                state_emissions[:, k - 1 :, state_start + k - 1] = curr - prev
        return state_emissions

    def _relative_position_aux_loss(self, r_pred: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.relative_position_loss_lambda <= 0:
            return r_pred.new_zeros(())

        B, L, _ = r_pred.shape
        lengths = mask.long().sum(dim=1).tolist()
        losses = []
        for b in range(B):
            length = int(lengths[b])
            spans = self._legacy_path_to_spans(targets[b].detach().cpu().tolist(), length)
            for start, end, label in spans:
                branch = label - 1
                if branch < 0 or branch >= self.n_segment_labels:
                    continue
                m = end - start + 1
                if m == 1:
                    r_true = torch.full((1,), 0.5, dtype=r_pred.dtype, device=r_pred.device)
                else:
                    r_true = torch.linspace(0.0, 1.0, steps=m, dtype=r_pred.dtype, device=r_pred.device)
                losses.append(F.smooth_l1_loss(r_pred[b, start : end + 1, branch], r_true, reduction="mean"))
        if not losses:
            return r_pred.new_zeros(())
        return torch.stack(losses).mean()

    def _debug_crf(self, targets):
        for i in range(targets.shape[0]):
            for j in range(1, targets.shape[1]):
                l = int(targets[i, j].item())
                l_prev = int(targets[i, j - 1].item())
                if (l_prev, l) not in self.allowed_transitions:
                    print(f"Found invalid transition from {l_prev} to {l}.")

    def forward(
        self,
        embeddings,
        mask,
        targets=None,
        skip_marginals: bool = False,
        top_k: int = 1,
        decode: bool = True,
        return_probs: bool = True,
    ):
        if top_k != 1:
            raise NotImplementedError("top_k > 1 is not supported for telescoping segmental CRF")
        mask = mask.bool()

        if isinstance(self.feature_extractor, CustomTransformerWrapper):
            features = self.feature_extractor(embeddings.transpose(1, 2))
        else:
            features = self.feature_extractor(embeddings, mask)

        coarse_logits = self.features_to_emissions(features)       # [B,L,num_labels]
        position_raw = self.position_head(features)                # [B,L,n_segment_labels]
        r_pred = torch.sigmoid(position_raw)                       # [B,L,n_segment_labels]
        span_scores = self._build_span_scores(coarse_logits, r_pred)
        state_emissions = self._span_scores_to_delta_emissions(coarse_logits, span_scores)

        loss = None
        if targets is not None:
            targets = targets.long()
            strict_targets = self._legacy_targets_to_strict(targets, mask)
            crf_loss = self.crf(emissions=state_emissions, tags=strict_targets, mask=mask, reduction="mean") * -1
            pos_loss = self._relative_position_aux_loss(r_pred, targets, mask)
            loss = crf_loss + self.relative_position_loss_lambda * pos_loss
            self.last_crf_loss = float(crf_loss.detach().item())
            self.last_position_loss = float(pos_loss.detach().item())
            if crf_loss.item() > 10000:
                self._debug_crf(strict_targets)

        viterbi_paths = None
        path_probs = None
        if decode:
            strict_paths, path_probs = self.crf.decode(emissions=state_emissions, mask=mask, top_k=top_k)
            viterbi_paths = self._strict_paths_to_legacy(strict_paths)

        probs = None
        if return_probs:
            probs = (
                self.crf.compute_marginal_probabilities(state_emissions, mask)
                if not skip_marginals
                else torch.softmax(state_emissions, dim=-1)
            )

        if targets is not None:
            return probs, viterbi_paths, loss
        return probs, viterbi_paths, path_probs

class LSTMCNNCRF(CRFBaseModel):
    '''LSTM-CNN feature extractor + multistate CRF.'''
    def __init__(
        self,
        input_size: int = 1280,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int =3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers : int = 1,
        num_labels: int = 2, #logits (=emissions) to produce by the NN
        num_states = 61, # total number of states in the state space model
        feature_extractor = 'LSTMCNN'
        ) -> None:


        super().__init__(num_labels, num_states)

        if feature_extractor == 'LSTMCNN':
            self.feature_extractor = LSTMCNN(input_size=input_size, dropout_input=dropout_input, n_filters=n_filters, filter_size=filter_size, hidden_size=hidden_size, num_lstm_layers=1, dropout_conv1=dropout_conv1, n_tissues=0)
        else:
            self.feature_extractor = CustomTransformerWrapper(input_dim=input_size)
        self.features_to_emissions = nn.Linear(n_filters*2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(self.max_len, self.min_len, n_branches=2 if num_labels==3 else 1)
        self.crf = CRF(num_states, batch_first=True, allowed_transitions=allowed_transitions, allowed_start=allowed_start, allowed_end=allowed_end)


class LSTMCNNCRFProjector(CRFBaseModel):
    """LSTM-CNN feature extractor + multistate CRF, with a small trainable projector on the input embeddings.

    Use this when you feed precomputed embeddings [B, C_in, L] (e.g., ESM2/ESMC/ProstT5),
    but want a lightweight trainable adaptation before the LSTMCNN.
    """
    def __init__(
        self,
        input_size: int = 1280,
        proj_size: int = 512,
        dropout_projector: float = 0.2,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
    ) -> None:
        super().__init__(num_labels, num_states)

        self.feature_extractor = ProjectedLSTMCNN(
            input_size=input_size,
            proj_size=proj_size,
            dropout_projector=dropout_projector,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            dropout_conv1=dropout_conv1,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
        )
        self.features_to_emissions = nn.Linear(n_filters * 2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )


class SimpleLSTMCNNCRF(CRFBaseModel):
    '''LSTM-CNN feature extractor with simple 2-state CRF model.'''
    def __init__(
        self,
        input_size: int = 1280,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int =3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers : int = 1,
        num_labels: int = 2, #logits (=emissions) to produce by the NN
        num_states = 2 # total number of states in the state space model
        ) -> None:


        super().__init__()

        self.feature_extractor = LSTMCNN(input_size=input_size, dropout_input=dropout_input, n_filters=n_filters, filter_size=filter_size, hidden_size=hidden_size, num_lstm_layers=1, dropout_conv1=dropout_conv1, n_tissues=0)
        self.features_to_emissions = nn.Linear(n_filters*2, num_labels)
        self.num_states = num_states
        self.crf = CRF(num_states, batch_first=True) # no constraints on CRF.


    # redefine forward because no emission repeating.
    def forward(self, embeddings, mask, targets = None, skip_marginals: bool = False):
        features = self.feature_extractor(embeddings, mask) # (batch_size, seq_len, feature_dim)
        emissions = self.features_to_emissions(features) # (batch_size, seq_len, num_labels)
        
        viterbi_paths, probs = self.crf.decode(emissions=emissions, mask = mask.bool())

        #pad the viterbi paths
        # max_pad_len = max([len(x) for x in viterbi_paths])
        # pos_preds = [x + [-1]*(max_pad_len-len(x)) for x in viterbi_paths] 
        # pos_preds = torch.tensor(pos_preds, device = emissions.device) #Tensor conversion is just for compatibility with downstream metric functions

        probs = self.crf.compute_marginal_probabilities(emissions, mask.bool()) if not skip_marginals else torch.softmax(emissions, dim=-1)

        if targets is not None:
            loss = self.crf(emissions = emissions, tags=targets.long(), mask = mask.bool(), reduction='mean') *-1
            return (probs, viterbi_paths, loss)
        else:
            return probs, viterbi_paths


class SelfAttentionFeatureNet(nn.Module):

    def __init__(self,
        input_size: float = 1280,
        hidden_size: float = 640,
        dropout_input: float = 0.25,
        n_heads: int = 4,
        attn_dropout: float = 0.1, 
        ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout_input)

        if hidden_size != input_size:
            self.projection = nn.Linear(input_size, hidden_size)
        self.mha = nn.MultiheadAttention(hidden_size,n_heads, attn_dropout, batch_first=True)


    def forward(self, inputs, mask):

        inputs = inputs.transpose(2,1)
        inputs = self.dropout(inputs)
        attn_mask = 1 -mask
        # key_padding_mask – If specified, a mask of shape (N, S)
        # indicating which elements within key to ignore for the purpose of attention (i.e. treat as “padding”). 
        # For unbatched query, shape should be (S)(S). Binary and byte masks are supported. 
        # For a binary mask, a True value indicates that the corresponding key value will be ignored for the purpose of attention. 
        # For a byte mask, a non-zero value indicates that the corresponding key value will be ignored
        #attn_mask = torch.
        if self.hidden_size != self.input_size:
            inputs = self.projection(inputs)

        out, attn = self.mha(inputs, inputs, inputs, key_padding_mask = attn_mask)

        return out


class SelfAttentionCRF(CRFBaseModel):
    '''Attention feature extractor + multistate CRF.'''
    def __init__(
        self,
        input_size: int = 1280,
        hidden_size: int = 128,
        dropout_input: float = 0.25,
        n_heads: int = 4,
        attn_dropout: float = 0.15,
        num_labels: int = 2, #logits (=emissions) to produce by the NN
        num_states = 61 # total number of states in the state space model
        ) -> None:


        super().__init__(num_labels, num_states)

        self.feature_extractor = SelfAttentionFeatureNet(input_size, hidden_size, dropout_input, n_heads, attn_dropout)

        self.features_to_emissions = nn.Linear(hidden_size, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(n_branches=2 if num_labels==3 else 1)
        self.crf = CRF(num_states, batch_first=True, allowed_transitions=allowed_transitions, allowed_start=allowed_start, allowed_end=allowed_end)

class LoRALinear(nn.Module):
    """LoRA wrapper for a frozen nn.Linear module.

    forward(x) = base(x) + scale * B(A(dropout(x)))
    The base weights are frozen; only A/B are trainable. B is zero-initialized so
    the wrapped layer initially behaves exactly like the pretrained layer.
    """
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_B(self.lora_A(self.dropout(x)))


def _parse_lora_layer_indices(spec: str | None, num_layers: int) -> set[int] | None:
    """Return allowed ESM layer indices for LoRA, or None for all layers.

    Supported forms:
      - "all", "", None: all layers
      - "last:4" or "last4": last 4 transformer layers
      - "0,1,2,31,32": explicit layer indices

    fair-esm ESM2 module names normally contain paths like "layers.31.fc1".
    """
    if spec is None:
        return None
    text = str(spec).strip().lower()
    if text in {"", "all", "none"}: 
        return None
    if text.startswith("last:"):
        n = int(text.split(":", 1)[1])
        return set(range(max(0, int(num_layers) - n), int(num_layers)))
    if text.startswith("last") and text[4:].isdigit():
        n = int(text[4:])
        return set(range(max(0, int(num_layers) - n), int(num_layers)))
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def _module_layer_index(full_name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", full_name)
    if match is None:
        return None
    return int(match.group(1))


def _replace_esm_linear_with_lora(
    module: nn.Module,
    target_modules,
    rank: int,
    alpha: float,
    dropout: float,
    prefix: str = "",
    allowed_layers: set[int] | None = None,
) -> int:
    """Recursively replace selected Linear layers with LoRALinear.

    target_modules is a collection of substrings matched against full module names.
    For fair-esm ESM2 useful names are usually q_proj,k_proj,v_proj,out_proj,fc1,fc2.
    allowed_layers restricts LoRA to transformer block indices from module paths like layers.31.fc1.
    """
    replaced = 0
    targets = tuple(t.strip() for t in target_modules if str(t).strip())
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        layer_idx = _module_layer_index(full_name)
        layer_allowed = allowed_layers is None or layer_idx is None or layer_idx in allowed_layers
        if isinstance(child, nn.Linear) and layer_allowed and any(t in full_name for t in targets):
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced += 1
        else:
            replaced += _replace_esm_linear_with_lora(
                child, targets, rank, alpha, dropout, prefix=full_name, allowed_layers=allowed_layers
            )
    return replaced


def _set_esm_layer_norm_trainable(module: nn.Module, trainable: bool = True) -> int:
    count = 0
    for m in module.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad = trainable
                count += p.numel()
    return count


def _available_fair_esm_loaders(pretrained_module) -> list[str]:
    return sorted(
        name for name in dir(pretrained_module)
        if name.startswith("esm2_") or name.startswith("esm1") or name.startswith("esm_msa")
    )


def _load_fair_esm_model(pretrained_module, model_name: str):
    """Load a fair-esm model robustly across fair-esm versions.

    Some fair-esm versions expose ESM2 checkpoints as functions, e.g.
    pretrained.esm2_t33_650M_UR50D(). Other installations only expose the generic
    load_model_and_alphabet/load_model_and_alphabet_hub helpers. Try all supported
    routes before failing, and report available loader names for debugging.
    """
    if hasattr(pretrained_module, model_name):
        loader = getattr(pretrained_module, model_name)
        if callable(loader):
            return loader()

    errors = []
    for helper_name in ("load_model_and_alphabet", "load_model_and_alphabet_hub"):
        helper = getattr(pretrained_module, helper_name, None)
        if callable(helper):
            try:
                return helper(model_name)
            except Exception as exc:  # keep trying other fair-esm APIs
                errors.append(f"{helper_name}({model_name!r}) failed: {type(exc).__name__}: {exc}")

    available = _available_fair_esm_loaders(pretrained_module)
    available_msg = ", ".join(available[:40]) if available else "<no esm loaders found in esm.pretrained>"
    detail = "\n".join(errors)
    raise ValueError(
        f"Could not load fair-esm model {model_name!r}. Available direct loader names: {available_msg}."
        + (f"\nFallback errors:\n{detail}" if detail else "")
    )


class LSTMCNNCRFESM2LoRA(CRFBaseModel):
    """Online ESM2 -> LSTMCNN -> multistate CRF with true LoRA inside ESM2.

    This model expects a batch of raw amino-acid sequences, not precomputed tensors.
    The pretrained ESM2 weights are frozen; selected ESM2 Linear layers are wrapped
    with trainable LoRA adapters. The downstream LSTMCNN, emission head, and CRF are
    ordinary trainable DeepPeptide components.
    """
    def __init__(
        self,
        esm2_model_name: str = "esm2_t33_650M_UR50D",
        esm2_repr_layer: int = -1,
        esm2_max_sequence_length: int = 1022,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_modules: str = "q_proj,k_proj,v_proj,out_proj,fc1,fc2",
        lora_layers: str = "all",
        train_esm_layer_norm: bool = False,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
    ) -> None:
        super().__init__(num_labels, num_states)

        try:
            from esm import pretrained
        except Exception as exc:
            raise ImportError(
                "LSTMCNNCRFESM2LoRA requires the fair-esm package (`import esm`). "
                "Install it in the project environment before using --model lstmcnncrf_esm2_lora."
            ) from exc

        self.esm_model, self.esm_alphabet = _load_fair_esm_model(pretrained, esm2_model_name)
        self.batch_converter = self.esm_alphabet.get_batch_converter()
        self.esm2_model_name = esm2_model_name
        self.esm2_max_sequence_length = int(esm2_max_sequence_length)

        self.esm_num_layers = int(getattr(self.esm_model, "num_layers", 0))
        if esm2_repr_layer is None or int(esm2_repr_layer) < 0:
            self.esm2_repr_layer = self.esm_num_layers
        else:
            self.esm2_repr_layer = int(esm2_repr_layer)

        # Infer ESM embedding dimension robustly across fair-esm versions.
        embed_dim = getattr(self.esm_model, "embed_dim", None)
        if embed_dim is None and hasattr(self.esm_model, "args"):
            embed_dim = getattr(self.esm_model.args, "embed_dim", None)
        if embed_dim is None:
            embed_dim = getattr(getattr(self.esm_model, "embed_tokens", None), "embedding_dim", None)
        if embed_dim is None:
            raise ValueError("Could not infer ESM2 embedding dimension from the fair-esm model.")
        self.esm_embed_dim = int(embed_dim)

        # Freeze all original ESM2 parameters, then insert trainable LoRA modules.
        for p in self.esm_model.parameters():
            p.requires_grad = False

        target_modules = [x.strip() for x in str(lora_target_modules).split(",") if x.strip()]
        allowed_lora_layers = _parse_lora_layer_indices(lora_layers, self.esm_num_layers)
        self.lora_layers = "all" if allowed_lora_layers is None else ",".join(str(i) for i in sorted(allowed_lora_layers))
        self.n_lora_modules = _replace_esm_linear_with_lora(
            self.esm_model,
            target_modules=target_modules,
            rank=int(lora_rank),
            alpha=float(lora_alpha),
            dropout=float(lora_dropout),
            allowed_layers=allowed_lora_layers,
        )
        if self.n_lora_modules == 0:
            raise ValueError(
                "No ESM2 Linear layers were matched for LoRA. "
                f"Targets were: {target_modules}; layers={lora_layers}. "
                "Try e.g. --esm2_lora_target_modules q_proj,v_proj --esm2_lora_layers last:4"
            )

        self.esm_layer_norm_trainable_params = 0
        if train_esm_layer_norm:
            self.esm_layer_norm_trainable_params = _set_esm_layer_norm_trainable(self.esm_model, True)

        self.feature_extractor = LSTMCNN(
            input_size=self.esm_embed_dim,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        self.features_to_emissions = nn.Linear(n_filters * 2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.allowed_transitions = allowed_transitions
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )

    def _sequences_to_esm_embeddings(self, sequences, max_residue_len: int, device: torch.device) -> torch.Tensor:
        if not isinstance(sequences, (list, tuple)):
            raise TypeError(
                "LSTMCNNCRFESM2LoRA expects a list/tuple of raw sequence strings from "
                "OnlineESMCSVForOverlapCRFDataset."
            )
        too_long = [len(s) for s in sequences if len(s) > self.esm2_max_sequence_length]
        if too_long:
            raise ValueError(
                f"Found sequence length {max(too_long)} > --esm2_max_sequence_length {self.esm2_max_sequence_length}. "
                "Online ESM2 LoRA currently does not chunk long proteins; filter them, increase the supported limit "
                "only if the ESM2 model can handle it, or implement chunked ESM2 embeddings."
            )

        data = [(str(i), seq) for i, seq in enumerate(sequences)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(device)
        out = self.esm_model(tokens, repr_layers=[self.esm2_repr_layer], return_contacts=False)
        reps = out["representations"][self.esm2_repr_layer]  # [B, T+2, C]
        reps = reps[:, 1 : 1 + max_residue_len, :]           # strip BOS; keep padded residue region
        return reps.transpose(1, 2).contiguous()             # [B, C, L]

    def forward(
        self,
        embeddings,
        mask,
        targets=None,
        skip_marginals: bool = False,
        top_k: int = 1,
        decode: bool = True,
        return_probs: bool = True,
    ):
        mask = mask.bool()
        device = mask.device
        max_residue_len = mask.size(1)

        esm_embeddings = self._sequences_to_esm_embeddings(embeddings, max_residue_len, device)
        features = self.feature_extractor(esm_embeddings, mask)
        emissions = self.features_to_emissions(features)
        emissions = self._repeat_emissions(emissions)

        loss = None
        if targets is not None:
            targets = targets.long()
            loss = self.crf(emissions=emissions, tags=targets, mask=mask, reduction="mean") * -1
            if loss.item() > 10000:
                self._debug_crf(targets)

        viterbi_paths = None
        path_probs = None
        if decode:
            viterbi_paths, path_probs = self.crf.decode(emissions=emissions, mask=mask, top_k=top_k)

        probs = None
        if return_probs:
            probs = (
                self.crf.compute_marginal_probabilities(emissions, mask)
                if not skip_marginals
                else torch.softmax(emissions, dim=-1)
            )

        if targets is not None:
            return probs, viterbi_paths, loss
        else:
            return probs, viterbi_paths, path_probs