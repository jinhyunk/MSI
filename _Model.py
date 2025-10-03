# model.py
import torch
import torch.nn as nn

# --- Core Model Components (from _Model.py) ---
# It's recommended to have these in a separate file like `components.py`
from _Model import MLP, Encoder_champ, Encoder_player, Encoder_region, Encoder_position, Encoder_ELO

# --- Refactored Data Modules ---
# These loaders now rely on the DataManager/PowerManager
from _Module import ChampionStatsLoader, PlayerStatsLoader, PowerOverTimeLoader 

# --- Project-wide Settings ---
from config import REGIONS, LEAGUES

class MsiModel(nn.Module):
    """
    Refactored MSI Model that relies on DataManager and PowerManager for all data loading.
    """
    def __init__(self, params, data_manager, power_manager):
        super().__init__()
        
        # Store data managers to pass them to loaders
        self.data_manager = data_manager
        self.power_manager = power_manager

        # --- Encoders (No changes here) ---
        self.enc_rank = Encoder_champ(
            init_s_wr=params['s_rank'], emb_size=params['emb_size_enc'], out_size=1, c_wr=params['c_wr']
        )
        self.enc_lg = Encoder_champ(
            init_s_wr=params['s_lg'], emb_size=params['emb_size_enc'], out_size=1, c_wr=params['c_wr']
        )
        self.enc_player = Encoder_player(
            init_s_wr=params['s_player_wr'], init_s_game=params['s_player_game'],
            emb_size=params['emb_size_enc'], out_size=params['emb_size_sum'],
            c_wr=params['c_wr'], mim_game=params['mim_game']
        )
        self.sum_region = Encoder_region(len(REGIONS), params['emb_size_enc'], params['emb_size_sum'])
        self.sum_lg = Encoder_region(len(LEAGUES), params['emb_size_enc'], params['emb_size_sum'])

        # --- Data Loaders (Now initialized with manager instances) ---
        self.loader_rank = ChampionStatsLoader(self.enc_rank, self.data_manager, 'rank')
        self.loader_lg = ChampionStatsLoader(self.enc_lg, self.data_manager, 'lg')
        self.loader_player = PlayerStatsLoader(self.enc_player, self.data_manager)
        self.loader_po = PowerOverTimeLoader(self.enc_rank.normalizer, self.power_manager)

        # --- Aggregators & Final Layers (No changes here) ---
        emb_sum_size = params['emb_size_sum']
        self.enc_position = Encoder_position(
            in_size=4 * emb_sum_size, emb_size=8 * emb_sum_size, out_size=4 * emb_sum_size
        )
        self.enc_elo = Encoder_ELO(
            emb_size=8 * emb_sum_size, out_size=4 * emb_sum_size,
            mu=params['elo_mu'], sigma=params['elo_sigma']
        )
        self.mlp = MLP(
            in_size=8 * emb_sum_size, emb_size=16 * emb_sum_size, out_size=1
        )
        # Note: self.find_elo is now handled by self.data_manager

    def forward_single(self, time, team, pb, idx_match):
        """Processes a single data sample for prediction."""
        
        # 1. Extract embeddings using the refactored loaders
        rank_emb = self.sum_region(self.loader_rank(pb))
        lg_emb = self.sum_lg(self.loader_lg(pb))
        player_emb = self.loader_player(team, pb)
        po_emb = self.sum_region(self.loader_po(time, pb))

        # 2. Aggregate features by position
        champion_features = self.enc_position(rank_emb, lg_emb, player_emb, po_emb)

        # 3. Get ELO using DataManager and create embedding
        # The refactored find_elo function is now part of data_manager
        elo_tensor = self.data_manager.find_elo(idx_match, team)
        if elo_tensor is None:
            # Handle cases where ELO might not be found
            elo_tensor = torch.tensor([self.enc_elo.mu], dtype=torch.float32, device=champion_features.device)
        
        elo_features = self.enc_elo(elo_tensor.to(champion_features.device))

        # 4. Combine all features and make the final prediction
        combined_features = torch.cat([champion_features, elo_features], dim=1)
        prediction = self.mlp(combined_features)
        
        return prediction

    def forward(self, times, teams, pbs, idx_matches):
        """Processes a batch of data for prediction."""
        batch_size = len(teams)
        # Ensure the results tensor is on the same device as the model parameters
        device = next(self.parameters()).device
        results = torch.zeros(batch_size, 1, device=device)
        
        for i in range(batch_size):
            results[i] = self.forward_single(times[i], teams[i], pbs[i], idx_matches[i])
            
        return results