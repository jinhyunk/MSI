# modules.py
import torch
import torch.nn as nn
from _Load import load_player, load_power
from config import REGIONS, LEAGUES, CPU_DEVICE

class ChampionStatsLoader(nn.Module):
    """ 
    챔피언의 랭크/리그 데이터를 로드하고 인코딩합니다.
    기존의 Loader_champ, reader, stacker를 하나로 캡슐화했습니다.
    """
    def __init__(self, encoder, loader, data_keys, data_prefix):
        super().__init__()
        self.encoder = encoder
        self.loader = loader
        self.data_keys = data_keys  # e.g., REGIONS or LEAGUES
        self.data_prefix = data_prefix # e.g., 'rank' or 'lg'
        self.cache = {}

    def _read_and_encode(self, data):
        output = {}
        for key in self.data_keys:
            wr_key = f'wr_{self.data_prefix}_{key}'
            pb_key = f'pb_{self.data_prefix}_{key}'
            if wr_key in data and pb_key in data:
                wr, pb = data[wr_key], data[pb_key]
                output[key] = self.encoder(wr, pb)
        return output

    def _stack_tensors(self, encoded_data):
        tensors = [encoded_data[key] for key in self.data_keys if key in encoded_data]
        if not tensors:
            return torch.empty(0, device=self.encoder.s_wr.device)
        return torch.stack(tensors).view(-1)

    def forward(self, pb):
        device = self.encoder.s_wr.device
        result = []
        for pos_idx, champ in enumerate(pb):
            key = (champ, pos_idx, self.data_prefix)
            if key in self.cache:
                out_tensor = self.cache[key].to(device)
            else:
                data = self.loader(champ, pos_idx)
                encoded_data = self._read_and_encode(data)
                out_tensor = self._stack_tensors(encoded_data)
                self.cache[key] = out_tensor.detach().to(CPU_DEVICE)
            result.append(out_tensor)
        return torch.stack(result)

class PlayerStatsLoader(nn.Module):
    """
    플레이어 데이터를 로드하고 인코딩합니다.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.loader = load_player
        self.cache = {}

    def forward(self, team, pb):
        device = self.encoder.s_wr.device
        result = []
        for pos_idx, champ in enumerate(pb):
            key = (team, champ, pos_idx)
            if key in self.cache:
                out = self.cache[key].to(device)
            else:
                data = self.loader(team, champ, pos_idx)
                game, wr = data["game_gamer"], data["wr_gamer"]
                out = self.encoder(
                    torch.tensor(game, dtype=torch.float32, device=device),
                    torch.tensor(wr, dtype=torch.float32, device=device)
                )
                self.cache[key] = out.detach().to(CPU_DEVICE)
            result.append(out)
        return torch.stack(result)

class PowerOverTimeLoader(nn.Module):
    """
    시간에 따른 챔피언 파워(PO) 데이터를 로드하고 인코딩합니다.
    """
    def __init__(self, normalizer):
        super().__init__()
        self.normalizer = normalizer
        self.loader = load_power
        self.cache = {}
        
    def _read_and_encode(self, data):
        output = {}
        for region in REGIONS:
            po_key = f'po_{region}'
            if po_key in data:
                val = data[po_key]
                tensor_val = torch.tensor(val, dtype=torch.float32)
                output[region] = self.normalizer(tensor_val).reshape(-1, 1)
        return output
        
    def _stack_tensors(self, encoded_data):
        tensors = [encoded_data[key] for key in REGIONS if key in encoded_data]
        if not tensors:
            return torch.empty(0)
        return torch.cat(tensors).view(-1)
        
    def forward(self, time, pb):
        device = self.normalizer.s_wr.device
        t_key = round(time.item() if isinstance(time, torch.Tensor) else float(time), 3)
        result = []
        for pos_idx, champ in enumerate(pb):
            key = (t_key, champ, pos_idx)
            if key in self.cache:
                out_tensor = self.cache[key].to(device)
            else:
                data = self.loader(time, champ, pos_idx)
                encoded = self._read_and_encode(data)
                out_tensor = self._stack_tensors(encoded)
                self.cache[key] = out_tensor.detach().to(CPU_DEVICE)
            result.append(out_tensor)
        return torch.stack(result)