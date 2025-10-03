# train.py
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os

# --- Refactored Imports ---
# Import the updated model and data management classes
from _Model import MsiModel 
from _Load import DataManager, PowerManager, load_match_train
from train_module import MSIDataset, collate_fn 
import config

def train(args):
    # --- Setup ---
    device = config.DEVICE
    os.makedirs(config.WEIGHT_SAVE_PATH, exist_ok=True)
    print(f"Using device: {device}")

    # --- Initialize Data and Power Managers ---
    # Create single instances of the managers to handle all data loading and caching
    data_manager = DataManager(base_path="./data/")
    power_manager = PowerManager(data_manager, po_json_path="./json/po/")

    # --- Initialize Model, Optimizer, and Criterion ---
    # Pass the manager instances to the model's constructor
    model = MsiModel(config.MODEL_PARAMS, data_manager, power_manager).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN_PARAMS['learning_rate'], weight_decay=config.TRAIN_PARAMS['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10) if config.TRAIN_PARAMS['use_scheduler'] else None
    criterion = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # --- Data Loader ---
    # The rest of the data loading process remains the same
    train_data = load_match_train()
    dataset = MSIDataset(train_data, device)
    dataloader = DataLoader(
        dataset,
        batch_size=config.TRAIN_PARAMS['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.TRAIN_PARAMS['num_workers'],
        pin_memory=True,
        prefetch_factor=4 if config.TRAIN_PARAMS['num_workers'] > 0 else 2,
    )

    # --- Training Loop ---
    best_loss = float('inf')
    start_time = time.time()
    print("===== Training Started =====")

    for epoch in range(config.TRAIN_PARAMS['epochs']):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.TRAIN_PARAMS['epochs']}"):
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                # Model forward pass
                wr_blues = model(batch['input_times'], batch['blue_teams'], batch['bp_blues'], batch['data_indices'])
                wr_reds  = model(batch['input_times'], batch['red_teams'],  batch['bp_reds'],  batch['data_indices'])
                
                # Calculate loss
                loss_blue = criterion(wr_blues, batch['wr_blue_gts'].to(device).view(-1, 1))
                loss_red  = criterion(wr_reds,  batch['wr_red_gts'].to(device).view(-1, 1))
                total_loss = loss_blue + loss_red

            # Backpropagation
            scaler.scale(total_loss).backward()
            if config.TRAIN_PARAMS['gradient_clip'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN_PARAMS['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | Best Loss: {best_loss:.6f}")

        if scheduler:
            scheduler.step(avg_loss)
            
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"Best model saved to {config.BEST_MODEL_PATH}")

    print(f"===== Training Finished in {(time.time() - start_time)/60:.2f} minutes =====")
    torch.save(model.state_dict(), config.FINAL_MODEL_PATH)
    print(f"Final model saved to {config.FINAL_MODEL_PATH}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # You can add arguments here to override config values from the command line if needed
    # e.g., parser.add_argument('--learning_rate', type=float)
    args = parser.parse_args()
    train(args)