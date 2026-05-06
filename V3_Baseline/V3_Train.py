print(">>> V3_Train.py: Module Loading...")
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
from pathlib import Path
import argparse

# Local Imports
from V3_Model import MultiTaskScalogramV3
from V3_HDF5_DataLoader import create_v3_dataloaders
from scalogramv4.models.V4_PINN_Loss import HierarchicalMultiTaskLoss

# ---------------------------------------------------------
# 1. Specialized Loss Functions
# ---------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        log_pt = -self.ce(inputs, targets)
        pt = torch.exp(log_pt)
        loss = -self.alpha * (1 - pt) ** self.gamma * log_pt
        return loss

class VonMisesLoss(nn.Module):
    """Circular Regression Loss for Azimuth."""
    def __init__(self, kappa=1.0):
        super(VonMisesLoss, self).__init__()
        self.kappa = kappa

    def forward(self, pred, target_rad):
        # pred: [sin, cos]
        # target_rad: scalar radian
        pred_sin = pred[:, 0]
        pred_cos = pred[:, 1]

        target_sin = torch.sin(target_rad)
        target_cos = torch.cos(target_rad)

        # Mean Square Error on [sin, cos] space
        loss_sin = (pred_sin - target_sin) ** 2
        loss_cos = (pred_cos - target_cos) ** 2
        return torch.mean(loss_sin + loss_cos)

# ---------------------------------------------------------
# 2. Training Routine
# ---------------------------------------------------------
def train_v3(data_h5, epochs=15, batch_size=32, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"ScalogramV3: Training on {device}")

    # Model
    print("Loading ScalogramV3 Model...")
    model = MultiTaskScalogramV3(pretrained=False).to(device)

    # Data
    print(f"Opening HDF5 Dataset: {data_h5}")
    train_loader, val_loader = create_v3_dataloaders(data_h5, batch_size=batch_size, num_workers=0)
    print(f"Ready. Total Train Batches: {len(train_loader)}")

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Losses
    # We now use HierarchicalMultiTaskLoss from V4 for the physics-informed constraint
    # The lambdas are [det, mag, azm, phys]
    criterion_hier = HierarchicalMultiTaskLoss(lambdas=[1.0, 0.5, 0.2, 0.5]).to(device)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        print(f"\n>>> Starting Epoch {epoch+1}/{epochs}")

        for i, (img, cosmic, y_det, y_mag, y_azm) in enumerate(train_loader):
            img = img.to(device)
            cosmic = cosmic.to(device)
            y_det = y_det.long().to(device)
            y_mag = y_mag.long().to(device)
            y_azm = y_azm.to(device)

            optimizer.zero_grad()

            # Forward
            # V3 now returns (out_det, out_mag, out_azm, reg_score, att_weights)
            out_det, out_mag, out_azm, reg_score, att_weights = model(img, cosmic)

            # Prepare targets for Hierarchical Loss
            # targets = {'det': y_det, 'mag': y_mag, 'azm': y_azm, 'epc_coords': None}
            # Note: y_azm is assumed to be in [sin, cos] for V4 compatibility.
            # If it's radians, it needs conversion.

            # For V3 training stability, we can either use the HierarchicalLoss or
            # the custom manual weighted sum. To fully port V4 physics, we use:

            outputs = {
                'stage1_logAnonymous Institution': out_det,
                'stage2_mag': out_mag.max(dim=1)[0].unsqueeze(1), # Proxy mag as max logit
                'stage3_azm': out_azm,
                'regional_score': reg_score,
                'attention': att_weights
            }
            targets = {
                'det': y_det,
                'mag': y_mag.float(),
                'azm': y_azm # Assume y_azm is provided as [sin, cos] vectors
            }

            # We use a dummy adj_matrix (identity) for V3 as it is single-station per sample
            adj_dummy = torch.eye(8, device=device)

            loss_dict = criterion_hier(outputs, targets, adj_dummy)
            total_loss = loss_dict['total']

            # Backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += total_loss.item()

            if i % 10 == 0:
                print(f"  Batch {i}/{len(train_loader)} | Loss: {total_loss.item():.4f} | Alpha: {loss_dict['alpha']:.6f}")

        avg_loss = train_loss / len(train_loader)
        print(f"Done. Avg Train Loss: {avg_loss:.5f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, cosmic, y_det, y_mag, y_azm in val_loader:
                img, cosmic = img.to(device), cosmic.to(device)
                y_det, y_mag, y_azm = y_det.to(device), y_mag.to(device), y_azm.to(device)

                out_det, out_mag, out_azm, reg_score, att_weights = model(img, cosmic)

                outputs = {
                    'stage1_logAnonymous Institution': out_det,
                    'stage2_mag': out_mag.max(dim=1)[0].unsqueeze(1),
                    'stage3_azm': out_azm,
                    'regional_score': reg_score,
                    'attention': att_weights
                }
                targets = {
                    'det': y_det.long(),
                    'mag': y_mag.float(),
                    'azm': y_azm
                }

                loss = criterion_hier(outputs, targets, adj_dummy)['total']
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.5f}")
        scheduler.step(avg_val_loss)

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/v3_best_fusion_model.pth')
            print("Checkpoint Saved!")

if __name__ == "__main__":
    print(">>> V3_Train.py: Main Entry Point Hit")
    parser = argparse.ArgumentParser(description="ScalogramV3 Training")
    parser.add_argument('--data_path', type=str, default=r"d:\multi\scalogramv3\scalogram_v3_cosmic.h5")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    print(f">>> Parameters: Batch={args.batch_size}, Epochs={args.epochs}")

    if os.path.exists(args.data_path):
        try:
            train_v3(args.data_path, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        except Exception as e:
            print(f"QUIT DUE TO ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Error: {args.data_path} not found.")
