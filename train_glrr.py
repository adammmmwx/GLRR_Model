import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("=== Launching High-Contrast GLRR Neural Training Engine ===")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matrix_path = r"C:\Users\awada\OneDrive\Desktop\GLRR_Model\glrr_training_matrix.npy"
weights_path = r"C:\Users\awada\OneDrive\Desktop\GLRR_Model\glrr_weights.pth"

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size=kernel_size, padding=padding)
        self.decoder = nn.Conv2d(in_channels=hidden_dim, out_channels=3, kernel_size=1)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, combined_conv.size(1) // 4, dim=1)
        i, f, o, g = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o), torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        output = self.decoder(h_next)
        return output, (h_next, c_next)

# CUSTOM HIGH-CONTRAST LOSS (Combines L1 + Spatial Gradient Penalties)
class HighContrastLoss(nn.Module):
    def __init__(self):
        super(HighContrastLoss, self).__init__()
        self.l1 = nn.L1Loss()
        
    def forward(self, pred, target):
        # 1. Base Absolute Pixel Error (Stops the blur)
        base_loss = self.l1(pred, target)
        
        # 2. Spatial Gradient Penalty (Forces sharp boundaries/edges)
        pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        grad_loss = self.l1(pred_grad_x, target_grad_x) + self.l1(pred_grad_y, target_grad_y)
        
        return base_loss + 0.5 * grad_loss

if not os.path.exists(matrix_path):
    print("[ERROR] Master training matrix missing, run harvester first!")
else:
    # Load Shape: (6, 6, 600, 1600, 3)
    master_matrix = np.load(matrix_path)
    num_events, seq_len, H, W, C = master_matrix.shape
    
    model = ConvLSTMCell(input_dim=3, hidden_dim=16, kernel_size=3).to(device)
    criterion = HighContrastLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    print("[OPTIMIZE] Commencing core model training loop across 15 epochs...")
    for epoch in range(15):
        epoch_loss = 0.0
        
        for e in range(num_events):
            event_sequence = master_matrix[e] # (6, 600, 1600, 3)
            tensor_seq = torch.from_numpy(event_sequence).float().permute(0, 3, 1, 2).to(device)
            if tensor_seq.max() > 1.0: tensor_seq /= 255.0
            
            h_state = torch.zeros(1, 16, H, W).to(device)
            c_state = torch.zeros(1, 16, H, W).to(device)
            
            optimizer.zero_grad()
            
            # Step through first 5 frames to generate final prediction
            for t in range(5):
                current_frame = tensor_seq[t].unsqueeze(0)
                predicted_output, (h_state, c_state) = model(current_frame, (h_state, c_state))
                
            # Compare final 6th hour forecast to the actual ground truth target
            target_frame = tensor_seq[5].unsqueeze(0)
            loss = criterion(predicted_output, target_frame)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f" └─> Epoch [{epoch+1:02d}/15] ┊ Integrated Sequence Loss: {epoch_loss / num_events:.6f}")

    torch.save(model.state_dict(), weights_path)
    print("\n==============================================")
    print(f"🥇 [SUCCESS] High-contrast weights saved to: {weights_path}")

input("\n[PROCESS FINISHED] Press Enter to exit...")