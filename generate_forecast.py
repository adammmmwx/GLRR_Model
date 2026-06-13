import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

print("=== Running Upgraded High-Res GLRR Forecast Render ===")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = r"C:\Users\awada\OneDrive\Desktop\GLRR_Model\glrr_training_matrix.npy"
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

if not os.path.exists(dataset_path) or not os.path.exists(weights_path):
    print("[ERROR] Required training matrix or trained weights missing, wallahi!")
else:
    # 1. Load Master Matrix: (Events, Time, H, W, C) -> (6, 6, 600, 1600, 3)
    master_data = np.load(dataset_path)
    
    # 2. Isolate event index 3 (The Feb 18th Surprise Thundersquall!)
    # This brings us down to a clean 4D array: (6, 600, 1600, 3)
    target_event_sequence = master_data[3]
    
    # 3. Separate into 5 input frames and 1 ground truth verification frame
    raw_input = target_event_sequence[:5]
    gt_frame = target_event_sequence[5]
    
    tensor_data = torch.from_numpy(raw_input).float().permute(0, 3, 1, 2).to(device)
    if tensor_data.max() > 1.0: tensor_data /= 255.0

    model = ConvLSTMCell(input_dim=3, hidden_dim=16, kernel_size=3).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    h_state = torch.zeros(1, 16, tensor_data.shape[2], tensor_data.shape[3]).to(device)
    c_state = torch.zeros(1, 16, tensor_data.shape[2], tensor_data.shape[3]).to(device)

    print("[COMPUTE] Passing February thundersquall data through optimized neural weights...")
    with torch.no_grad():
        for t in range(tensor_data.size(0)):
            current_frame = tensor_data[t].unsqueeze(0)
            predicted_output, (h_state, c_state) = model(current_frame, (h_state, c_state))

    ai_prediction = torch.clamp(predicted_output.squeeze(0).permute(1, 2, 0), 0.0, 1.0).cpu().numpy()

    print("[RENDER] Building Clean Weather Canvas Comparison...")
    # High-Res coordinate tracking space matching the new harvester grid bounds
    lon_west, lon_east, lat_south, lat_north = -90.0, -74.0, 41.0, 47.0
    map_proj = ccrs.PlateCarree()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9), subplot_kw={'projection': map_proj})
    
    for ax, title, data in zip([ax1, ax2], ["AI Predicted Radar", "Actual Real-World Radar"], [ai_prediction, gt_frame]):
        ax.set_extent([lon_west, lon_east, lat_south, lat_north], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='white', linewidth=1.2)
        ax.add_feature(cfeature.LAKES.with_scale('50m'), facecolor='none', edgecolor='cyan', linewidth=1.5)
        ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.6)
        
        ax.imshow(data, origin='upper', extent=[lon_west, lon_east, lat_south, lat_north], transform=ccrs.PlateCarree())
        ax.set_title(title, fontsize=14, fontweight='bold')

    plt.suptitle("GLRR High-Res Model Testing | Feb 18 Surprise Thundersquall Event", fontsize=18, fontweight='bold', y=0.92)
    
    output_filename = r"C:\Users\awada\OneDrive\Desktop\GLRR_Model\glrr_ai_forecast_prediction.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n==============================================")
    print(f"🥇 [SUCCESS] High-res comparison frame saved to: {output_filename}")

input("\n[PROCESS FINISHED] Press Enter to exit...")