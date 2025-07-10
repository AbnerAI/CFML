import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

# Set matplotlib font to avoid display issues
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # Correctly display negative signs
plt.rcParams['pdf.fonttype'] = 42  # Ensure fonts in PDF are editable
plt.rcParams['ps.fonttype'] = 42   # PostScript font type
plt.rcParams['savefig.dpi'] = 300  # High resolution

# IC mapping table: display number -> real IC number
# Note that if the brain region order of the data is 
# not consistent with the template, the mapping relationship is provided here. If it is consistent, it can be ignored.

IC_MAPPING = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20,
    21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30,
    31: 31, 32: 32, 33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40,
    41: 41, 42: 42, 43: 43, 44: 44, 45: 45, 46: 46, 47: 47, 48: 48, 49: 49, 50: 50
}

# Global configuration variables
GLOBAL_CONFIG = {
    'attention_dir': "/mnt/data/home/cxx/PycharmProjects/pythonProject/CFML-main/workspace/fusion/2025-07-08-11-34-56/attention_weights",
    'repeat_idx': 1
}

def load_attention_weights(attention_dir, repeat_idx, fold_idx, debug=False):
    """Load attention weights for all epochs of specified repeat and fold"""
    pattern = f'attention_epoch_*_repeat_{repeat_idx}_fold_{fold_idx}.npy'
    file_pattern = os.path.join(attention_dir, pattern)
    
    if debug:
        print(f"Debug: Loading fold {fold_idx}, pattern: {file_pattern}")
    
    files = glob.glob(file_pattern)
    
    if debug:
        print(f"Debug: Found {len(files)} files for fold {fold_idx}")
        if len(files) > 0:
            for i, f in enumerate(files[:3]):
                print(f"  {i+1}. {os.path.basename(f)}")
    
    if not files:
        error_msg = f"No matching files found: {file_pattern}"
        if debug:
            all_files = glob.glob(os.path.join(attention_dir, "*.npy"))
            matching_repeat = [f for f in all_files if f"repeat_{repeat_idx}" in f]
            print(f"Debug: Files with repeat_{repeat_idx}: {len(matching_repeat)}")
        raise ValueError(error_msg)
    
    epoch_data = []
    for file in files:
        filename = os.path.basename(file)
        match = re.search(r'attention_epoch_(\d+)_', filename)
        if match:
            epoch_num = int(match.group(1))
            attention_weights = np.load(file)
            if attention_weights.ndim == 2:
                attention_weights = attention_weights.squeeze()
            epoch_data.append((epoch_num, attention_weights))
    
    if not epoch_data:
        raise ValueError(f"No valid epoch data found for fold {fold_idx}")
    
    epoch_data.sort(key=lambda x: x[0])
    epochs = [data[0] for data in epoch_data]
    attention_matrix = np.array([data[1] for data in epoch_data])
    
    if debug:
        print(f"Debug: Fold {fold_idx} loaded successfully, shape: {attention_matrix.shape}")
    
    return attention_matrix, epochs

def get_available_folds(attention_dir, repeat_idx, debug=False):
    """Get all available folds for the specified repeat"""
    pattern = f'attention_epoch_*_repeat_{repeat_idx}_fold_*.npy'
    file_pattern = os.path.join(attention_dir, pattern)
    files = glob.glob(file_pattern)
    
    if debug:
        print(f"Debug: Searching pattern: {file_pattern}")
        print(f"Debug: Found {len(files)} files")
        if len(files) == 0:
            if os.path.exists(attention_dir):
                all_files = os.listdir(attention_dir)
                npy_files = [f for f in all_files if f.endswith('.npy')]
                print(f"Debug: Directory contains {len(npy_files)} .npy files")
                if npy_files:
                    print("Debug: Sample .npy files:")
                    for i, f in enumerate(npy_files[:5]):
                        print(f"  {i+1}. {f}")
    
    folds = set()
    for file in files:
        filename = os.path.basename(file)
        match = re.search(r'fold_(\d+)\.npy', filename)
        if match:
            fold_num = int(match.group(1))
            folds.add(fold_num)
    
    available_folds = sorted(list(folds))
    if debug:
        print(f"Debug: Final available folds: {available_folds}")
    
    return available_folds

def manual_fold_input():
    """Manual input of fold information"""
    print("\n=== Manual Fold Input ===")
    try:
        fold_input = input("Enter available fold numbers separated by commas (e.g., 1,2,3,4,5,6,7,8,9,10): ").strip()
        available_folds = [int(x.strip()) for x in fold_input.split(',')]
        available_folds = sorted(list(set(available_folds)))
        print(f"You specified {len(available_folds)} folds: {available_folds}")
        return available_folds
    except ValueError:
        print("Invalid input format. Please use numbers separated by commas.")
        return None

def visualize_attention_evolution(attention_matrix, epochs, repeat_idx, fold_idx, 
                                save_path=None, figsize=(15, 8), 
                                x_tick_num=None, show_all_ics=False, 
                                x_label_rotation=45, x_label_fontsize=8):
    """Visualize attention evolution across epochs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Flip matrix so epochs increase from bottom to top
    attention_flipped = np.flipud(attention_matrix)
    epochs_flipped = epochs[::-1]
    
    im1 = ax1.imshow(attention_flipped, aspect='auto', cmap='viridis', interpolation='nearest')
    ax1.set_xlabel('IC Components (Brain Regions)', fontsize=12)
    ax1.set_ylabel('Training Epochs (Bottom to Top)', fontsize=12)
    ax1.set_title(f'Attention Evolution - Repeat {repeat_idx}, Fold {fold_idx}', fontsize=14)
    
    # Set y-axis ticks
    n_epochs = len(epochs)
    step = max(1, n_epochs // 10)
    y_ticks = list(range(0, n_epochs, step))
    y_labels = [str(epochs_flipped[i]) for i in y_ticks]
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels)
    
    # Set x-axis ticks
    n_components = attention_matrix.shape[1]
    
    if show_all_ics:
        x_ticks = list(range(n_components))
        # Show real IC numbers
        x_labels = []
        for i in x_ticks:
            display_number = i + 1  # Display number (1-based)
            real_ic = IC_MAPPING.get(display_number, display_number)  # Real IC number
            x_labels.append(f'{real_ic}')
    else:
        if x_tick_num is None:
            x_tick_num = min(20, max(10, n_components // 2))
        
        x_tick_num = min(x_tick_num, n_components)
        x_step = max(1, n_components // x_tick_num)
        x_ticks = list(range(0, n_components, x_step))
        
        if x_ticks[-1] != n_components - 1:
            x_ticks.append(n_components - 1)
        
        # Show real IC numbers
        x_labels = []
        for i in x_ticks:
            display_number = i + 1  # Display number (1-based)
            real_ic = IC_MAPPING.get(display_number, display_number)  # Real IC number
            x_labels.append(f'{real_ic}')
    
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels, rotation=x_label_rotation, fontsize=x_label_fontsize, ha='right')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Attention Weight', fontsize=10)
    
    # Select IC components with the highest variance to show trends
    attention_var = np.var(attention_matrix, axis=0)
    top_varying_ics = np.argsort(attention_var)[-5:]
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(top_varying_ics)))
    
    for i, ic_idx in enumerate(top_varying_ics):
        # Get real IC number for legend display
        display_number = ic_idx + 1  # Display number (1-based)
        real_ic = IC_MAPPING.get(display_number, display_number)  # Real IC number
        
        ax2.plot(epochs, attention_matrix[:, ic_idx], 
                color=colors[i], linewidth=2, marker='o', markersize=3,
                label=f'IC{real_ic} (var={attention_var[ic_idx]:.4f})')
    
    ax2.set_xlabel('Training Epochs', fontsize=12)
    ax2.set_ylabel('Attention Weight', fontsize=12)
    ax2.set_title('Top 5 Most Variable IC Components', fontsize=14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    else:
        plt.show()

def analyze_attention_statistics(attention_matrix, epochs):
    """Analyze attention statistics"""
    print("=== Attention Statistics ===")
    print(f"Epochs: {len(epochs)} (from {epochs[0]} to {epochs[-1]})")
    print(f"IC Components: {attention_matrix.shape[1]}")
    print(f"Attention weight range: [{attention_matrix.min():.4f}, {attention_matrix.max():.4f}]")
    print(f"Mean attention: {attention_matrix.mean():.4f}")
    print(f"Std attention: {attention_matrix.std():.4f}")
    
    mean_attention = np.mean(attention_matrix, axis=0)
    top_ics = np.argsort(mean_attention)[-5:]
    print(f"\nTop 5 Most Important IC Components (by mean attention):")
    for i, ic_idx in enumerate(top_ics[::-1]):
        print(f"  {i+1}. IC{ic_idx+1}: {mean_attention[ic_idx]:.4f}")
    
    attention_var = np.var(attention_matrix, axis=0)
    top_varying_ics = np.argsort(attention_var)[-5:]
    print(f"\nTop 5 Most Variable IC Components:")
    for i, ic_idx in enumerate(top_varying_ics[::-1]):
        print(f"  {i+1}. IC{ic_idx+1}: variance={attention_var[ic_idx]:.6f}")

def load_multiple_folds_attention(attention_dir, repeat_idx, selected_folds, debug=True):
    """Load attention weights for multiple folds and compute average"""
    print(f"Loading attention data for folds: {selected_folds}")
    
    fold_data = {}
    min_epochs = float('inf')
    
    for fold_idx in selected_folds:
        try:
            print(f"  Loading fold {fold_idx}...")
            attention_matrix, epochs = load_attention_weights(attention_dir, repeat_idx, fold_idx, debug=debug)
            fold_data[fold_idx] = {
                'attention': attention_matrix,
                'epochs': epochs,
                'n_epochs': len(epochs)
            }
            min_epochs = min(min_epochs, len(epochs))
            print(f"  Fold {fold_idx}: {len(epochs)} epochs, shape {attention_matrix.shape}")
        except Exception as e:
            print(f"  Warning: Could not load fold {fold_idx}: {e}")
    
    if not fold_data:
        raise ValueError("No valid fold data found")
    
    print(f"Successfully loaded {len(fold_data)} folds")
    print(f"Common epoch range: {min_epochs} epochs")
    
    # Find common epochs
    first_fold = list(fold_data.values())[0]
    common_epochs = first_fold['epochs'][:min_epochs]
    
    # Align all fold data
    aligned_attention = []
    for fold_idx, data in fold_data.items():
        truncated_attention = data['attention'][:min_epochs]
        aligned_attention.append(truncated_attention)
    
    aligned_attention = np.array(aligned_attention)
    mean_attention = np.mean(aligned_attention, axis=0)
    std_attention = np.std(aligned_attention, axis=0)
    
    print(f"Computed average attention across {len(selected_folds)} folds")
    print(f"Final attention matrix shape: {mean_attention.shape}")
    
    return mean_attention, std_attention, common_epochs, fold_data

def visualize_average_attention(mean_attention, std_attention, common_epochs, 
                              repeat_idx, selected_folds, 
                              save_path=None, figsize=(18, 8), 
                              x_tick_num=None, show_all_ics=False, 
                              x_label_rotation=45, x_label_fontsize=8,
                              show_std=True, save_individual=True, save_format="both"):
    """
    Visualize average attention across multiple folds
    
    Args:
        save_individual: Whether to save each subplot separately
        save_format: Save format ("png", "pdf", "both")
    """
    
    # Determine save format list based on save_format
    if save_format == "png":
        formats = [".png"]
    elif save_format == "pdf":
        formats = [".pdf"]
    elif save_format == "both":
        formats = [".png", ".pdf"]
    else:
        formats = [".png"]  # Default PNG
    
    if show_std:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        axes = [ax1, ax2, ax3]
        subplot_names = ["heatmap", "trends", "std"]
        subplot_titles = [
            f'Average Attention - Repeat {repeat_idx}',
            'Top 5 Most Important IC Components',
            'Attention Standard Deviation'
        ]
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*2/3, figsize[1]))
        axes = [ax1, ax2]
        subplot_names = ["heatmap", "trends"]
        subplot_titles = [
            f'Average Attention - Repeat {repeat_idx}',
            'Top 5 Most Important IC Components'
        ]
    
    # Prepare basic data
    mean_flipped = np.flipud(mean_attention)
    epochs_flipped = common_epochs[::-1]
    
    # Set axis parameters
    n_epochs = len(common_epochs)
    step = max(1, n_epochs // 10)
    y_ticks = list(range(0, n_epochs, step))
    y_labels = [str(epochs_flipped[i]) for i in y_ticks]
    
    n_components = mean_attention.shape[1]
    if show_all_ics:
        x_ticks = list(range(n_components))
        # Show real IC numbers
        x_labels = []
        for i in x_ticks:
            display_number = i + 1  # Display number (1-based)
            real_ic = IC_MAPPING.get(display_number, display_number)  # Real IC number
            x_labels.append(f'{real_ic}')
    else:
        if x_tick_num is None:
            x_tick_num = min(20, max(10, n_components // 2))
        x_tick_num = min(x_tick_num, n_components)
        x_step = max(1, n_components // x_tick_num)
        x_ticks = list(range(0, n_components, x_step))
        if x_ticks[-1] != n_components - 1:
            x_ticks.append(n_components - 1)
        # Show real IC numbers
        x_labels = []
        for i in x_ticks:
            display_number = i + 1  # Display number (1-based)
            real_ic = IC_MAPPING.get(display_number, display_number)  # Real IC number
            x_labels.append(f'{real_ic}')
    
    # 1. Average Attention Heatmap
    im1 = ax1.imshow(mean_flipped, aspect='auto', cmap='viridis', interpolation='nearest')
    ax1.set_xlabel('IC Components (Brain Regions)', fontsize=12)
    ax1.set_ylabel('Training Epochs (Bottom to Top)', fontsize=12)
    
    fold_str = f"{len(selected_folds)} folds"
    ax1.set_title(f'Average Attention - Repeat {repeat_idx} ({fold_str})', fontsize=14)
    
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels, rotation=x_label_rotation, fontsize=x_label_fontsize, ha='right')
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Mean Attention Weight', fontsize=10)
    
    # 2. Most important IC component trend plot
    mean_attention_by_ic = np.mean(mean_attention, axis=0)
    top_important_ics = np.argsort(mean_attention_by_ic)[-5:]
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(top_important_ics)))
    
    for i, ic_idx in enumerate(top_important_ics):
        mean_vals = mean_attention[:, ic_idx]
        std_vals = std_attention[:, ic_idx]
        
        # Get real IC number for legend display
        display_number = ic_idx + 1  # Display number (1-based)
        real_ic = IC_MAPPING.get(display_number, display_number)  # Real IC number
        
        ax2.plot(common_epochs, mean_vals, 
                color=colors[i], linewidth=2, marker='o', markersize=3,
                label=f'IC{real_ic} (mean={mean_attention_by_ic[ic_idx]:.4f})')
        
        ax2.fill_between(common_epochs, 
                        mean_vals - std_vals, 
                        mean_vals + std_vals,
                        color=colors[i], alpha=0.2)
    
    ax2.set_xlabel('Training Epochs', fontsize=12)
    ax2.set_ylabel('Attention Weight', fontsize=12)
    ax2.set_title('Top 5 Most Important IC Components\n(with std deviation)', fontsize=14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Standard deviation Heatmap
    if show_std:
        std_flipped = np.flipud(std_attention)
        
        im3 = ax3.imshow(std_flipped, aspect='auto', cmap='Reds', interpolation='nearest')
        ax3.set_xlabel('IC Components (Brain Regions)', fontsize=12)
        ax3.set_ylabel('Training Epochs (Bottom to Top)', fontsize=12)
        ax3.set_title('Attention Standard Deviation\nAcross Folds', fontsize=14)
        
        ax3.set_yticks(y_ticks)
        ax3.set_yticklabels(y_labels)
        ax3.set_xticks(x_ticks)
        ax3.set_xticklabels(x_labels, rotation=x_label_rotation, fontsize=x_label_fontsize, ha='right')
        
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
        cbar3.set_label('Std Attention Weight', fontsize=10)
    
    plt.tight_layout()
    
    # Save combined plot
    if save_path:
        for fmt in formats:
            combined_path = save_path.replace('.png', '').replace('.pdf', '') + f'_combined{fmt}'
            plt.savefig(combined_path, dpi=300, bbox_inches='tight')
            print(f"Combined attention image saved to: {combined_path}")
    
    # Save each subplot separately
    if save_individual and save_path:
        base_path = save_path.replace('.png', '').replace('.pdf', '')
        fold_str = "_".join(map(str, selected_folds))
        
        for i, ax in enumerate(axes):
            # Create separate figure for each subplot
            individual_fig = plt.figure(figsize=(8, 6))
            individual_ax = individual_fig.add_subplot(111)
            
            if i == 0:  # Heatmap
                im = individual_ax.imshow(mean_flipped, aspect='auto', cmap='viridis', interpolation='nearest')
                individual_ax.set_xlabel('IC Components (Brain Regions)', fontsize=12)
                individual_ax.set_ylabel('Training Epochs (Bottom to Top)', fontsize=12)
                individual_ax.set_title(f'Average Attention Heatmap - Repeat {repeat_idx} ({fold_str})', fontsize=14)
                
                individual_ax.set_yticks(y_ticks)
                individual_ax.set_yticklabels(y_labels)
                individual_ax.set_xticks(x_ticks)
                individual_ax.set_xticklabels(x_labels, rotation=x_label_rotation, fontsize=x_label_fontsize, ha='right')
                
                cbar = plt.colorbar(im, ax=individual_ax, shrink=0.8)
                cbar.set_label('Mean Attention Weight', fontsize=10)
            
            elif i == 1:  # Trends
                for j, ic_idx in enumerate(top_important_ics):
                    mean_vals = mean_attention[:, ic_idx]
                    std_vals = std_attention[:, ic_idx]
                    
                    # Get real IC number for legend display
                    display_number = ic_idx + 1  # Display number (1-based)
                    real_ic = IC_MAPPING.get(display_number, display_number)  # Real IC number
                    
                    individual_ax.plot(common_epochs, mean_vals, 
                            color=colors[j], linewidth=2, marker='o', markersize=3,
                            label=f'IC{real_ic} (mean={mean_attention_by_ic[ic_idx]:.4f})')
                    
                    individual_ax.fill_between(common_epochs, 
                                    mean_vals - std_vals, 
                                    mean_vals + std_vals,
                                    color=colors[j], alpha=0.2)
                
                individual_ax.set_xlabel('Training Epochs', fontsize=12)
                individual_ax.set_ylabel('Attention Weight', fontsize=12)
                individual_ax.set_title(f'Top 5 Most Important IC Components - Repeat {repeat_idx} ({fold_str})', fontsize=14)
                individual_ax.legend(loc='best')
                individual_ax.grid(True, alpha=0.3)
            
            elif i == 2 and show_std:  # Std Heatmap
                im = individual_ax.imshow(std_flipped, aspect='auto', cmap='Reds', interpolation='nearest')
                individual_ax.set_xlabel('IC Components (Brain Regions)', fontsize=12)
                individual_ax.set_ylabel('Training Epochs (Bottom to Top)', fontsize=12)
                individual_ax.set_title(f'Attention Standard Deviation - Repeat {repeat_idx} ({fold_str})', fontsize=14)
                
                individual_ax.set_yticks(y_ticks)
                individual_ax.set_yticklabels(y_labels)
                individual_ax.set_xticks(x_ticks)
                individual_ax.set_xticklabels(x_labels, rotation=x_label_rotation, fontsize=x_label_fontsize, ha='right')
                
                cbar = plt.colorbar(im, ax=individual_ax, shrink=0.8)
                cbar.set_label('Std Attention Weight', fontsize=10)
            
            plt.tight_layout()
            
            # Save individual plots
            for fmt in formats:
                individual_path = f"{base_path}_{subplot_names[i]}{fmt}"
                individual_fig.savefig(individual_path, dpi=300, bbox_inches='tight')
                print(f"Individual {subplot_names[i]} saved to: {individual_path}")
            
            plt.close(individual_fig)  # Close individual figure to free memory
    
    # Display or save main plot
    if not save_path:
        plt.show()
    else:
        plt.close(fig)  # Close main figure

def analyze_average_attention_statistics(mean_attention, std_attention, common_epochs, selected_folds, save_ranking=True):
    """Analyze average attention statistics"""
    print("=== Average Attention Statistics ===")
    print(f"Folds analyzed: {len(selected_folds)} ({selected_folds})")
    print(f"Common epochs: {len(common_epochs)} (from {common_epochs[0]} to {common_epochs[-1]})")
    print(f"IC Components: {mean_attention.shape[1]}")
    print(f"Mean attention range: [{mean_attention.min():.4f}, {mean_attention.max():.4f}]")
    print(f"Overall mean attention: {mean_attention.mean():.4f}")
    print(f"Overall std attention: {mean_attention.std():.4f}")
    
    mean_std_by_ic = np.mean(std_attention, axis=0)
    print(f"Average std across folds: {mean_std_by_ic.mean():.4f}")
    
    # Calculate mean attention weight for each IC
    mean_attention_by_ic = np.mean(mean_attention, axis=0)
    
    # Most important IC components (show real IC numbers)
    top_ics = np.argsort(mean_attention_by_ic)[-5:]
    print(f"\nTop 5 Most Important IC Components (by mean attention):")
    for i, ic_idx in enumerate(top_ics[::-1]):
        display_number = ic_idx + 1  # Display number (1-based)
        real_ic = IC_MAPPING.get(display_number, display_number)  # Real IC number
        mean_val = mean_attention_by_ic[ic_idx]
        std_val = mean_std_by_ic[ic_idx]
        print(f"  {i+1}. Display IC{display_number} (Real IC{real_ic}): mean={mean_val:.4f}, std_across_folds={std_val:.4f}")
    
    # Most stable IC components (show real IC numbers)
    most_stable_ics = np.argsort(mean_std_by_ic)[:5]
    print(f"\nTop 5 Most Stable IC Components (lowest std across folds):")
    for i, ic_idx in enumerate(most_stable_ics):
        display_number = ic_idx + 1  # Display number (1-based)
        real_ic = IC_MAPPING.get(display_number, display_number)  # Real IC number
        mean_val = mean_attention_by_ic[ic_idx]
        std_val = mean_std_by_ic[ic_idx]
        print(f"  {i+1}. Display IC{display_number} (Real IC{real_ic}): mean={mean_val:.4f}, std_across_folds={std_val:.4f}")
    
    # Save complete IC ranking results to txt file
    if save_ranking:
        # Sort by mean attention weight from high to low
        all_ics_ranked = np.argsort(mean_attention_by_ic)[::-1]  # High to low
        
        fold_str = "_".join(map(str, selected_folds))
        ranking_filename = f"ic_ranking_average_folds_{fold_str}.txt"
        
        with open(ranking_filename, 'w', encoding='utf-8') as f:
            f.write("# IC Ranking Results (Average across Multiple Folds)\n")
            f.write(f"# Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Folds Analyzed: {selected_folds}\n")
            f.write(f"# Total Folds: {len(selected_folds)}\n")
            f.write(f"# Common Epochs: {len(common_epochs)} (from {common_epochs[0]} to {common_epochs[-1]})\n")
            f.write(f"# Total IC Components: {len(mean_attention_by_ic)}\n")
            f.write("#\n")
            f.write("# Format: Rank, Display_IC, Real_IC, Mean_Attention, Std_Across_Folds, Variance_Across_Epochs\n")
            f.write("# Display_IC: IC number shown in visualization (1-50)\n")
            f.write("# Real_IC: Actual IC number in the brain template\n")
            f.write("#\n")
            
            # Calculate variance for each IC across the time dimension (reflects changes during training)
            temporal_variance = np.var(mean_attention, axis=0)
            
            for rank, ic_idx in enumerate(all_ics_ranked, 1):
                display_number = ic_idx + 1  # Display number (1-based)
                real_ic = IC_MAPPING.get(display_number, display_number)  # Real IC number
                mean_val = mean_attention_by_ic[ic_idx]
                std_val = mean_std_by_ic[ic_idx]
                temp_var = temporal_variance[ic_idx]
                f.write(f"{rank:3d}, {display_number:3d}, {real_ic:3d}, {mean_val:.6f}, {std_val:.6f}, {temp_var:.6f}\n")
        
        print(f"\n✅ IC ranking results saved to: {ranking_filename}")
        print(f"   - Total ICs ranked: {len(all_ics_ranked)}")
        print(f"   - Ranking criteria: Mean attention weight across {len(selected_folds)} folds")
        print(f"   - File includes: Rank, Display IC, Real IC, Mean Attention, Cross-fold Std, Temporal Variance")
        
        # Also save a simplified version (top 20 only)
        top20_filename = f"ic_ranking_top20_folds_{fold_str}.txt"
        with open(top20_filename, 'w', encoding='utf-8') as f:
            f.write("# Top 20 Most Important IC Components (Average across Multiple Folds)\n")
            f.write(f"# Folds: {selected_folds}\n")
            f.write("# Format: Rank, Display_IC, Real_IC, Mean_Attention, Std_Across_Folds\n")
            f.write("# Display_IC: IC number in visualization, Real_IC: Actual brain template IC\n")
            f.write("#\n")
            
            for rank, ic_idx in enumerate(all_ics_ranked[:20], 1):
                display_number = ic_idx + 1  # Display number (1-based)
                real_ic = IC_MAPPING.get(display_number, display_number)  # Real IC number
                mean_val = mean_attention_by_ic[ic_idx]
                std_val = mean_std_by_ic[ic_idx]
                f.write(f"{rank:2d}, {display_number:3d}, {real_ic:3d}, {mean_val:.6f}, {std_val:.6f}\n")
        
        print(f"✅ Top 20 IC ranking saved to: {top20_filename}")
        
        # Create file sorted by real IC numbers
        real_ic_filename = f"ic_ranking_by_real_ic_folds_{fold_str}.txt"
        with open(real_ic_filename, 'w', encoding='utf-8') as f:
            f.write("# IC Ranking Results (Sorted by Real IC Numbers)\n")
            f.write(f"# Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Folds: {selected_folds}\n")
            f.write("# Format: Real_IC, Display_IC, Rank, Mean_Attention, Std_Across_Folds\n")
            f.write("#\n")
            
            # Create list sorted by real IC numbers
            real_ic_data = []
            for rank, ic_idx in enumerate(all_ics_ranked, 1):
                display_number = ic_idx + 1
                real_ic = IC_MAPPING.get(display_number, display_number)
                mean_val = mean_attention_by_ic[ic_idx]
                std_val = mean_std_by_ic[ic_idx]
                real_ic_data.append((real_ic, display_number, rank, mean_val, std_val))
            
            # Sort by real IC number
            real_ic_data.sort(key=lambda x: x[0])
            
            for real_ic, display_number, rank, mean_val, std_val in real_ic_data:
                f.write(f"{real_ic:3d}, {display_number:3d}, {rank:3d}, {mean_val:.6f}, {std_val:.6f}\n")
        
        # Also save file sorted by importance (showing real IC numbers)
        importance_filename = f"ic_ranking_by_importance_real_ic_folds_{fold_str}.txt"
        with open(importance_filename, 'w', encoding='utf-8') as f:
            f.write("# IC Ranking by Importance (Real IC Numbers)\n")
            f.write(f"# Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Folds: {selected_folds}\n")
            f.write("# Format: Rank, Real_IC, Mean_Attention, Std_Across_Folds\n")
            f.write("# Sorted by importance (highest attention first)\n")
            f.write("#\n")
            
            for rank, ic_idx in enumerate(all_ics_ranked, 1):
                display_number = ic_idx + 1
                real_ic = IC_MAPPING.get(display_number, display_number)
                mean_val = mean_attention_by_ic[ic_idx]
                std_val = mean_std_by_ic[ic_idx]
                f.write(f"{rank:3d}, {real_ic:3d}, {mean_val:.6f}, {std_val:.6f}\n")
        
        print(f"✅ Importance-based ranking (Real IC) saved to: {importance_filename}")
        
        return {
            'ranking_full': all_ics_ranked,
            'mean_attention': mean_attention_by_ic,
            'std_attention': mean_std_by_ic,
            'temporal_variance': temporal_variance,
            'files_saved': [ranking_filename, top20_filename, real_ic_filename, importance_filename]
        }

def visualize_multiple_folds_comparison(attention_dir, repeat_idx, selected_folds=None, 
                                       x_tick_num=15, show_all_ics=False):
    """Visualize attention comparison across multiple folds"""
    if selected_folds is None:
        available_folds = get_available_folds(attention_dir, repeat_idx)
        print(f"\nAvailable folds: {available_folds}")
        
        # Let user select multiple folds
        print("Please select folds to compare (separate with commas, e.g.: 1,2,3):")
        fold_input = input("Enter fold numbers: ").strip()
        
        try:
            selected_folds = [int(x.strip()) for x in fold_input.split(',')]
            # Validate fold availability
            invalid_folds = [f for f in selected_folds if f not in available_folds]
            if invalid_folds:
                print(f"Warning: folds {invalid_folds} not available, will be ignored")
                selected_folds = [f for f in selected_folds if f in available_folds]
        except ValueError:
            print("Invalid input format, using default folds")
            selected_folds = available_folds[:min(6, len(available_folds))]
    
    if not selected_folds:
        print("No available fold data")
        return
    
    # Dynamically determine subplot layout
    n_folds = len(selected_folds)
    if n_folds <= 3:
        rows, cols = 1, n_folds
        figsize = (6*n_folds, 5)
    elif n_folds <= 6:
        rows, cols = 2, 3
        figsize = (18, 10)
    else:
        rows, cols = 3, 3
        figsize = (18, 15)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_folds == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for i, fold_idx in enumerate(selected_folds):
        if i >= len(axes):
            break
            
        try:
            attention_matrix, epochs = load_attention_weights(attention_dir, repeat_idx, fold_idx)
            
            # Flip matrix so epochs increase from bottom to top
            attention_flipped = np.flipud(attention_matrix)
            epochs_flipped = epochs[::-1]
            
            im = axes[i].imshow(attention_flipped, aspect='auto', cmap='viridis', interpolation='nearest')
            axes[i].set_title(f'Fold {fold_idx}', fontsize=12)
            axes[i].set_xlabel('IC Components')
            axes[i].set_ylabel('Epochs')
            
            # Set y-axis ticks
            n_epochs = len(epochs)
            step = max(1, n_epochs // 5)
            y_ticks = list(range(0, n_epochs, step))
            y_labels = [str(epochs_flipped[j]) for j in y_ticks]
            axes[i].set_yticks(y_ticks)
            axes[i].set_yticklabels(y_labels)
            
            # Set x-axis ticks (using new custom functionality)
            n_components = attention_matrix.shape[1]
            
            if show_all_ics:
                x_ticks = list(range(n_components))
                # Show real IC numbers
                x_labels = []
                for j in x_ticks:
                    display_number = j + 1  # Display number (1-based)
                    real_ic = IC_MAPPING.get(display_number, display_number)  # Real IC number
                    x_labels.append(f'{real_ic}')
                rotation = 90
                fontsize = 6
            else:
                x_tick_num_actual = min(x_tick_num, n_components)
                x_step = max(1, n_components // x_tick_num_actual)
                x_ticks = list(range(0, n_components, x_step))
                if x_ticks[-1] != n_components - 1:
                    x_ticks.append(n_components - 1)
                # Show real IC numbers
                x_labels = []
                for j in x_ticks:
                    display_number = j + 1  # Display number (1-based)
                    real_ic = IC_MAPPING.get(display_number, display_number)  # Real IC number
                    x_labels.append(f'{real_ic}')
                rotation = 45
                fontsize = 8
            
            axes[i].set_xticks(x_ticks)
            axes[i].set_xticklabels(x_labels, rotation=rotation, fontsize=fontsize, ha='right')
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f'No data\nfor fold {fold_idx}\n{str(e)}', 
                        ha='center', va='center', transform=axes[i].transAxes)
    
    # Hide excess subplots
    for i in range(len(selected_folds), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    save_name = f'attention_comparison_repeat{repeat_idx}_folds_{"_".join(map(str, selected_folds))}.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Multi-fold comparison saved: {save_name}")
    plt.show()

def main():
    """Main function"""
    print("=== CFML Attention Visualization Tool ===")
    
    # Configuration settings
    attention_dir = GLOBAL_CONFIG['attention_dir']
    repeat_idx = GLOBAL_CONFIG['repeat_idx']
    
    print(f"Current settings:")
    print(f"  attention_dir: {attention_dir}")
    print(f"  repeat_idx: {repeat_idx}")
    
    modify_settings = input("\nDo you want to modify these settings? (y/n, default n): ").strip().lower()
    if modify_settings == 'y':
        attention_dir = input(f"Enter attention directory path: ").strip() or attention_dir
        repeat_idx = int(input(f"Enter repeat index (default {repeat_idx}): ").strip() or str(repeat_idx))
        print(f"Updated settings: attention_dir={attention_dir}, repeat_idx={repeat_idx}")
    
    print("\n1. Single fold detailed analysis")
    print("2. Multiple folds comparison")
    print("3. Auto analyze all available folds")
    print("4. Average attention across multiple folds")
    
    choice = input("\nPlease select function (1/2/3/4): ").strip()
    
    try:
        if choice == "1":
            # Single fold analysis
            print(f"\n=== Single Fold Detailed Analysis ===")
            available_folds = get_available_folds(attention_dir, repeat_idx, debug=True)
            print(f"Available folds: {available_folds}")
            
            if len(available_folds) == 0:
                print("No folds found. Try manual input:")
                available_folds = manual_fold_input()
                if not available_folds:
                    return
            
            # Select fold
            print("\n=== Available Folds ===")
            for i, fold in enumerate(available_folds):
                print(f"{i+1}. Fold {fold}")
            
            while True:
                try:
                    choice_input = input(f"Select fold (1-{len(available_folds)} or fold number): ").strip()
                    choice_num = int(choice_input)
                    if 1 <= choice_num <= len(available_folds):
                        fold_idx = available_folds[choice_num - 1]
                        break
                    elif choice_num in available_folds:
                        fold_idx = choice_num
                        break
                    else:
                        print("Invalid input!")
                except ValueError:
                    print("Invalid input!")
            
            # Load and analyze
            attention_matrix, epochs = load_attention_weights(attention_dir, repeat_idx, fold_idx, debug=True)
            analyze_attention_statistics(attention_matrix, epochs)
            
            # Visualize
            save_path = f"attention_evolution_repeat{repeat_idx}_fold{fold_idx}.png"
            visualize_attention_evolution(attention_matrix, epochs, repeat_idx, fold_idx, 
                                        save_path=save_path, show_all_ics=True, figsize=(20, 8))
        
        elif choice == "2":
            # === Multiple fold comparison ===
            print(f"\n=== Multiple Folds Comparison Analysis ===")
            
            print("\nSelect IC display mode:")
            print("1. Show all IC components")
            print("2. Show specified number of IC components")
            
            ic_choice = input("Please select (1/2, default 2): ").strip() or "2"
            
            if ic_choice == "1":
                show_all_ics = True
                x_tick_num = None
            else:
                show_all_ics = False
                x_tick_num = int(input("Enter number of ICs to display (default 15): ").strip() or "15")
            
            # Multi-fold comparison visualization
            visualize_multiple_folds_comparison(attention_dir, repeat_idx, 
                                              selected_folds=None,  # Interactive selection
                                              x_tick_num=x_tick_num, 
                                              show_all_ics=show_all_ics)
        
        elif choice == "3":
            # === Auto analyze all available folds ===
            print(f"\n=== Auto Analyze All Available Folds ===")
            available_folds = get_available_folds(attention_dir, repeat_idx, debug=True)
            print(f"Found {len(available_folds)} available folds: {available_folds}")
            
            if not available_folds:
                print("No available fold data found")
                return
            
            # Generate visualization for each fold
            for fold_idx in available_folds:
                try:
                    print(f"\nProcessing Fold {fold_idx}...")
                    attention_matrix, epochs = load_attention_weights(attention_dir, repeat_idx, fold_idx)
                    
                    # Use automatic configuration
                    save_path = f"attention_evolution_auto_repeat{repeat_idx}_fold{fold_idx}.png"
                    visualize_attention_evolution(attention_matrix, epochs, repeat_idx, fold_idx, 
                                                save_path=save_path, x_tick_num=20, figsize=(16, 8))
                    
                except Exception as e:
                    print(f"Error processing Fold {fold_idx}: {e}")
            
            # Generate comparison plot for all folds
            print(f"\nGenerating comparison plot for all folds...")
            visualize_multiple_folds_comparison(attention_dir, repeat_idx, 
                                              selected_folds=available_folds,
                                              x_tick_num=15, show_all_ics=False)
        
        elif choice == "4":
            # === Average attention analysis ===
            print(f"\n=== Average Attention Across Multiple Folds ===")
            available_folds = get_available_folds(attention_dir, repeat_idx, debug=True)
            print(f"Automatically detected {len(available_folds)} folds: {available_folds}")
            
            if len(available_folds) < 2:
                print("Need at least 2 folds. Try manual input:")
                available_folds = manual_fold_input()
                if not available_folds or len(available_folds) < 2:
                    print("Insufficient folds for averaging")
                    return
            
            # Select folds to analyze
            print("\n1. Use all available folds")
            print("2. Select specific folds")
            
            fold_choice = input("Please select (1/2, default 1): ").strip() or "1"
            
            if fold_choice == "1":
                selected_folds = available_folds
            else:
                print(f"Available folds: {available_folds}")
                fold_input = input("Enter fold numbers (e.g., 1,2,3,4,5): ").strip()
                try:
                    selected_folds = [int(x.strip()) for x in fold_input.split(',')]
                    selected_folds = [f for f in selected_folds if f in available_folds]
                except ValueError:
                    selected_folds = available_folds
            
            if len(selected_folds) < 2:
                print("Need at least 2 valid folds")
                return
            
            # Select save format
            print("\nOutput format options:")
            print("1. PNG only")
            print("2. PDF only")
            print("3. Both PNG and PDF (recommended)")
            
            format_choice = input("Please select format (1/2/3, default 3): ").strip() or "3"
            format_map = {"1": "png", "2": "pdf", "3": "both"}
            save_format = format_map.get(format_choice, "both")
            
            print(f"Computing average attention for folds: {selected_folds}")
            
            # Calculate average attention
            mean_attention, std_attention, common_epochs, fold_data = load_multiple_folds_attention(
                attention_dir, repeat_idx, selected_folds, debug=True)
            
            # Analyze statistical characteristics and save IC ranking
            ranking_results = analyze_average_attention_statistics(mean_attention, std_attention, common_epochs, selected_folds, save_ranking=True)
            
            # Visualize
            fold_str = "_".join(map(str, selected_folds))
            save_path = f"attention_average_repeat{repeat_idx}_folds_{fold_str}.png"
            
            print(f"\nSaving format: {save_format}")
            if save_format == "both":
                print("Will save both combined and individual figures in PNG and PDF formats")
            else:
                print(f"Will save both combined and individual figures in {save_format.upper()} format")
            
            visualize_average_attention(
                mean_attention, std_attention, common_epochs,
                repeat_idx, selected_folds, save_path=save_path,
                show_all_ics=True, figsize=(24, 8), show_std=True,
                save_individual=True, save_format=save_format
            )
        
        else:
            print("Invalid selection")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()