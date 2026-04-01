"""
Analyze Embedding Space Efficiency

Measures:
1. Participation Ratio (effective dimensionality)
2. Cumulative Explained Variance (how many dims needed for 95%?)
3. Anisotropy Score (are vectors clustered or spread out?)
4. Singular Value Spectrum (visualization)

Usage:
    python analyze_embedding_space.py
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_embeddings(path=None):
    """Load embedding matrix from trained model or generate random baseline"""
    if path and Path(path).exists():
        # Load from saved model
        import torch
        state_dict = torch.load(path, map_location='cpu')
        emb = state_dict.get('tok_emb.weight', state_dict.get('embedding.weight'))
        return emb.numpy()
    else:
        # Generate different types of embeddings for comparison
        np.random.seed(42)
        vocab_size, dim = 8192, 512
        
        embeddings = {}
        
        # 1. Random Gaussian (baseline - should be isotropic)
        embeddings['random_gaussian'] = np.random.randn(vocab_size, dim).astype(np.float32)
        
        # 2. Random with low effective rank (simulating redundancy)
        # Project through a bottleneck
        bottleneck = 100
        W1 = np.random.randn(dim, bottleneck)
        W2 = np.random.randn(bottleneck, dim)
        low_rank = np.random.randn(vocab_size, dim) @ W1 @ W2
        embeddings['low_rank_100'] = (low_rank / np.linalg.norm(low_rank, axis=1, keepdims=True)).astype(np.float32)
        
        # 3. Clustered (simulating anisotropic space)
        centers = np.random.randn(10, dim)  # 10 cluster centers
        cluster_ids = np.random.randint(0, 10, vocab_size)
        clustered = centers[cluster_ids] + 0.1 * np.random.randn(vocab_size, dim)
        embeddings['clustered'] = clustered.astype(np.float32)
        
        return embeddings

def compute_participation_ratio(singular_values):
    """
    Participation Ratio = (sum σ²)² / sum σ⁴
    
    Measures effective dimensionality.
    If all singular values equal: PR = d (full rank)
    If only one non-zero: PR = 1 (rank 1)
    """
    s2 = singular_values ** 2
    s4 = singular_values ** 4
    return (s2.sum() ** 2) / (s4.sum() + 1e-10)

def compute_explained_variance(singular_values):
    """Cumulative explained variance ratio for each component"""
    variance = singular_values ** 2
    total_variance = variance.sum()
    cumulative = np.cumsum(variance) / total_variance
    return cumulative

def compute_anisotropy(embeddings, n_samples=1000):
    """
    Anisotropy = average cosine similarity between random pairs
    
    0 = isotropic (vectors uniformly distributed) - GOOD
    1 = all vectors pointing same direction - BAD
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)
    
    # Sample random pairs
    n = min(n_samples, len(embeddings))
    idx1 = np.random.choice(len(embeddings), n, replace=False)
    idx2 = np.random.choice(len(embeddings), n, replace=False)
    
    # Compute cosine similarities
    cos_sims = (normalized[idx1] * normalized[idx2]).sum(axis=1)
    
    return cos_sims.mean(), cos_sims.std()

def analyze_embedding_matrix(name, embeddings):
    """Full analysis of an embedding matrix"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"Shape: {embeddings.shape}")
    print(f"{'='*60}")
    
    # Center the embeddings (remove mean)
    centered = embeddings - embeddings.mean(axis=0)
    
    # SVD
    U, S, Vh = np.linalg.svd(centered, full_matrices=False)
    
    # 1. Participation Ratio
    pr = compute_participation_ratio(S)
    print(f"\n📊 Participation Ratio: {pr:.1f} / {embeddings.shape[1]} dimensions")
    print(f"   Efficiency: {pr/embeddings.shape[1]*100:.1f}%")
    
    # 2. Explained Variance
    cumvar = compute_explained_variance(S)
    dims_90 = np.searchsorted(cumvar, 0.90) + 1
    dims_95 = np.searchsorted(cumvar, 0.95) + 1
    dims_99 = np.searchsorted(cumvar, 0.99) + 1
    
    print(f"\n📈 Cumulative Explained Variance:")
    print(f"   90% variance in {dims_90} dimensions")
    print(f"   95% variance in {dims_95} dimensions")
    print(f"   99% variance in {dims_99} dimensions")
    
    # 3. Anisotropy
    aniso_mean, aniso_std = compute_anisotropy(embeddings)
    print(f"\n🎯 Anisotropy Score: {aniso_mean:.4f} ± {aniso_std:.4f}")
    if aniso_mean < 0.1:
        print("   ✅ Good! Space is roughly isotropic")
    elif aniso_mean < 0.3:
        print("   ⚠️ Moderate anisotropy, some redundancy")
    else:
        print("   ❌ High anisotropy! Vectors are clustered")
    
    # 4. Top singular values
    print(f"\n📉 Top 10 Singular Values:")
    for i in range(10):
        bar = '█' * int(S[i] / S[0] * 30)
        print(f"   σ_{i+1:2d}: {S[i]:8.2f} {bar}")
    
    return {
        'name': name,
        'shape': embeddings.shape,
        'participation_ratio': pr,
        'efficiency': pr / embeddings.shape[1],
        'dims_90': dims_90,
        'dims_95': dims_95,
        'dims_99': dims_99,
        'anisotropy': aniso_mean,
        'singular_values': S,
        'cumulative_variance': cumvar,
    }

def plot_analysis(results_list, save_path='embedding_analysis.png'):
    """Create visualization of the analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Singular Value Spectrum
    ax1 = axes[0, 0]
    for r in results_list:
        ax1.plot(r['singular_values'][:100], label=r['name'], linewidth=2)
    ax1.set_xlabel('Component Index')
    ax1.set_ylabel('Singular Value')
    ax1.set_title('Singular Value Spectrum (first 100)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Cumulative Explained Variance
    ax2 = axes[0, 1]
    for r in results_list:
        ax2.plot(r['cumulative_variance'][:200], label=r['name'], linewidth=2)
    ax2.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
    ax2.set_xlabel('Number of Dimensions')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('How Many Dimensions Do We Really Need?')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Participation Ratio Comparison
    ax3 = axes[1, 0]
    names = [r['name'] for r in results_list]
    prs = [r['participation_ratio'] for r in results_list]
    max_dim = results_list[0]['shape'][1]
    
    bars = ax3.bar(names, prs, color=['#4CAF50', '#2196F3', '#FF9800'][:len(names)])
    ax3.axhline(y=max_dim, color='red', linestyle='--', label=f'Max ({max_dim})')
    ax3.set_ylabel('Participation Ratio (Effective Dimensions)')
    ax3.set_title('Effective Dimensionality')
    ax3.legend()
    
    # Add value labels on bars
    for bar, pr in zip(bars, prs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{pr:.0f}', ha='center', va='bottom', fontsize=11)
    
    # Plot 4: Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = []
    headers = ['Metric', *names]
    table_data.append(['Participation Ratio'] + [f'{r["participation_ratio"]:.0f}' for r in results_list])
    table_data.append(['Efficiency'] + [f'{r["efficiency"]*100:.1f}%' for r in results_list])
    table_data.append(['Dims for 95%'] + [str(r['dims_95']) for r in results_list])
    table_data.append(['Anisotropy'] + [f'{r["anisotropy"]:.3f}' for r in results_list])
    
    table = ax4.table(cellText=table_data, colLabels=headers, loc='center',
                      cellLoc='center', colColours=['#E8E8E8']*len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax4.set_title('Summary Comparison', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved analysis plot to {save_path}")
    plt.close()

def main():
    print("="*60)
    print("Embedding Space Efficiency Analysis")
    print("="*60)
    
    # Load or generate embeddings
    embeddings_dict = load_embeddings()
    
    if isinstance(embeddings_dict, dict):
        # Multiple embedding types for comparison
        results = []
        for name, emb in embeddings_dict.items():
            result = analyze_embedding_matrix(name, emb)
            results.append(result)
        
        plot_analysis(results)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY: What This Means for Parameter Golf")
        print("="*60)
        print("""
If your trained model has:
- Low Participation Ratio (e.g., 100 out of 512): 
  → ~80% of dimensions are "wasted"
  → Opportunity: Could use smaller dim with whitening/MRL

- High Anisotropy (e.g., > 0.3):
  → Vectors are clustered, not using full space
  → Opportunity: Whitening could spread them out

- 95% variance in few dims (e.g., 150 out of 512):
  → Most information in subset of dimensions
  → Opportunity: PCA-based compression

POTENTIAL GAINS:
If we can compress 512 dims to 256 effective dims:
- Save ~50% embedding parameters
- Use saved budget for more layers
- Possibly IMPROVE performance (remove noise dimensions)
        """)
    else:
        # Single embedding matrix
        analyze_embedding_matrix("trained_model", embeddings_dict)

if __name__ == "__main__":
    main()
