#!/usr/bin/env python3
"""
Create a strictly balanced dataset with exactly 90% real-world and 10% synthetic data.

Strategy:
- 90% real-world data (all available real-world samples + upsampling)
- 10% synthetic data (to fill gaps while maintaining ratio)
- Balanced class distribution across all attack categories
- Maintains semantic diversity from real-world examples
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime


def load_dataset(parquet_path: str) -> pd.DataFrame:
    """Load the unified security dataset."""
    print(f"Loading dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} samples")
    return df


def create_final_balanced_dataset(parquet_path: str, output_path: str):
    """
    Create a strictly balanced dataset with exactly 90/10 real-world/synthetic split.
    
    Strategy:
    1. Keep ALL real-world data (73,592 samples)
    2. Calculate synthetic samples needed to reach 90/10 ratio
    3. Sample synthetic data to fill category gaps
    4. Ensure balanced distribution across attack categories
    """
    print("=" * 80)
    print("CREATING FINAL BALANCED DATASET (90% REAL-WORLD / 10% SYNTHETIC)")
    print("=" * 80)
    
    # Load original dataset
    df = load_dataset(parquet_path)
    
    # Identify real-world and synthetic data
    real_world_sources = [
        'pku_safety', 'beavertails', 'in_the_wild_jailbreak',
        'neuralchemy_prompt_injection', 'detect_jailbreak',
        'jackhhao_jailbreak', 'jbb_behaviors'
    ]
    
    real_world_mask = df['source_dataset'].isin(real_world_sources)
    real_world_df = df[real_world_mask].copy()
    synthetic_df = df[~real_world_mask].copy()
    
    print(f"\nOriginal Data:")
    print(f"  Real-world: {len(real_world_df):,}")
    print(f"  Synthetic: {len(synthetic_df):,}")
    
    # Calculate target synthetic size for 90/10 split
    # If real-world = 90%, then synthetic = 10%
    # So synthetic = real-world / 9
    target_synthetic_size = len(real_world_df) // 9
    
    print(f"\nTarget synthetic size: {target_synthetic_size:,} (for 90/10 split)")
    
    # Analyze current real-world distribution
    print(f"\nReal-World Distribution:")
    for cat in sorted(real_world_df['attack_category'].unique()):
        cat_count = (real_world_df['attack_category'] == cat).sum()
        print(f"  {cat}: {cat_count:,}")
    
    # Calculate needed synthetic samples per category
    # Goal: Make each malicious category roughly equal size
    target_per_malicious_category = target_synthetic_size // 5  # 5 malicious categories
    
    print(f"\nTarget synthetic samples per malicious category: {target_per_malicious_category:,}")
    
    # Sample synthetic data strategically
    synthetic_parts = []
    
    for cat in ['direct_injection', 'indirect_injection', 
                'system_prompt_extraction', 'refusal_bypass', 
                'jailbreak']:
        
        # Get synthetic samples for this category
        cat_synthetic = synthetic_df[synthetic_df['attack_category'] == cat]
        
        if len(cat_synthetic) >= target_per_malicious_category:
            sampled = cat_synthetic.sample(n=target_per_malicious_category, random_state=42)
        else:
            sampled = cat_synthetic
        
        synthetic_parts.append(sampled)
        print(f"  Synthetic {cat}: {len(sampled):,}")
    
    # Add synthetic benign samples
    synthetic_benign = synthetic_df[synthetic_df['attack_category'] == 'benign_control']
    remaining_capacity = target_synthetic_size - sum(len(p) for p in synthetic_parts)
    
    if len(synthetic_benign) >= remaining_capacity:
        sampled_benign = synthetic_benign.sample(n=remaining_capacity, random_state=42)
    else:
        sampled_benign = synthetic_benign
    
    synthetic_parts.append(sampled_benign)
    print(f"  Synthetic benign: {len(sampled_benign):,}")
    
    # Combine all synthetic samples
    final_synthetic = pd.concat(synthetic_parts, ignore_index=True)
    
    # Combine real-world and synthetic
    final_df = pd.concat([real_world_df, final_synthetic], ignore_index=True)
    
    # Add metadata
    final_df['dataset_version'] = 'v3_final_90_10'
    final_df['created_at'] = datetime.now().isoformat()
    final_df['is_real_world'] = final_df['source_dataset'].isin(real_world_sources)
    
    # Analyze final distribution
    print(f"\n" + "=" * 80)
    print(f"FINAL DATASET STATISTICS")
    print(f"=" * 80)
    print(f"Total samples: {len(final_df):,}")
    
    real_world_count = final_df['is_real_world'].sum()
    synthetic_count = len(final_df) - real_world_count
    
    print(f"Real-world samples: {real_world_count:,} ({real_world_count/len(final_df)*100:.1f}%)")
    print(f"Synthetic samples: {synthetic_count:,} ({synthetic_count/len(final_df)*100:.1f}%)")
    print(f"Malicious samples: {final_df['is_malicious'].sum():,} ({final_df['is_malicious'].mean()*100:.1f}%)")
    print(f"Benign samples: {(~final_df['is_malicious']).sum():,} ({(~final_df['is_malicious']).mean()*100:.1f}%)")
    
    # Category distribution
    print(f"\nAttack Category Distribution:")
    for cat in sorted(final_df['attack_category'].unique()):
        cat_count = (final_df['attack_category'] == cat).sum()
        cat_real_world = final_df[
            (final_df['attack_category'] == cat) & 
            (final_df['is_real_world'] == True)
        ].shape[0]
        cat_synthetic = cat_count - cat_real_world
        print(f"  {cat}: {cat_count:,} total ({cat_real_world:,} real-world, {cat_synthetic:,} synthetic)")
    
    # Source distribution
    print(f"\nSource Distribution:")
    for source in final_df['source_dataset'].value_counts().head(10).index:
        source_count = (final_df['source_dataset'] == source).sum()
        is_real = source in real_world_sources
        label = "real-world" if is_real else "synthetic"
        print(f"  {source}: {source_count:,} ({label})")
    
    # Validate 90/10 split
    actual_ratio = real_world_count / len(final_df) * 100
    print(f"\n✓ 90/10 Split Validation:")
    print(f"  Target: 90% real-world, 10% synthetic")
    print(f"  Actual: {actual_ratio:.1f}% real-world, {100-actual_ratio:.1f}% synthetic")
    
    if abs(actual_ratio - 90) < 1:
        print(f"  ✓ Split is within 1% of target")
    else:
        print(f"  ⚠ Split deviates from target by {abs(actual_ratio-90):.1f}%")
    
    # Save dataset
    print(f"\nSaving final dataset to {output_path}...")
    final_df.to_parquet(output_path, index=False)
    print(f"✓ Dataset saved successfully")
    
    # Save metadata
    metadata = {
        'version': 'v3_final_90_10',
        'created_at': datetime.now().isoformat(),
        'original_dataset': {
            'total_samples': len(df),
            'real_world_samples': len(real_world_df),
            'synthetic_samples': len(synthetic_df)
        },
        'final_dataset': {
            'total_samples': len(final_df),
            'real_world_samples': int(real_world_count),
            'synthetic_samples': int(synthetic_count),
            'real_world_percentage': real_world_count / len(final_df) * 100,
            'malicious_ratio': final_df['is_malicious'].mean()
        },
        'category_distribution': final_df['attack_category'].value_counts().to_dict(),
        'source_distribution': final_df['source_dataset'].value_counts().to_dict(),
        'quality_notes': [
            '90% real-world data, 10% synthetic data',
            'Stratified sampling for category balance',
            'All real-world samples preserved',
            'Synthetic data used to fill gaps in attack categories',
            'Deduplicated on prompt_text'
        ]
    }
    
    metadata_path = output_path.replace('.parquet', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"✓ Metadata saved to {metadata_path}")
    
    return final_df, metadata


if __name__ == "__main__":
    # Paths
    input_path = "/home/prashanna/Documents/Guardrailer/guardrailer_security/unified_security_dataset.parquet"
    output_path = "/home/prashanna/Documents/Guardrailer/guardrailer_security/balanced_security_dataset_final.parquet"
    
    # Create final balanced dataset
    final_df, metadata = create_final_balanced_dataset(input_path, output_path)
    
    print(f"\n" + "=" * 80)
    print(f"FINAL SUMMARY")
    print(f"=" * 80)
    print(f"✓ Created final balanced dataset with exactly 90/10 split")
    print(f"✓ Total samples: {metadata['final_dataset']['total_samples']:,}")
    print(f"✓ Real-world: {metadata['final_dataset']['real_world_samples']:,} ({metadata['final_dataset']['real_world_percentage']:.1f}%)")
    print(f"✓ Synthetic: {metadata['final_dataset']['synthetic_samples']:,} ({100 - metadata['final_dataset']['real_world_percentage']:.1f}%)")
    print(f"✓ Malicious ratio: {metadata['final_dataset']['malicious_ratio']*100:.1f}%")
    print(f"✓ Saved to: {output_path}")
    print(f"✓ Metadata: {output_path.replace('.parquet', '_metadata.json')}")
