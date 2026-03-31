"""
Multimodal dataset handler - uses synthetic data generation.
No external downloads required.
"""

from pathlib import Path
from .synthetic_data import get_generator

class MultimodalDatasetDownloader:
    def __init__(self, data_root: str = "data/multimodal"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
    
    def download_all(self, force: bool = False):
        print("="*60)
        print("📦 VHAS Multimodal Dataset (Synthetic)")
        print("="*60)
        print("\nGenerating synthetic multimodal dataset for training/evaluation...")
        
        gen = get_generator()
        data = gen.generate_dataset(num_samples=2000, save_dir=self.data_root)
        
        print(f"\n✅ Generated {len(data['labels'])} synthetic samples.")
        print(f"   - Audio features: {data['audio_features'].shape}")
        print(f"   - Thermal features: {data['thermal_features'].shape}")
        print(f"   - Gait features: {data['gait_features'].shape}")
        print(f"\n📁 Data stored in: {self.data_root.absolute()}")
        print("="*60 + "\n")
        return data

def ensure_datasets_downloaded(data_root: str = "data/multimodal", force: bool = False):
    downloader = MultimodalDatasetDownloader(data_root)
    downloader.download_all(force=force)
    return downloader

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_datasets_downloaded(force=args.force)