# src/measure_baseline_fid.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
import tqdm

def calculate_baseline_fid(device='cuda'):
    """
    진짜 MNIST 데이터(Train vs Test) 간의 Baseline FID를 측정합니다.
    단, 16x16으로 줄였다가 299x299로 복원하는 전처리를 거칩니다.
    """
    print(f"Using device: {device}")
    
    # --- 1. FID Metric initialization ---
    # feature=2048
    # normalize=True
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # --- 2. load data (MNIST) ---
    transform = transforms.Compose([
        transforms.ToTensor(), # 0~1로 변환
    ])

    # Train dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # Test dataset
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    num_samples = 2000
    
    # Subset generation
    train_subset = torch.utils.data.Subset(train_dataset, range(num_samples))
    test_subset = torch.utils.data.Subset(test_dataset, range(num_samples))

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    # --- 3. preprocessing ---
    def process_images(imgs):
        """
        Input: (B, 1, 28, 28) -> 16x16 Downsample -> 299x299 Upsample -> RGB
        """
        imgs = imgs.to(device)
        
        # (1) Downsample to 16x16
        imgs_16 = F.interpolate(imgs, size=(16, 16), mode='bilinear', align_corners=False)
        
        # (2) Upsample to 299x299
        imgs_299 = F.interpolate(imgs_16, size=(299, 299), mode='bilinear', align_corners=False)
        
        # (3) Grayscale to RGB (1ch -> 3ch)
        imgs_rgb = imgs_299.repeat(1, 3, 1, 1)
        
        return imgs_rgb

    # --- 4. FID update ---
    print("Processing Real Train Data (Baseline 1)...")
    for batch, _ in tqdm.tqdm(train_loader):
        processed = process_images(batch)
        fid.update(processed, real=True) # real=True

    print("Processing Real Test Data (Baseline 2)...")
    for batch, _ in tqdm.tqdm(test_loader):
        processed = process_images(batch)
        fid.update(processed, real=False) # real=False

    # --- 5. calculation ---
    print("Computing FID...")
    score = fid.compute()
    return score.item()

if __name__ == "__main__":
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        baseline_score = calculate_baseline_fid(device)
        
        print("\n" + "="*50)
        print(f"RESULT: Baseline Real-to-Real FID Score: {baseline_score:.4f}")
        print("="*50)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Try: pip install torchmetrics[image] torchvision")