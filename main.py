import argparse
import torch
from torch.utils.data import DataLoader
from models.unet import UNet
from train.trainer import train_model
from test.test_utils import test_model
from data.dataset import SegmentationDataset
from utils.helpers import save_checkpoint

def main():
    parser = argparse.ArgumentParser(description="U-Net Segmentation")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train_images", type=str, default="data/train/images")
    parser.add_argument("--train_masks", type=str, default="data/train/masks")
    parser.add_argument("--test_images", type=str, default="data/test/images")
    parser.add_argument("--test_masks", type=str, default="data/test/masks")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == "train":
        train_dataset = SegmentationDataset(args.train_images, args.train_masks, transform=None)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        trained_model = train_model(args, model, train_loader, optimizer, device)
        save_checkpoint(trained_model, optimizer, filename="unet_checkpoint.pth.tar")

    elif args.mode == "test":
        test_dataset = SegmentationDataset(args.test_images, args.test_masks, transform=None)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint["state_dict"])
        test_model(args, model, test_loader, device)

if __name__ == "__main__":
    main()
