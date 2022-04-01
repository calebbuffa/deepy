from __future__ import annotations

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score
from torchvision import transforms

class SemanticSegmentationDataset(Dataset):
    def __init__(self, path: str, partition: str):
        self.images, self.masks = load_imagery(path, partition)
        # x_transforms = [transforms.ToTensor()]
        # y_transforms = [transforms.ToTensor()]
        # self.x_transforms = transforms.Compose(x_transforms)
        # self.y_transforms = transforms.Compose(y_transforms)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # image = self.x_transforms(self.images[idx])
        # mask = self.x_transforms(self.masks[idx])
        return (
            torch.from_numpy(self.images[idx]).float().contiguous(),
            torch.from_numpy(self.masks[idx]).float().contiguous(),
        )


class SemanticSegmentationDataModule:
    def __init__(self, path: str, **kwargs):
        self.train_dataset = SemanticSegmentationDataset(path, "train")
        self.val_dataset = SemanticSegmentationDataset(path, "val")
        self.kwargs = kwargs

    @property
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, shuffle=True, **self.kwargs)

    @property
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, shuffle=False, **self.kwargs)


class UNetTrainer:
    def __init__(
        self, model: nn.Module, optimizer: torch.optim = None, out_path: str = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        if self.model.n_classes == 1:
            self.loss_fn = nn.BCELoss()
            self.sigmoid = nn.Sigmoid()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.out_path = out_path
        self.dice = DiceLoss(n_classes=self.model.n_classes)

    def predict(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        if len(logits.size) == 4:
            dim = 1
        elif len(logits.size) == 3:
            dim = 0

        probs = F.softmax(x, dim=dim).argmax(dim=dim)

        if self.model.n_classes == 1:
            return (probs > 0.1).squeeze().detach().cpu().numpy()

        return probs[probs.argmax(dim=1)].detach().cpu().numpy()

    def train_step(self, train_batch) -> Tensor:
        x, y = train_batch
        x = x.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(x)
        return self.compute_loss(y_pred, y)

    def val_step(self, val_vatch) -> tuple[Tensor, Tensor]:
        x, y = val_vatch
        x = x.to(self.device)
        y = y.to(self.device)
        with torch.no_grad():
            y_pred = self.model(x)
            loss = self.compute_loss(y_pred, y)
            y_pred = (
                F.softmax(y_pred, dim=1).argmax(dim=1)
                if self.model.n_classes > 1
                else F.sigmoid(y_pred).squeeze()
            )
        return loss, y_pred, y

    def compute_loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if isinstance(self.loss_fn, nn.CrossEntropyLoss):
            ce_loss = self.loss_fn(y_hat.float(), y.long())
        elif isinstance(self.loss_fn, nn.BCELoss):
            ce_loss = self.loss_fn(self.sigmoid(y_hat).squeeze(), y.float())
        else:
            ce_loss = self.loss_fn(y_hat, y)

        dice_loss = 1 - self.dice(y_hat.float(), y.long())
        return ce_loss + dice_loss

    def train_epoch(self, train_dataloader: DataLoader) -> Tensor:
        self.model.train()
        torch.set_grad_enabled(True)

        train_loss = []
        for batch in train_dataloader:
            self.optimizer.zero_grad()
            loss = self.train_step(batch)
            train_loss.append(loss.detach())
            loss.backward()
            self.optimizer.step()

        return torch.mean(torch.stack(train_loss))

    def val_epoch(self, val_dataloader: DataLoader) -> tuple[Tensor, list[Tensor]]:
        self.model.eval()
        torch.set_grad_enabled(False)

        val_loss = []
        preds = []
        ground_truth = []
        for batch in val_dataloader:
            loss, pred, y = self.val_step(batch)
            val_loss.append(loss)
            preds.append(pred.detach().cpu())
            ground_truth.append(y.detach().cpu())


        return torch.mean(torch.stack(val_loss)), preds, ground_truth

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int
    ) -> dict[str, list]:
        train_losses: list[float] = []
        val_losses = []
        val_preds = []
        val_ground_truths = []
        for _ in tqdm(range(epochs), desc="Epoch"):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_pred, ground_truth = self.val_epoch(val_loader)
            train_losses.append(train_loss.detach().cpu().numpy().tolist())
            val_losses.append(val_loss.detach().cpu().numpy().tolist())
            val_preds.extend(val_pred)
            val_ground_truths.extend(ground_truth)

        torch.save(self.model.state_dict(), self.out_path)

        return {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_preds": val_preds,
            "val_ground_truths": val_ground_truths,
        }

    @staticmethod
    def compute_metrics(y_hat: Tensor, y: Tensor) -> Tensor:
        return f1_score(
            y.flatten().detach().cpu().numpy(), y_hat.flatten().detach().cpu().numpy()
        )
