import torch, os, cv2
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.video_paths, self.labels = [], []
        self.label_map = {cls: i for i, cls in enumerate(os.listdir(root_dir))}
        for label in os.listdir(root_dir):
            class_path = os.path.join(root_dir, label)
            for vid in os.listdir(class_path):
                self.video_paths.append(os.path.join(class_path, vid))
                self.labels.append(self.label_map[label])
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_paths[idx])
        frames = []
        while len(frames) < 16:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
        cap.release()
        video = torch.tensor(frames).permute(3, 0, 1, 2) / 255.0
        return video.float(), self.labels[idx]

class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.fc = nn.Linear(8 * 8 * 56 * 56, 3)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train():
    dataset = VideoDataset("data/")
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = Simple3DCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), torch.tensor(targets).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "models/exercise_classifier.pth")

if __name__ == "__main__":
    train()
