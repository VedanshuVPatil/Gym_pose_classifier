import torch, os, cv2
from torch.utils.data import DataLoader
from train import VideoDataset, Simple3DCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels = ['bicep_curl', 'lateral_raise', 'squat']

def evaluate():
    dataset = VideoDataset("data/")  
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = Simple3DCNN().to(device)
    model.load_state_dict(torch.load("models/exercise_classifier.pth"))
    model.eval()

    correct, total = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), torch.tensor(targets).to(device)
        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    acc = 100 * correct / total
    print(f" Evaluation Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    evaluate()
