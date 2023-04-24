import importlib
import argparse
import torch
import torchvision.transforms as transforms

from datasets.EggDataset import EggDataset

parser = argparse.ArgumentParser(description='Inference script')
parser.add_argument('model_path', type=str, default='model.pth', help='Path to the model')
parser.add_argument('input_dir', type=str, help='Directory containing the input image', )
parser.add_argument('--device', type=str, default='cpu', help='Device to be used for computations')
args = parser.parse_args()

device = torch.device(args.device)

model = torch.load(args.model_path).to(device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = EggDataset(args.input_dir, transforms=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

for data in dataloader:
    images, labels = data
    images = images.to(device)
    output = model(images)
    _, predicted = torch.max(output.data, 1)
    print('Predicted class: {}, Real class: {}'.format(dataset.classes[predicted.item()], dataset.classes[labels.item()]))

