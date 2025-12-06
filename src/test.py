from torch.utils.data import DataLoader

# Создание датасета и загрузчика
transform = T.Compose([
    T.Resize(64),  # tiny-imagenet — 64x64, но можно оставить как есть
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = TinyImageNetDataset(root="tiny-imagenet-200", split="train", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Создание модели
model = ResNet18(num_classes=200)
sample_batch, labels = next(iter(dataloader))
output = model(sample_batch)
print(output.shape)  # torch.Size([32, 200])