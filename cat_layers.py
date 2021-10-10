import torch

model_path = '/Users/lxy/PycharmProjects/mnist_web/densenet201_3.pth'
weight_path = '/Users/lxy/PycharmProjects/mnist_web/densenet201_3_best_acc.pth'
model = torch.load(model_path, map_location='cpu')
model.load_state_dict(torch.load(weight_path, map_location='cpu'))
for name in model.named_modules():
    print(name)

print(model.features[-1])
