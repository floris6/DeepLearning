import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Model class again before loading
class Model(nn.Module):

    def __init__(self, in_features=7, h1=32, h2=32, out_features=14):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x


model = Model()



model.load_state_dict(torch.load("energie.pth", weights_only=True))
model.eval()

print("Model loaded successfully!")


new_input = torch.tensor([0.325, 0, 1.55, 0.3, -0.2, -1, 0.3]).float()  # Ensure tensor is float
with torch.no_grad():  # Disable gradient computation
    prediction = model(new_input)
    predicted_class = prediction.argmax().item()

print(f"Predicted class: {predicted_class}")
