import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split


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


torch.manual_seed(12)
model = Model()



# FROM GITHUB RAW FILE.
url = 'https://raw.githubusercontent.com/Peter5687/dataset/refs/heads/main/merged_weather_energy.csv'

my_df = pd.read_csv(url)



# Ensure EUR is a float and scale by dividing by 10
my_df['EUR'] = my_df['EUR'].astype(int)

#print(my_df['EUR'])









# X is input and y is outcome
X = my_df.drop('EUR', axis=1).values
y = my_df['EUR'].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=12)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)



y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

epoch = 1000
losses = []
for i in range(epoch):
    y_pred = model.forward(X_train)

    loss = criterion(y_pred, y_train)

    losses.append(loss.detach().numpy())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


correct = 0
loss_r_one = 0
loss_r_two = 0
loss_r_three = 0
total = len(X_test)
with torch.no_grad():
    for i in range(len(X_test)):  # Loop over all test samples
        x_val = X_test[i].unsqueeze(0)  # Add batch dimension
        y_val = model.forward(x_val)  # Get prediction
        predicted_class = y_val.argmax().item()  # Get the predicted class label

        # Output the real and predicted labels
        # print(f'{i + 1}.) {y_test[i].item()} \t  {predicted_class}')

        if predicted_class == y_test[i].item():
            correct += 1
        elif abs(predicted_class - y_test[i].item()) == 1:
            loss_r_one += 1
        elif abs(predicted_class - y_test[i].item()) == 2:
            loss_r_two += 1
        elif abs(predicted_class - y_test[i].item()) == 3:
            loss_r_three += 1

    accuracy = correct / total
    accuracy_1 = loss_r_one / total
    accuracy_2 = loss_r_two / total
    accuracy_3 = loss_r_three / total
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'\nPredictions off by 1: {accuracy_1 * 100:.2f}%')
    print(f'Predictions off by 2: {accuracy_2 * 100:.2f}%')
    print(f'\n\trough accuracy: {(accuracy + accuracy_1) * 100:.2f}%')
    print(f'\t{loss_r_one + loss_r_two}/{total}')

    print(f'\n\trough loss; {((1 - accuracy - accuracy_1) * 100):.2f}% (off by more than 3)')



torch.save(model.state_dict(), "energie.pth")
#new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])

