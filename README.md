# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
Regression is a supervised learning technique used to predict continuous values.
A Neural Network Regression Model consists of an input layer, one or more hidden layers, and an output layer. The model learns patterns by adjusting weights using backpropagation and minimizing a loss function.

In this experiment, a neural network is trained to learn the relationship between Age and Spending Score using Mean Squared Error (MSE) loss.

## Neural Network Model

<img width="746" height="797" alt="image" src="https://github.com/user-attachments/assets/f68b2319-626c-4773-87dd-9f4f4c1a53bb" />

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: YAZHINI R R
### Register Number: 212224100063
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)
        self.history = {'loss': []}

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.01)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = ai_brain(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")




```
## Dataset Information

<img width="206" height="305" alt="image" src="https://github.com/user-attachments/assets/fcd3c880-6d2e-4d88-b00b-223bea420d28" />

## OUTPUT

### Training Loss Vs Iteration Plot

<img width="803" height="553" alt="image" src="https://github.com/user-attachments/assets/0e86938d-ab54-4e6b-a5ee-f4ebfa43f9df" />

### New Sample Data Prediction

<img width="518" height="77" alt="image" src="https://github.com/user-attachments/assets/3d9d97a2-bcb5-470d-b969-b491311a820f" />

## RESULT
Thus, a Neural Network Regression Model was successfully developed and trained using PyTorch. The model was able to predict continuous output values with minimal loss and good performance on test data.
