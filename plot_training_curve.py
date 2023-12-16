import re
import matplotlib.pyplot as plt

# Read the file
with open('training_loss.txt', 'r') as file:
    lines = file.readlines()

# Extract loss values and epoch numbers
losses = [float(re.search(r"Loss: (\d+\.\d+)", line).group(1)) for line in lines]
epochs = [int(re.search(r"epoch (\d+)/", line).group(1)) for line in lines]

# Visualizing Loss vs. Iterations
plt.figure(figsize=(15, 5))
plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs. Iterations')
# plt.show()
plt.savefig("loss_vs_iterations.jpg")
plt.close()

# Calculating Average Loss for Each Epoch
unique_epochs = list(set(epochs))
average_losses = [sum([losses[i] for i in range(len(losses)) if epochs[i] == epoch]) / epochs.count(epoch) for epoch in unique_epochs]

plt.figure(figsize=(15, 5))
plt.plot(unique_epochs, average_losses, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.title('Average Loss vs. Epochs')
# plt.show()
plt.savefig("loss_vs_epochs.jpg")
plt.close()

