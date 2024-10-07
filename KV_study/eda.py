import torch

# Load the .pth file
data = torch.load('hidden_data.pth')

# Print the type and content of the loaded data
print(type(data))
print(len(data))

for key in data.keys():
    print(key)

print(data['Key at layer 31, token 9'].shape)