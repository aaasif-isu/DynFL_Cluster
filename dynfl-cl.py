import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms, models
import argparse
from collections import defaultdict
import copy
from copy import deepcopy 
from torchsummary import summary
from torchvision.models import resnet50, ResNet50_Weights

class CustomResNetServer(nn.Module):
    def __init__(self, original_resnet_server):
        super(CustomResNetServer, self).__init__()
        self.features = nn.Sequential(*list(original_resnet_server.children())[:-2]) # Exclude the last two layers (AvgPool and FC)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = list(original_resnet_server.children())[-1] # The original FC layer

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return image, label

    def __len__(self):
        return len(self.indices)

def get_k_shot_indices(dataset, k, num_classes ,num_clients,replace=False):
    class_examples = [[] for _ in range(num_classes)]
    
    for idx, (_, label) in enumerate(dataset):
        class_examples[label].append(idx)
        
    client_indices = []
    for _ in range(num_clients):
        indices = []
        for class_idx in range(num_classes):
            indices += np.random.choice(class_examples[class_idx], k, replace=replace).tolist()
        client_indices.append(indices)

    return client_indices


def load_data(dataset_name, k_shot, transform, num_clients):
    if dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
        num_classes = 100
    
    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        num_classes = 10
        
    elif dataset_name == "flowers102":
        train_dataset = datasets.Flowers102(root="./data", split='train', download=True, transform=transform)
        num_classes = 102
        val_dataset = datasets.Flowers102(root="./data", split='test', download=True, transform=transform)
    
    elif dataset_name == "Caltech101":
        transform = transforms.Compose([transforms.Resize((224, 224)),transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

        dataset = datasets.Caltech101(root="./data", download=True, transform=transform) 
        
        train_dataset, val_dataset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
        num_classes = 101

    elif dataset_name == "Food101":
        train_dataset = datasets.Food101(root="./data", split='train', download=True, transform=transform)
        num_classes = 101
        val_dataset = datasets.Food101(root="./data", split='test', download=True, transform=transform)

    replace = False
    if dataset_name == "flowers102" :
        replace = True

    indices = get_k_shot_indices(train_dataset, k_shot, num_classes, num_clients, replace=replace)
    client_datasets = [Subset(train_dataset, indices) for indices in indices]

    return client_datasets, val_dataset, num_classes


def get_full_resnet_model():
    # Load a pre-trained ResNet model with default weights
    #model = models.resnet50(pretrained=True)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # for idx, child in enumerate(model.children()):
    #     print(f"Layer {idx}: {child}")
    return model


def assess_device_capabilities(device_type):

    cpu_score = 0.8
    ram_score = 0.5
    battery_score = 0.2

    # Define weights for each resource based on device type
    if device_type == "Mobile": #0.38
        cpu_weight = 0.2
        ram_weight = 0.2
        battery_weight = 0.6
    elif device_type == "Laptop": #0.47
        cpu_weight = 0.3
        ram_weight = 0.3
        battery_weight = 0.4
    elif device_type == "Desktop": #0.56
        cpu_weight = 0.4
        ram_weight = 0.4
        battery_weight = 0.2
    else:
        raise ValueError("Invalid device type")

    # Calculate the overall device capability score
    device_capability_score = (cpu_weight * cpu_score +
                               ram_weight * ram_score +
                               battery_weight * battery_score)

    return device_capability_score

def infer_server_input_size(server_model):
    # Assuming the first layer of the server model is a Conv2d layer
    first_layer = next(server_model.children())
    if isinstance(first_layer, nn.Conv2d):
        # Calculate the input height and width required for the Conv2d layer
        # This is a simple example and might need to be adjusted based on the layer type and structure
        kernel_size = first_layer.kernel_size
        stride = first_layer.stride
        padding = first_layer.padding

        # Example calculation (assumes square input and kernel)
        input_size = (kernel_size[0] - 1) * stride[0] - 2 * padding[0] + 1
        return (input_size, input_size)
    else:
        raise ValueError("Server model's first layer is not a Conv2d layer.")

def split_resnet_model(model, device_capability_score):
    # Define split points based on capability score
    HIGH_RESOURCE_THRESHOLD = 0.50
    MEDIUM_RESOURCE_THRESHOLD = 0.45
    # client_layers = nn.Sequential(*list(model.children())[:7])
    # server_layers = nn.Sequential(*list(model.children())[7:])
    
    model_copy = deepcopy(model)
    
    # Accessing the children of ResNet50 and converting to a list
    #children = list(model.children())
    layers = list(model_copy.children())
    
    if device_capability_score > HIGH_RESOURCE_THRESHOLD:
        # High resource device - more layers on client
        split_idx = 7
        print("desktop")
    elif device_capability_score > MEDIUM_RESOURCE_THRESHOLD:
        # Medium resource device - balanced split
        split_idx = 6
        print("laptop")
    else:
        # Low resource device - more layers on server
        split_idx = 5
        print("mobile")
        
    # Creating client and server models
    # client_layers = nn.Sequential(*layers[:split_idx])
    # server_layers = nn.Sequential(*layers[split_idx:])
    client_layers = nn.Sequential(*list(model.children())[:split_idx])
    server_layers = CustomResNetServer(nn.Sequential(*list(model.children())[split_idx:]))
    
    #print("First layer of server model:", server_layers[0])
    
    #print("client layers", client_layers)
    #print("server layers", server_layers)
    #dummy_data = torch.randn(128, 3, 224, 224)
    
    # Test forward pass
    #client_output = forward_pass_on_client(client_model, dummy_data)
    #server_output = server_model(client_output)
    
    # Print the expected input shape of the client and server layers
#     dummy_input = torch.randn(1, 3, 224, 224)  # Adjust dimensions as needed
#     client_output = client_layers(dummy_input)
#     server_output = server_layers(dummy_input)
    
#     print("Expected Input Shape - Client Layer:", dummy_input.shape)
#     print("Expected Input Shape - Server Layer:", dummy_input.shape)


    return client_layers, server_layers

# Function to perform a forward pass on the client model and generate smashed data
def forward_pass_on_client(client_model, data):
    client_model.eval()
    with torch.no_grad():
        # Forward pass to get the smashed data
        smashed_data = client_model(data)
        # print("Shape of client model output:", smashed_data.shape)
    return smashed_data

# Function to perform a forward and backward pass on the server model and obtain gradients
def forward_and_backward_pass_on_server(server_model, smashed_data, target):
    server_model.train()
    optimizer = torch.optim.Adam(server_model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    #smashed_data = smashed_data.view(smashed_data.size(0), -1)
    #reshaped_smashed_data = smashed_data.view(-1, 3, 32, 32)  # Adjust as per your model's requirement
    
    # Print the shapes
    #print(f"Client Data Shape: {smashed_data.shape}")
    #print(f"Server Model Expected Input Shape: {reshaped_smashed_data.shape}")
    



    optimizer.zero_grad()
    output = server_model(smashed_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # Return the updated server model parameters instead of gradients
    return [param.data for param in server_model.parameters()]

# Function to update the local client model with received gradients
def update_client_model(client_model, server_model_params):
    with torch.no_grad():
        for param, new_param in zip(client_model.parameters(), server_model_params):
            param.data = new_param.clone()
    
# averages model parameters using Federated Averaging
def average_model_updates(model_params_list):
    avg_params = []
    for params in zip(*model_params_list):
        # Averaging the parameters
        avg_params.append(sum(params) / len(params))
    return avg_params


# function applies averaged model parameters to the server model.
def apply_update_to_server_model(server_model, averaged_params):
    with torch.no_grad():
        for param, avg_param in zip(server_model.parameters(), averaged_params):
            param.data = avg_param.clone()
        
# Function to evaluate the performance of the global server model (optional)
def evaluate_model_performance(model, val_dataloader):
    model.eval()
    total_correct, total = 0, 0
    with torch.no_grad():
        for data, target in val_dataloader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            total_correct += (predicted == target).sum().item()
    accuracy = 100 * total_correct / total
    print(f'Validation Accuracy: {accuracy}%')
    return accuracy




#def federated_training(args, model, local_dataloaders, val_dataloader, criterion, device='cuda'):
#def federated_training(server_model, client_models, data_loader, clusters, num_rounds=10):
def federated_training(args, full_model, local_dataloaders, val_dataloader, num_rounds=10,device='cuda'):
    # server_model: Model residing on the server
    # client_models: Dictionary mapping client IDs to their respective client-side models
    # data_loader: Data loader for each client
    # clusters: Grouping of clients into clusters
    # num_rounds: Number of training rounds
    
    
    # Define client clusters based on device types
    client_clusters = {
        "Mobile": local_dataloaders[:len(local_dataloaders) // 3],
        "Laptop": local_dataloaders[len(local_dataloaders) // 3: 2 * len(local_dataloaders) // 3],
        "Desktop": local_dataloaders[2 * len(local_dataloaders) // 3:]
    }

    for round in range(num_rounds):
        print(f"Training Round {round + 1}/{num_rounds}")

        # Step 1: Local Training on Clients
        cluster_updates = defaultdict(list)
        for cluster_type, dataloaders in client_clusters.items():
            capability_score = assess_device_capabilities(cluster_type)
            client_model, server_model = split_resnet_model(full_model, capability_score)
            
            #local_client_model = copy.deepcopy(client_model).to(device)
            #local_client_model = copy.deepcopy(client_model).to(device)
            
            for data_loader in dataloaders:
                i = 1
                for data, target in data_loader:
                    print("epoch ", i)
                    i+=1
                    print("data shape", data.shape)
                    # Local training on client and sending smashed data to server
                    smashed_data = forward_pass_on_client(client_model, data)
                    print("Shape of client model output:", smashed_data.shape)
                    server_model_params = forward_and_backward_pass_on_server(server_model, smashed_data, target)

                 # Update client model with server model parameters
                update_client_model(client_model, server_model_params)

                # Collect server model parameters for aggregation
                cluster_updates[cluster_type].append(server_model_params)

        # Step 2: Intra-Cluster FedAvg
        for cluster_type in client_clusters:
            avg_model_params = average_model_updates(cluster_model_updates[cluster_type])
            
            # Apply averaged model parameters to server model for each cluster
            apply_update_to_server_model(server_model, avg_model_params)

        # Optional: Evaluate model performance after each round
        evaluate_model_performance(server_model, val_dataloader)







def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Test with different device types
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    train_datasets, val_dataset, num_classes = load_data(args.dataset, args.k_shot, transform, args.num_clients)
    train_dataloaders = [DataLoader(train_dataset, batch_size=128, shuffle=True) for train_dataset in train_datasets]
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    full_model = get_full_resnet_model()
    federated_training(args, full_model, train_dataloaders,val_dataloader, num_rounds=10, device='cuda')
        
        
        
        
        #print(f"Device Type: {device_type}")
        #print(f"Device Capability Score: {device_capability_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-shot learning with pre-trained models")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar100", "flowers102", "Caltech101", "cifar10", "Food101"],
                        help="Dataset to use (currently only cifar100 is supported)")
    parser.add_argument("--k_shot", type=int, default=2,
                        help="Number of samples per class for few-shot learning")
    parser.add_argument('--num_clients', type=int, default=10, 
                         help='Number of clients')
    args = parser.parse_args()
    main(args)
    
    
    
#python base_split_cluster_quan.py --model resnet --dataset cifar10 --k_shot 4 --num_epochs 100 --method fl --num_clients 100 --num_local_epochs 10 --num_rounds 100 --num_clients_per_round 10 --lr 1e-4
    
