import torch

# config = {
#     "address":
# }

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('**Using: ', torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print('**Using: cpu')

address = {
    "ip": "192.168.119.128",
    "port": 50025
}

env_params = {
    "max_cycles": 20 * 60
}

user_params = {
    "device": device
}
