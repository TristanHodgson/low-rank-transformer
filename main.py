from torch.utils.data import DataLoader

from modules.data import get_data


########################
### Loading the data ###
########################

def create_dataloader(data, batch_size=32, shuffle=True):
    data.set_format(type="torch", columns=["encrypted_tokens", "tokens"])
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


train, test = get_data()
train = create_dataloader(train)
test = create_dataloader(test)

batch = next(iter(test))
print(batch.keys()) 
print(batch["encrypted_tokens"].shape)
print(batch["tokens"].shape)

print(batch)

print("Batch of encrypted tokens:", batch["encrypted_tokens"])

