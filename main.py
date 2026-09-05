import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tabulate import tabulate

from modules.data import create_dataloader, get_data
from modules.model import train, TransformerModel, evaluate

########################
### Create and train ###
########################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, test_data = get_data()
train_dataloader = create_dataloader(train_data)
test_dataloader = create_dataloader(test_data, shuffle=False)

LOAD = False

if LOAD:
    model = TransformerModel.load("model/full_rank.pth").to(device)
else:
    model = train(train_dataloader, test_dataloader, EPOCHS=10, LR=1e-4)


########################
###  Evaluate model  ###
########################

print("\n"*3)
criterion = nn.CrossEntropyLoss()

train_loss, train_char_acc, train_seq_acc = evaluate(
    model, train_dataloader, criterion)
val_loss, val_char_acc, val_seq_acc = evaluate(
    model, test_dataloader, criterion)

table_data = []
table_data.append(["Full", train_loss, train_char_acc, train_seq_acc, val_loss, val_char_acc, val_seq_acc])
table_headers = ["Model Rank", "Train Loss", "Train Char Acc", "Train Seq Acc", "Val Loss", "Val Char Acc", "Val Seq Acc"]


#########################
### Compress function ###
#########################

def get_rank(singular_values, threshold=0.975):
    cum_sum = torch.cumsum(singular_values, dim=0)
    threshold_total = cum_sum[-1] * threshold
    rank = torch.searchsorted(cum_sum, threshold_total).item() + 1
    return rank

########################
###  Compress model  ###
########################

singular_values = {}

rank_table_data = []
for name, module in list(model.named_modules()):
    if isinstance(module, nn.Linear) and not name.endswith("output") and not name.endswith("head"):
        W = module.weight.data

        U, D, V = torch.linalg.svd(W)
        singular_values[name] = D
        rank = get_rank(D, threshold=0.975)
        rank_table_data.append([name, rank, len(D)])

        layer_B = nn.Linear(module.in_features, rank, bias=False).to(W.device)
        layer_A = nn.Linear(rank, module.out_features, bias=(module.bias is not None)).to(W.device)

        layer_B.weight.data = torch.diag(torch.sqrt(D[:rank])) @ V[:rank, :]
        layer_A.weight.data = U[:, :rank] @ torch.diag(torch.sqrt(D[:rank]))
        if module.bias is not None:
            layer_A.bias.data = module.bias.data

        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)
                [-1], nn.Sequential(layer_B, layer_A))

print(tabulate(rank_table_data, headers=["Layer Name", "Rank", "Original Rank"], tablefmt="github"))

#################################
### Evaluate compressed model ###
#################################

train_loss, train_char_acc, train_seq_acc = evaluate(
    model, train_dataloader, criterion)
val_loss, val_char_acc, val_seq_acc = evaluate(
    model, test_dataloader, criterion)

table_data.append(["97.5\% of singular value weight", train_loss, train_char_acc, train_seq_acc, val_loss, val_char_acc, val_seq_acc])
print(tabulate(table_data, headers=table_headers, tablefmt="github"))


########################
###    Skee plots    ###
########################

def format_name(raw_name: str) -> str:
    name = raw_name.replace("blocks.", "Block ")
    name = name.replace(".attn.", " Attention ")
    name = name.replace(".ffn.", " FFN ")
    name = name.replace("q_proj", "Q Projection")
    name = name.replace("k_proj", "K Projection")
    name = name.replace("v_proj", "V Projection")
    name = name.replace("out_proj", "Output Projection")
    name = name.replace("linear1", "Layer 1")
    name = name.replace("linear2", "Layer 2")
    return name.title()


os.makedirs(f"img/scree_plots/{RANK}", exist_ok=True)

for name, D in singular_values.items():
    readable_name = format_name(name)
    plt.figure(figsize=(6, 3))
    plt.plot(D.cpu().numpy())
    plt.title(f"Scree Plot: {readable_name}")
    plt.yscale("log")
    plt.ylabel("Singular Value (Log Scale)")
    plt.xlabel("Index")
    plt.tight_layout()
    plt.savefig(f"img/scree_plots/{RANK}/{readable_name}.png", bbox_inches="tight")
    plt.close()