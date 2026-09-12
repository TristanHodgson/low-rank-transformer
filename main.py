import os
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tabulate import tabulate

from modules.data import create_dataloader, get_data
from modules.model import train, Model, evaluate


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


def compress_and_evaluate(base_model, rank_fn, train_loader, test_loader, criterion):
    model = copy.deepcopy(base_model).to(device)
    singular_values = {}
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not name.endswith("output"):
            W = module.weight.data

            U, D, V = torch.linalg.svd(W, full_matrices=False)
            singular_values[name] = D

            rank = rank_fn(name, D)

            if rank >= min(module.in_features, module.out_features):
                continue

            layer_B = nn.Linear(module.in_features, rank,
                                bias=False).to(W.device)
            layer_A = nn.Linear(rank, module.out_features, bias=(
                module.bias is not None)).to(W.device)

            layer_B.weight.data = torch.diag(
                torch.sqrt(D[:rank])) @ V[:rank, :]
            layer_A.weight.data = U[:,
                                    :rank] @ torch.diag(torch.sqrt(D[:rank]))
            if module.bias is not None:
                layer_A.bias.data = module.bias.data

            parent = model.get_submodule(name.rsplit(".", 1)[0])
            setattr(parent, name.rsplit(".", 1)
                    [-1], nn.Sequential(layer_B, layer_A))

    train_res = evaluate(model, train_loader, criterion)
    val_res = evaluate(model, test_loader, criterion)
    param_count = sum(p.numel() for p in model.parameters())
    return [*train_res, *val_res], singular_values, param_count


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, test_data = get_data()
train_dataloader = create_dataloader(train_data)
test_dataloader = create_dataloader(test_data, shuffle=False)


LOAD = False
if LOAD:
    model = Model(
        vocab_size=32,
        seq_len=32,
        d_model=768,
        n_heads=12,
        d_ff=3072,
        n_layers=12,
    ).to(device)

    model.load_state_dict(torch.load(
        "model/full_rank.pth", map_location=device, weights_only=True))
else:
    model = train(train_dataloader, test_dataloader, EPOCHS=10,LR=1e-4, save_path="full_rank.pth")
print("Model Loaded")

print("\n"*3)
criterion = nn.CrossEntropyLoss()
train_loss, train_char_acc, train_seq_acc = evaluate(
    model, train_dataloader, criterion)
val_loss, val_char_acc, val_seq_acc = evaluate(
    model, test_dataloader, criterion)
param_count = sum(p.numel() for p in model.parameters())


table_data = [["Full", train_loss, train_char_acc, train_seq_acc, val_loss, val_char_acc, val_seq_acc, param_count]]
table_headers = ["Strategy", "Train Loss", "Train Char Acc", "Train Seq Acc", "Val Loss", "Val Char Acc", "Val Seq Acc", "Model Parameters Count"]


########################
###       Rank       ###
########################

x = [i for i in range(10, 770, 50)]
y, z, w = [], [], []

STRATEGIES = {"R" + str(i): lambda name, S, i=i: i for i in x}


for strat_name, rank_fn in STRATEGIES.items():
    print(f"{strat_name}")
    results, sv, param_count = compress_and_evaluate(model, rank_fn, train_dataloader, test_dataloader, criterion)
    y.append(results[-1])
    z.append(results[-2])
    w.append(param_count)
    table_data.append([strat_name] + results + [param_count])

ax, fig = plt.subplots(2, 1, figsize=(6, 6))
fig[0].plot(x, y, color="blue", label="Sequence Accuracy")
fig[0].plot(x, z, color="orange", label="Character Accuracy")
fig[0].set_title("Validation Accuracy vs Rank")
fig[0].set_xlabel("Rank")
fig[0].set_ylabel("Accuracy")
fig[0].legend()

fig[1].plot(x, w, color="blue")
fig[1].set_title("Model Parameters Count vs Rank")
fig[1].set_xlabel("Rank")
fig[1].set_ylabel("Model Parameters Count")
fig[1].set_yscale("log")

plt.tight_layout()
plt.savefig("img/rank_vs_loss.png", dpi=600, bbox_inches="tight")
plt.close()



########################
###      Energy      ###
########################

x = [i for i in range(0, 100, 5)]
y, z, w = [], [], []


STRATEGIES = {"Energy" + str(i): lambda name, S, i=i: (torch.cumsum(S, dim=0) / torch.sum(
    S) >= i / 100).nonzero(as_tuple=True)[0][0].item() + 1 for i in x}

for strat_name, rank_fn in STRATEGIES.items():
    print(f"{strat_name}")
    results, sv, param_count = compress_and_evaluate(model, rank_fn, train_dataloader, test_dataloader, criterion)
    y.append(results[-1])
    z.append(results[-2])
    w.append(param_count)
    table_data.append([strat_name] + results + [param_count])

ax, fig = plt.subplots(2, 1, figsize=(6, 6))
fig[0].plot(x, y, color="blue", label="Sequence Accuracy")
fig[0].plot(x, z, color="orange", label="Character Accuracy")
fig[0].set_title("Validation Accuracy vs Energy Retained (%)")
fig[0].set_xlabel("Energy Retained (%)")
fig[0].set_ylabel("Accuracy")
fig[0].legend()

fig[1].plot(x, w, color="blue")
fig[1].set_title("Model Parameters Count vs Energy Retained (%)")
fig[1].set_xlabel("Energy Retained (%)")
fig[1].set_ylabel("Model Parameters Count")
fig[1].set_yscale("log")

plt.tight_layout()
plt.savefig("img/rank_vs_loss.png", dpi=600, bbox_inches="tight")
plt.close()

########################
###      Results     ###
########################

print(tabulate(table_data, headers=table_headers, tablefmt="github"))

os.makedirs("img/scree_plots", exist_ok=True)
for name, S in sv.items():
    readable_name = format_name(name)
    plt.figure(figsize=(6, 3))
    plt.plot(S.cpu().numpy(), color="blue")
    plt.title(f"Scree Plot: {readable_name}")
    plt.yscale("log")
    plt.ylabel("Singular Value (Log Scale)")
    plt.xlabel("Index")
    plt.tight_layout()
    plt.savefig(f"img/scree_plots/{readable_name}.png",
                dpi=600, bbox_inches="tight")
    plt.close()
