import os
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tabulate import tabulate

from modules.data import create_dataloader, get_data
from modules.model import train, TransformerModel, evaluate

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
    rank_table_data = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not name.endswith("output") and not name.endswith("head"):
            W = module.weight.data
            
            U, D, V = torch.linalg.svd(W, full_matrices=False)
            singular_values[name] = D
            
            rank = rank_fn(name, D)
            rank_table_data.append([format_name(name), rank, len(D), name])
            
            if rank >= min(module.in_features, module.out_features):
                continue
            
            layer_B = nn.Linear(module.in_features, rank, bias=False).to(W.device)
            layer_A = nn.Linear(rank, module.out_features, bias=(module.bias is not None)).to(W.device)
            
            layer_B.weight.data = torch.diag(torch.sqrt(D[:rank])) @ V[:rank, :]
            layer_A.weight.data = U[:, :rank] @ torch.diag(torch.sqrt(D[:rank]))
            if module.bias is not None:
                layer_A.bias.data = module.bias.data

            parent = model.get_submodule(name.rsplit(".", 1)[0])
            setattr(parent, name.rsplit(".", 1)[-1], nn.Sequential(layer_B, layer_A))

    train_res = evaluate(model, train_loader, criterion)
    val_res = evaluate(model, test_loader, criterion)
    print(tabulate(rank_table_data, headers=["Layer", "Rank", "Original Rank", "Module Name"], tablefmt="github"))
    return [*train_res, *val_res], singular_values



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, test_data = get_data()
train_dataloader = create_dataloader(train_data)
test_dataloader = create_dataloader(test_data, shuffle=False)



LOAD = False
if LOAD: 
    model = TransformerModel.load("model/full_rank.pth").to(device)
else: 
    model = train(train_dataloader, test_dataloader, EPOCHS=10, LR=1e-4, save_path="full_rank.pth")



print("\n"*3)
criterion = nn.CrossEntropyLoss()
train_loss, train_char_acc, train_seq_acc = evaluate(model, train_dataloader, criterion)
val_loss, val_char_acc, val_seq_acc = evaluate(model, test_dataloader, criterion)





table_data = [["Full", train_loss, train_char_acc, train_seq_acc, val_loss, val_char_acc, val_seq_acc]]
table_headers = ["Strategy", "Train Loss", "Train Char Acc", "Train Seq Acc", "Val Loss", "Val Char Acc", "Val Seq Acc"]

STRATEGIES = {
    "R10": lambda name, S: 10,
    "R100": lambda name, S: 100,
    "R200": lambda name, S: 200,
    "R300": lambda name, S: 300,
    "R500": lambda name, S: 500,
    "R600": lambda name, S: 600,
    "R700": lambda name, S: 700,


    "Energy95": lambda name, S: (torch.cumsum(S, dim=0) / torch.sum(S) >= 0.95).nonzero(as_tuple=True)[0][0].item() + 1,
    "Energy90": lambda name, S: (torch.cumsum(S, dim=0) / torch.sum(S) >= 0.90).nonzero(as_tuple=True)[0][0].item() + 1,
    "Energy85": lambda name, S: (torch.cumsum(S, dim=0) / torch.sum(S) >= 0.85).nonzero(as_tuple=True)[0][0].item() + 1,
    "Energy80": lambda name, S: (torch.cumsum(S, dim=0) / torch.sum(S) >= 0.80).nonzero(as_tuple=True)[0][0].item() + 1,
    "Energy75": lambda name, S: (torch.cumsum(S, dim=0) / torch.sum(S) >= 0.75).nonzero(as_tuple=True)[0][0].item() + 1,
    "Energy70": lambda name, S: (torch.cumsum(S, dim=0) / torch.sum(S) >= 0.70).nonzero(as_tuple=True)[0][0].item() + 1,
    "Energy65": lambda name, S: (torch.cumsum(S, dim=0) / torch.sum(S) >= 0.65).nonzero(as_tuple=True)[0][0].item() + 1,
    "Energy60": lambda name, S: (torch.cumsum(S, dim=0) / torch.sum(S) >= 0.60).nonzero(as_tuple=True)[0][0].item() + 1,
    "Energy55": lambda name, S: (torch.cumsum(S, dim=0) / torch.sum(S) >= 0.55).nonzero(as_tuple=True)[0][0].item() + 1,
    "Energy50": lambda name, S: (torch.cumsum(S, dim=0) / torch.sum(S) >= 0.50).nonzero(as_tuple=True)[0][0].item() + 1,

    "Q10": lambda name, S: 10 if "q_proj" in name else len(S),
    "K10": lambda name, S: 10 if "k_proj" in name else len(S),
    "V10": lambda name, S: 10 if "v_proj" in name else len(S),

    "Q100": lambda name, S: 100 if "q_proj" in name else len(S),
    "K100": lambda name, S: 100 if "k_proj" in name else len(S),
    "V100": lambda name, S: 100 if "v_proj" in name else len(S),

    "Q200": lambda name, S: 200 if "q_proj" in name else len(S),
    "K200": lambda name, S: 200 if "k_proj" in name else len(S),
    "V200": lambda name, S: 200 if "v_proj" in name else len(S),

}


saved_sv = None

for strat_name, rank_fn in STRATEGIES.items():
    print(f"\n\n\n\n ### {strat_name}")
    results, sv = compress_and_evaluate(model, rank_fn, train_dataloader, test_dataloader, criterion)
    table_data.append([strat_name] + results)
    if saved_sv is None:
        saved_sv = sv

print(tabulate(table_data, headers=table_headers, tablefmt="github"))

os.makedirs("img/scree_plots", exist_ok=True)
for name, S in saved_sv.items():
    readable_name = format_name(name)
    plt.figure(figsize=(6, 3))
    plt.plot(S.cpu().numpy(), color="blue")
    plt.title(f"Scree Plot: {readable_name}")
    plt.yscale("log")
    plt.ylabel("Singular Value (Log Scale)")
    plt.xlabel("Index")
    plt.tight_layout()
    plt.savefig(f"img/scree_plots/{readable_name}.png", dpi=600, bbox_inches="tight")
    plt.close()