import os
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tabulate import tabulate

from modules.data import create_dataloader, get_data
from modules.model import train, Model, evaluate


def format_name(raw_name: str) -> str:
    replacements = {
        "blocks.": "Block ",
        ".attn.": " Attention ",
        ".ffn.": " FFN ",
        "q_proj": "Q Projection",
        "k_proj": "K Projection",
        "v_proj": "V Projection",
        "out_proj": "Output Projection",
        "linear1": "Layer 1",
        "linear2": "Layer 2"
    }
    for old, new in replacements.items():
        raw_name = raw_name.replace(old, new)
    return raw_name.title()


def compress_and_evaluate(base_model, rank_fn, train_loader, test_loader, criterion, skip_val=False, return_model=False):
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

            layer_B = nn.Linear(module.in_features, rank, bias=False).to(W.device)
            layer_A = nn.Linear(rank, module.out_features, bias=(module.bias is not None)).to(W.device)

            layer_B.weight.data = torch.diag(torch.sqrt(D[:rank])) @ V[:rank, :]
            layer_A.weight.data = U[:, :rank] @ torch.diag(torch.sqrt(D[:rank]))
            if module.bias is not None:
                layer_A.bias.data = module.bias.data

            parent = model.get_submodule(name.rsplit(".", 1)[0])
            setattr(parent, name.rsplit(".", 1)[-1], nn.Sequential(layer_B, layer_A))

    train_res = evaluate(model, train_loader, criterion)
    val_res = [0.0, 0.0, 0.0] if skip_val else evaluate(model, test_loader, criterion)
    param_count = sum(p.numel() for p in model.parameters())
    
    if return_model:
        return [*train_res, *val_res], singular_values, param_count, model
    return [*train_res, *val_res], singular_values, param_count


def run_greedy_strategy(base_model, train_loader, test_loader, criterion, acc_floor=0.95, rank_step=50, top_k=5):
    strat_name = f"Greedy{int(acc_floor * 100)}_{rank_step}_{top_k}"
    print(f"Running {strat_name} Strategy...")

    # Unconditionally start all target layers at rank 384
    current_ranks = {
        name: 384
        for name, m in base_model.named_modules()
        if isinstance(m, nn.Linear) and not name.endswith("output")
    }

    step = 0
    while True:
        step += 1
        losses = {}
        
        # 1. Test reducing rank by rank_step for each layer independently
        for name in current_ranks:
            if current_ranks[name] <= rank_step:
                continue
            
            test_ranks = {**current_ranks, name: current_ranks[name] - rank_step}
            results, _, _ = compress_and_evaluate(
                base_model, lambda n, D, r=test_ranks: r[n], train_loader, test_loader, criterion, skip_val=True
            )
            losses[name] = results[0]

        if not losses:
            break
            
        # 2. Pick top_k candidates with lowest training loss
        top_candidates = sorted(losses, key=losses.get)[:top_k]
        for name in top_candidates:
            current_ranks[name] -= rank_step
            
        # 3. Evaluate combined step
        results, _, p_count = compress_and_evaluate(
            base_model, lambda n, D, r=current_ranks: r[n], train_loader, test_loader, criterion, skip_val=True
        )
        
        train_char_acc = results[1]
        
        # 4. Check accuracy threshold
        if train_char_acc <= acc_floor:
            print(f"Greedy Step {step} hit {train_char_acc:.4f} accuracy. Reverting to maintain >{acc_floor:.0%}.")
            for name in top_candidates:
                current_ranks[name] += rank_step
            break

        print(f"Greedy Step {step} (delta={rank_step}) - Train Char Acc: {train_char_acc:.4f}, Params: {p_count}")

    # 5. Full evaluation on the final state & save model
    print(f"Evaluating final {strat_name} configuration...")
    final_results, sv, final_p_count, final_model = compress_and_evaluate(
        base_model, lambda n, D, r=current_ranks: r[n], train_loader, test_loader, criterion, skip_val=False, return_model=True
    )
    
    save_path = f"model/{strat_name}.pth"
    torch.save(final_model.state_dict(), save_path)
    print(f"Saved compressed model to {save_path}")

    return strat_name, final_results, sv, final_p_count, current_ranks


def plot_metrics(x, y, z, w, xlabel, filename):
    ax, fig = plt.subplots(2, 1, figsize=(6, 6))
    fig[0].plot(x, y, color="blue", label="Sequence Accuracy")
    fig[0].plot(x, z, color="red", label="Character Accuracy")
    fig[0].set_title(f"Validation Accuracy vs {xlabel}")
    fig[0].set_xlabel(xlabel)
    fig[0].set_ylabel("Accuracy")
    fig[0].legend()

    fig[1].plot(x, w, color="blue")
    fig[1].set_title(f"Model Parameters Count vs {xlabel}")
    fig[1].set_xlabel(xlabel)
    fig[1].set_ylabel("Model Parameters Count")
    fig[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(f"img/{filename}", dpi=600, bbox_inches="tight")
    plt.close()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("model", exist_ok=True)
os.makedirs("img/scree_plots", exist_ok=True)

train_data, test_data = get_data()
train_dataloader = create_dataloader(train_data)
test_dataloader = create_dataloader(test_data, shuffle=False)

LOAD = False
if LOAD:
    model = Model(vocab_size=32, seq_len=32, d_model=768, n_heads=12, d_ff=3072, n_layers=12).to(device)
    model.load_state_dict(torch.load("model/full_rank.pth", map_location=device, weights_only=True))
else:
    model = train(train_dataloader, test_dataloader, EPOCHS=10, LR=1e-4, save_path="full_rank.pth")

print("Model Loaded\n\n\n")

criterion = nn.CrossEntropyLoss()
train_loss, train_char_acc, train_seq_acc = evaluate(model, train_dataloader, criterion)
val_loss, val_char_acc, val_seq_acc = evaluate(model, test_dataloader, criterion)
param_count = sum(p.numel() for p in model.parameters())

table_data = [["Full", train_loss, train_char_acc, train_seq_acc, val_loss, val_char_acc, val_seq_acc, param_count]]
table_headers = ["Strategy", "Train Loss", "Train Char Acc", "Train Seq Acc", "Val Loss", "Val Char Acc", "Val Seq Acc", "Model Parameters Count"]


########################
###       Rank       ###
########################
print("Running Rank Strategy...")
x_rank = list(range(10, 770, 10))
y_rank, z_rank, w_rank = [], [], []

for i in x_rank:
    strat_name_rank = f"R{i}"
    results, sv_rank, p_count = compress_and_evaluate(model, lambda name, S, rank=i: rank, train_dataloader, test_dataloader, criterion)
    y_rank.append(results[-1])
    z_rank.append(results[-2])
    w_rank.append(p_count)
    table_data.append([strat_name_rank] + results + [p_count])

plot_metrics(x_rank, y_rank, z_rank, w_rank, "Rank", "rank_vs_loss.png")


########################
###      Weight      ###
########################
print("Running Weight Strategy...")
x_eng = list(range(0, 100, 2))
y_eng, z_eng, w_eng = [], [], []

for i in x_eng:
    strat_name_weight = f"Weight{i}"
    rank_fn = lambda name, S, thresh=i: (torch.cumsum(S, dim=0) / torch.sum(S) >= thresh / 100).nonzero(as_tuple=True)[0][0].item() + 1
    
    results, sv_eng, p_count = compress_and_evaluate(model, rank_fn, train_dataloader, test_dataloader, criterion)
    y_eng.append(results[-1])
    z_eng.append(results[-2])
    w_eng.append(p_count)
    table_data.append([strat_name_weight] + results + [p_count])

plot_metrics(x_eng, y_eng, z_eng, w_eng, "Weight Retained (%)", "Weight_vs_loss.png")


########################
###      Greedy      ###
########################
strat_name, results, sv, p_count, final_ranks = run_greedy_strategy(
    model, train_dataloader, test_dataloader, criterion, acc_floor=0.985, rank_step=50, top_k=15
)
table_data.append([strat_name] + results + [p_count])


########################
###      Results     ###
########################

print("\n")
print(tabulate(table_data, headers=table_headers, tablefmt="github"))

print("\n")
rank_table_data = []
for name, S in sv.items():
    rank = final_ranks[name]
    retained_frac = (torch.sum(S[:rank]) / torch.sum(S)).item()
    rank_table_data.append([format_name(name), rank, f"{retained_frac:.4f}"])

print(tabulate(rank_table_data, headers=["Layer", "Greedy Final Rank", "Retained SV Weight"], tablefmt="github"))

########################
###   Heatmap Plot   ###
########################
df = pd.DataFrame(rank_table_data, columns=["Layer", "Rank", "Weight"])
df["Weight"] = df["Weight"].astype(float)

comp_map = {
    "Sa.Q Projection": "Q projection",
    "Sa.K Projection": "K projection",
    "Sa.V Projection": "V projection",
    "Sa.Output Projection": "self attention output",
    "Ffwd.Net.0": "ffwd 0",
    "Ffwd.Net.2": "ffwd 2"
}

df["Block"] = df["Layer"].apply(lambda x: x.split(".")[0])
df["ComponentRaw"] = df["Layer"].apply(lambda x: ".".join(x.split(".")[1:]))
df["Component"] = df["ComponentRaw"].map(comp_map)

cols_order = ["Q projection", "K projection", "V projection", "self attention output", "ffwd 0", "ffwd 2"]
blocks_order = [f"Block {i}" for i in range(12)]

rank_pivot = df.pivot(index="Block", columns="Component", values="Rank").reindex(index=blocks_order, columns=cols_order)
weight_pivot = df.pivot(index="Block", columns="Component", values="Weight").reindex(index=blocks_order, columns=cols_order)

fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(weight_pivot.values, cmap="viridis", aspect="auto")

ax.set_xticks(np.arange(len(cols_order)))
ax.set_yticks(np.arange(len(blocks_order)))
ax.set_xticklabels(cols_order, rotation=45, ha="right")
ax.set_yticklabels(blocks_order)

for i in range(len(blocks_order)):
    for j in range(len(cols_order)):
        text_color = "white" if weight_pivot.values[i, j] < 0.5 else "black"
        ax.text(j, i, int(rank_pivot.values[i, j]),
                ha="center", va="center", color=text_color)

cbar = fig.colorbar(cax, ax=ax)
cbar.set_label("Retained SV Weight")

ax.set_title(f"Matrix Rank and Retained Singular Values Weight for {strat_name}", pad=20, fontsize=14)
ax.spines[:].set_visible(False)

fig.tight_layout()
plt.savefig("img/heatmap.png", dpi=600, bbox_inches="tight")
plt.close()

########################
###   Scree Plots    ###
########################
for name, S in sv.items():
    readable_name = format_name(name)
    rank = final_ranks[name]
    
    # -1 to align with 0-indexed plots. Ensure we don't index negative bounds
    idx = max(0, rank - 1)
    val = S[idx].item()
    
    plt.figure(figsize=(6, 3))
    plt.plot(S.cpu().numpy(), color="blue")
    
    # Dotted red intersection lines
    plt.axvline(x=idx, color="red", linestyle="dotted")
    plt.axhline(y=val, color="red", linestyle="dotted")
    
    plt.title(f"Scree Plot: {readable_name}")
    plt.yscale("log")
    plt.ylabel("Singular Value (Log Scale)")
    plt.xlabel("Index")
    plt.tight_layout()
    plt.savefig(f"img/scree_plots/{readable_name}.png", dpi=600, bbox_inches="tight")
    plt.close()