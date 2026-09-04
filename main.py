import torch.nn as nn

from tabulate import tabulate

from modules.data import create_dataloader, get_data
from modules.model import train, TransformerModel, evaluate

########################
### Create and train ###
########################

train_data, test_data = get_data()
train_dataloader = create_dataloader(train_data)
test_dataloader = create_dataloader(test_data, shuffle=False)

LOAD = False

if LOAD:
    model = TransformerModel.load("model/full_rank.pth")
else:
    model = train(train_dataloader, test_dataloader, EPOCHS=10, LR=1e-4, save_path="model/full_rank.pth")


########################
###  Evaluate model  ###
########################

print("\n"*3)
criterion = nn.CrossEntropyLoss()
train_loss, train_char_acc, train_seq_acc = evaluate(model, test_dataloader, criterion)
val_loss, val_char_acc, val_seq_acc = evaluate(model, test_dataloader, criterion)

table_data= []
table_data.append(["Full", train_loss, train_char_acc, train_seq_acc, val_loss, val_char_acc, val_seq_acc])
table_headers = ["Model Rank", "Train Loss", "Train Char Acc", "Train Seq Acc", "Val Loss", "Val Char Acc", "Val Seq Acc"]
print(tabulate(table_data, headers=table_headers, tablefmt="github"))

########################
###  Compress model  ###
########################

RANK = 10


#################################
### Evaluate compressed model ###
#################################

########################
###    Skee plots    ###
########################

