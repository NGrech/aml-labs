from prettytable import PrettyTable
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def count_parameters(model):
    table = PrettyTable(["Modules","Trainable", "Parameters"])
    total_params = 0
    total_trainable_params = 0
    for name, parameter in model.named_parameters():
        trainable = parameter.requires_grad
        params = parameter.numel()
        table.add_row([name, trainable, params])
        total_params+=params
        if trainable: 
            total_trainable_params += params
    print(table)
    print(f"Trainable Params: {total_trainable_params} of {total_params} total params")

def plot_train_test_loss(train_loss, test_loss, plot_title="Train & test Loss"):
    fig = go.Figure()
    epochs_index = list(range(1,len(train_loss)+1))

    fig.add_trace(go.Scatter(
        y=train_loss,
        x=epochs_index,
        name='Train Loss'
    ))

    fig.add_trace(go.Scatter(
        y=test_loss,
        x=epochs_index,
        name='Test Loss'
    ))

    fig.update_layout(
        title=plot_title,
        xaxis_title="Epochs")

    fig.show()

def plot_images(inp, title=None):
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    fig = px.imshow(inp)
    fig.update_layout(title_text=title, title_x=0.5)
    fig.show()
