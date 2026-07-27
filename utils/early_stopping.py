import torch


class EarlyStopping:

    def __init__(self, patience=3):

        self.patience = patience

        self.counter = 0

        self.best_loss = float("inf")

        self.stop = False

    def __call__(self, loss, model, path):

        if loss < self.best_loss:

            self.best_loss = loss

            self.counter = 0

            torch.save(model.state_dict(), path)

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.stop = True