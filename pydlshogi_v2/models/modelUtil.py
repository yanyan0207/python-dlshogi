import pandas as pd
import pickle
import pickle
import pandas as pd


class History_trained_model(object):
    def __init__(self, history, epoch, params):
        self.history = history
        self.epoch = epoch
        self.params = params


def save_history(history, path):
    if path:
        if path.endswith('.csv'):
            pd.DataFrame.from_dict(history.history).to_csv(path, index=False)
        elif path.endswith('.pickle'):
            with open(path, 'wb') as file:
                model_history = History_trained_model(
                    history.history, history.epoch, history.params)
                pickle.dump(model_history, file, pickle.HIGHEST_PROTOCOL)
