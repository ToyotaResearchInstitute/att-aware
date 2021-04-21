class CHMTrainer(object):
    def __init__(self, params_dict):
        pass

    def fit(self, module):
        module.trainer = self

        module.configure_optimizers()