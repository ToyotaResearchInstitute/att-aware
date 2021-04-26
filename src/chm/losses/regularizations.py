import torch


class EPSpatialRegularization(torch.nn.Module):
    def __init__(self, image_width, image_height, eps=1e-3, sig_scale_factor=1):
        super(EPSpatialRegularization, self).__init__()

    def forward(self):
        import IPython

        IPython.embed(banner1="in EP spatial")


class EPTemporalRegularization(torch.nn.Module):
    def __init__(self, image_width, image_height, eps=1e-3, sig_scale_factor=1):
        super(EPTemporalRegularization, self).__init__()

    def forward(self):
        import IPython

        IPython.embed(banner1="in EP temp")
