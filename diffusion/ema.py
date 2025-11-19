"""
Exponential Moving Average (EMA) utility for PyTorch models.

This module tracks a smoothed (averaged) version of a model's parameters over time.
EMA is commonly used in diffusion models, GANs, and other deep-learning setups to
stabilize training and obtain better final model weights.

How it works:
-------------
Given a decay constant mu (typically very close to 1, e.g., 0.999 or 0.9999), the
EMA updates parameters using the rule:
    shadow = mu * shadow + (1 - mu) * param
Where:
    - "param" is the model's current parameter
    - "shadow" is the exponentially averaged parameter tracked by EMA

The shadow parameters are *not* used during training unless explicitly swapped
into the model.

This file implements:
    - register(module): initialize EMA shadows for a model
    - update(module): update EMA parameters after each optimizer step
    - ema(module): overwrite model parameters with EMA parameters
    - ema_copy(module): return a fresh copy of the model with EMA parameters applied
    - state_dict() / load_state_dict(): save or load EMA weights
"""

import torch


class EMA(object):
    def __init__(self, mu=0.999):
        """
        Initialize the EMA tracker.

        Parameters
        ----------
        mu : float
            The EMA decay rate. Values close to 1 mean slow updates
            (long-term averaging), while smaller values update faster.
        """
        self.mu = mu              # Decay rate
        self.shadow = {}          # Dictionary storing the EMA parameters

    # ----------------------------------------------------------------------
    def register(self, module):
        """
        Initialize EMA shadow weights using the parameters of the given module.

        This should be called once before training begins.
        """
        for name, param in module.named_parameters():
            if param.requires_grad:
                # Clone the parameter data so we decouple from the live parameter
                self.shadow[name] = param.data.clone()

    # ----------------------------------------------------------------------
    def update(self, module):
        """
        Update EMA parameters using the module's latest parameters.

        Call this AFTER each optimizer step.
        """
        for name, param in module.named_parameters():
            if param.requires_grad:
                # EMA update: shadow = mu * shadow + (1 - mu) * param
                new_average = (
                    (1. - self.mu) * param.data + self.mu * self.shadow[name].data
                )
                self.shadow[name].data = new_average

    # ----------------------------------------------------------------------
    def ema(self, module):
        """
        Copy EMA parameters into the given module IN-PLACE.

        After calling this, the model will run using its EMA-averaged weights.
        Be careful: this permanently overwrites the module's parameters.
        """
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    # ----------------------------------------------------------------------
    def ema_copy(self, module):
        """
        Return a new independent copy of the module with EMA parameters applied.
        Commonly used at evaluation time to avoid overwriting the training model.
        Assumes the module can be reconstructed by calling type(module)(module.config).
        """
        module_copy = type(module)(module.config).to(module.config.device)
        # Load original model parameters first
        module_copy.load_state_dict(module.state_dict())
        # Then overwrite them with EMA parameters
        self.ema(module_copy)
        return module_copy

    # ----------------------------------------------------------------------
    def state_dict(self):
        """
        Return the stored EMA parameters so they can be saved.
        """
        return self.shadow

    # ----------------------------------------------------------------------
    def load_state_dict(self, state_dict):
        """
        Load previously stored EMA parameters.
        """
        self.shadow = state_dict
