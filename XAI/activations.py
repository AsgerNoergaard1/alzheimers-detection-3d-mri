"""
Hook-based activation and gradient extractor for target layers.

Used by xai/base_cam.py to collect intermediate feature maps and their
gradients during a forward/backward pass through the model.
"""


class ActivationsAndGradients:
    """
    Register forward hooks on target layers to capture activations
    and gradients needed for gradient-weighted CAM methods.

    Args:
        model: The model to hook.
        target_layers: List of nn.Module layers to observe.
        reshape_transform: Optional callable applied to each activation/gradient
            tensor before storing (e.g. for ViT patch tokens).
    """

    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform

        self.activations = []
        self.gradients = []
        self.handles = []

        for layer in target_layers:
            self.handles.append(layer.register_forward_hook(self._save_activation))
            self.handles.append(layer.register_forward_hook(self._register_grad_hook))

    def _save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def _register_grad_hook(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return

        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
