import torch

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class FreezeForLatentWorldModelMomAD6sHook(Hook):

    def __init__(
        self,
        train_keywords=(
            "motion_plan_head.latent_world_model",
            "motion_plan_head.final_fusion_bias",
        ),
        freeze_batch_norm=True,
    ):
        self.train_keywords = tuple(train_keywords)
        self.freeze_batch_norm = freeze_batch_norm

    @staticmethod
    def _unwrap(model):
        if hasattr(model, "module"):
            return model.module
        return model

    def before_run(self, runner):
        model = self._unwrap(runner.model)

        # Freeze the complete pretrained MomAD model.
        for _, parameter in model.named_parameters():
            parameter.requires_grad_(False)

        # Enable only the new latent world model and fusion gate.
        trainable_names = []

        for name, parameter in model.named_parameters():
            if any(keyword in name for keyword in self.train_keywords):
                parameter.requires_grad_(True)
                trainable_names.append(name)

        if not trainable_names:
            raise RuntimeError(
                "No trainable world-model parameters were found. "
                f"Keywords: {self.train_keywords}"
            )

        runner.logger.info(
            "Trainable world-model parameters:\n%s",
            "\n".join(trainable_names),
        )

        self._freeze_bn(model)

    def before_train_epoch(self, runner):
        self._freeze_bn(self._unwrap(runner.model))

    def _freeze_bn(self, model):
        if not self.freeze_batch_norm:
            return

        for module in model.modules():
            if isinstance(
                module,
                torch.nn.modules.batchnorm._BatchNorm,
            ):
                module.eval()

                for parameter in module.parameters():
                    parameter.requires_grad_(False)
