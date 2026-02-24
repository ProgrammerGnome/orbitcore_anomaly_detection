import torch
from anomalib import LearningType
from anomalib.data import Batch
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from lightning.pytorch.utilities.types import STEP_OUTPUT

from .torch_model import OrbitCoreModel

class OrbitCoreLightning(MemoryBankMixin, AnomalibModule):
    def __init__(self, layers=["layer2", "layer3"], target_dim=128, coreset_sampling_ratio=0.01, use_srp=True, auto_optimize=False):
        super().__init__(
            pre_processor=True,
            post_processor=True,
            evaluator=True,
            visualizer=True
        )
        self.auto_optimize = auto_optimize
        self.model = OrbitCoreModel(
            layers=layers, 
            target_dim=target_dim, 
            coreset_sampling_ratio=coreset_sampling_ratio,
            use_srp=use_srp
        )
        self.embeddings = []

    @property
    def trainer_arguments(self) -> dict:
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0, "devices": 1}

    @property
    def learning_type(self) -> LearningType:
        return LearningType.ONE_CLASS

    def configure_optimizers(self):
        return None

    def training_step(self, batch: Batch, *args, **kwargs):
        features = self.model(batch.image)
        # ---> FIX: CPU-RA MENTJÜK A KINYERT JELLEMZŐKET <---
        #self.embeddings.append(features.cpu())
        self.embeddings.append(features)
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        if len(self.embeddings) == 0:
            return
            
        embeddings_cat = torch.cat(self.embeddings, dim=0)
        self.model.fit_coreset(embeddings_cat)
        self.embeddings.clear()
        # ---> FIX: Takarítás <---
        #torch.cuda.empty_cache()

    def on_test_start(self) -> None:
        if self.auto_optimize:
            print("\n[Lightning] Automatikus Grid Search engedélyezve. Irányítás átadása a belső motornak...")
            train_loader = self.trainer.datamodule.train_dataloader()
            test_loader = self.trainer.datamodule.test_dataloader()
            self.model.optimize_hyperparameters(train_loader, test_loader, self.pre_processor, self.device)

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        anomaly_scores, anomaly_maps = self.model(batch.image)
        return batch.update(
            pred_score=anomaly_scores, 
            anomaly_map=anomaly_maps.squeeze(1)
        )

    def test_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        return self.validation_step(batch, batch_idx, *args, **kwargs)