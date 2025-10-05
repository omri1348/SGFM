from pathlib import Path
from sgfm.common.utils import PROJECT_ROOT
import hydra
import pytorch_lightning as pl


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.1")
def set_data(cfg):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
    cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup('fit')
    datamodule.setup('test')

if __name__ == "__main__":
    set_data()