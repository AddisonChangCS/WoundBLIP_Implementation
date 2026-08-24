from pathlib import Path
import torch
from omegaconf import OmegaConf, DictConfig
from config import paths

class ConfigManager:
    def __init__(self, config_path: str):
        """
        Args:
            config_path(str): YAML file path
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"config not exist: {self.config_path.resolve()}")
        self._cfg: DictConfig = OmegaConf.load(self.config_path)

        self._resolve_run_dir()
        self._resolve_device()

    @property
    def cfg(self) -> DictConfig:
        return self._cfg
    
    def _resolve_run_dir(self):
        """Resolve and build run dir under result dir
        """
        run_dir_name = self._cfg.run_dir
        run_dir = (paths.RESULT_DIR / run_dir_name).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.update(self._cfg, "run_dir", str(run_dir))

    def _resolve_device(self):
        """
        Detect available hardware and update 'device' field if it is set to 'auto'.
        Priority: CUDA > CPU
        """
        current_device = self._cfg.get("device", "none")

        if current_device == "auto":
            if torch.cuda.is_available():
                resolved = "cuda"
                print(f"[Config]\t Device 'auto' resolved to: CUDA ({torch.cuda.get_device_name(0)})")
            else:
                resolved = "cpu"
                print("[Config]\t Device 'auto' resolved to: CPU")
            
            OmegaConf.update(self._cfg, "device", resolved)

    def save(self, save_path: str):
        """
        save current config
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self._cfg, save_path)

    