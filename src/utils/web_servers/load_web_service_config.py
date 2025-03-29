# src/utils/web_servers/load_web_service_config.py
import yaml
from pyprojroot import here

class LoadWebServicesConfig:
    def __init__(self):
        with open(here("configs/web_services.yml")) as cfg:
            self.config = yaml.safe_load(cfg)
        self.llava_service_port = self.config.get("llava_service_port")
        self.rag_reference_service_port = self.config.get("rag_reference_service_port")
        self.stable_diffusion_service_port = self.config.get("stable_diffusion_service_port")
        self.whisper_service = self.config.get("whisper_service")
