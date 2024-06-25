import asyncio
import logging
import os
import time
import subprocess
from abc import ABC
from dataclasses import asdict, dataclass, field
from logging import getLogger
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import AsyncInferenceClient
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from .utils import get_free_port, styled_logs

LOGGER = getLogger("Inference-Server")
logging.basicConfig(level=logging.INFO)


@dataclass
class InferenceServerConfig:
    # Common options
    model_id: Optional[str] = None
    revision: Optional[str] = "main"
    # Image to use for the container
    image: Optional[str] = None
    # Shared memory size for the container
    shm_size: Optional[str] = None
    # List of custom devices to forward to the container e.g. ["/dev/kfd", "/dev/dri"] for ROCm
    devices: Optional[List[str]] = None
    # NVIDIA-docker GPU device options e.g. "all" (all) or "0,1,2,3" (ids) or 4 (count)
    gpus: Optional[Union[str, int]] = None

    ports: Dict[str, Any] = field(
        default_factory=lambda: {"80/tcp": ("0.0.0.0", 0)},
        metadata={"help": "Dictionary of ports to expose from the container."},
    )
    volumes: Dict[str, Any] = field(
        default_factory=lambda: {HUGGINGFACE_HUB_CACHE: {"bind": "/data", "mode": "rw"}},
        metadata={"help": "Dictionary of volumes to mount inside the container."},
    )
    environment: List[str] = field(
        default_factory=lambda: ["HUGGINGFACE_HUB_TOKEN"],
        metadata={"help": "List of environment variables to forward to the container."},
    )

    max_concurrent_requests: Optional[int] = None
    timeout: int = 60

    def __post_init__(self) -> None:
        if self.ports["80/tcp"][1] == 0:
            LOGGER.info("\t+ Getting a free port for the server")
            self.ports["80/tcp"] = (self.ports["80/tcp"][0], get_free_port())

        if self.shm_size is None:
            LOGGER.warning("\t+ Shared memory size not provided. Defaulting to '1g'.")
            self.shm_size = "1g"


class InferenceServer(ABC):
    NAME: str = "Inference-Server"
    SUCCESS_SENTINEL: str = "Success"
    FAILURE_SENTINEL: str = "Failure"

    def __init__(self, config: InferenceServerConfig) -> None:
        self.config = config

        LOGGER.info(f"\t+ Building {self.NAME} command")
        self.command = []

        if self.config.model_id is not None:
            self.command = ["--model-id", self.config.model_id]
        if self.config.revision is not None:
            self.command.extend(["--revision", self.config.revision])

        for k, v in asdict(self.config).items():
            if k in InferenceServerConfig.__annotations__:
                continue
            elif v is not None:
                if isinstance(v, bool) and not k == "sharded":
                    self.command.append(f"--{k.replace('_', '-')}")
                else:
                    self.command.append(f"--{k.replace('_', '-')}={str(v).lower()}")

        self.command.append("--json-output")

        LOGGER.info(f"\t+ Building {self.NAME} environment")
        self.environment = {}
        for key in self.config.environment:
            if key in os.environ:
                self.environment[key] = os.environ[key]
            else:
                LOGGER.warning(f"\t+ Environment variable {key} not found in the system")

        self.command = ["text-generation-launcher"] + self.command

        LOGGER.info(f"\t+ Running {self.NAME} process")
        self.process = subprocess.Popen(args=self.command, stdout=subprocess.PIPE)

        LOGGER.info(f"\t+ Streaming {self.NAME} server logs")
        for line in iter(lambda: self.process.stdout.readline(), b""):
            log = line.decode("utf-8").strip()
            log = styled_logs(log)

            if self.SUCCESS_SENTINEL.lower() in log.lower():
                LOGGER.info(f"\t+ {log}")
                break
            elif self.FAILURE_SENTINEL.lower() in log.lower():
                LOGGER.info(f"\t+ {log}")
                raise Exception(f"{self.NAME} server failed to start")
            else:
                LOGGER.info(f"\t+ {log}")

        address, port = "localhost", "80"
        self.url = f"http://{address}:{port}"

        try:
            asyncio.set_event_loop(asyncio.get_event_loop())
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        LOGGER.info(f"\t+ Waiting for {self.NAME} server to be ready")
        start_time = time.time()
        while time.time() - start_time < self.config.timeout:
            try:
                if not hasattr(self, "client"):
                    LOGGER.info(f"\t+ Trying to connect to {self.url}")
                    self.client = AsyncInferenceClient(model=self.url)

                asyncio.run(self.single_client_call(f"Hello {self.NAME}!"))
                LOGGER.info(f"\t+ Connected to {self.NAME} server successfully")
                break
            except Exception:
                LOGGER.info(f"\t+ {self.NAME} server is not ready yet, waiting 1 second")
                time.sleep(1)

    async def single_client_call(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    async def batch_client_call(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        if hasattr(self, "process"):
            LOGGER.info("\t+ Stoping Process")
            if self.process.poll() is not None:
                self.process.kill()
            LOGGER.info("\t+ process stopped")

        if hasattr(self, "semaphore"):
            if self.semaphore.locked():
                self.semaphore.release()
            del self.semaphore

        if hasattr(self, "client"):
            del self.client

    def __del__(self) -> None:
        self.close()
