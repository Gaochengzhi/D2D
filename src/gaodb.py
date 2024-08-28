import os
import json
from datetime import datetime
import multiprocessing


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class gaodb(metaclass=Singleton):
    _initialized = False
    _lock = multiprocessing.Lock()  # Class-level lock for multiprocessing

    def __init__(self, config_path="../log", project="", task=""):
        if self._initialized:
            return  # Skip re-initialization

        if config_path is None or project is None or task is None:
            raise ValueError(
                "Must initialize with config_path, project_name, and task_name parameters"
            )

        self.config_path = config_path
        self.project_name = project
        self.task_name = task
        self.log_file_path = self._generate_log_file_path()
        self._initialized = True

    def _generate_log_file_path(self):
        os.makedirs(self.config_path, exist_ok=True)
        log_files = [
            f
            for f in os.listdir(self.config_path)
            if f.startswith(f"{self.project_name}_{self.task_name}")
        ]
        file_index = len(log_files) + 1
        return os.path.join(
            self.config_path, f"{self.project_name}_{self.task_name}_{file_index}.log"
        )

    def log(self, entry):
        with self._lock:  # Acquire the lock before writing
            with open(self.log_file_path, "a") as log_file:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                entry_with_timestamp = {"timestamp": timestamp, **entry}
                log_file.write(json.dumps(entry_with_timestamp) + "\n")
                log_file.flush()

    @classmethod
    def init(cls, config_path="../log", project="", task=""):
        return cls(config_path, project, task)

    @classmethod
    def get(cls):
        if not hasattr(cls, "_instances") or cls not in cls._instances:
            raise RuntimeError(
                "LocalWandB has not been initialized. Call init() first."
            )
        return cls._instances[cls]
