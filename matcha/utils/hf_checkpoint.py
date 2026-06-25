"""Lightning ModelCheckpoint that also uploads each saved checkpoint to the Hugging Face Hub.

The HF token is read from the ``HF_TOKEN`` environment variable (so it never lands in a
config file or checkpoint). If no token is available the callback degrades gracefully to a
plain ModelCheckpoint and logs a warning, so local runs without HF credentials still work.
"""
import os

from lightning.pytorch.callbacks import ModelCheckpoint

from matcha.utils import pylogger

log = pylogger.get_pylogger(__name__)


class HFModelCheckpoint(ModelCheckpoint):
    def __init__(self, repo_id=None, path_in_repo="checkpoints", private=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repo_id = repo_id
        self.path_in_repo = path_in_repo.rstrip("/")
        self.private = private
        self._api = None
        self._repo_ready = False

    def _ensure_repo(self):
        if self._repo_ready:
            return self._api is not None
        self._repo_ready = True  # only try to set up once
        token = os.environ.get("HF_TOKEN")
        if not self.repo_id or not token:
            log.warning(
                "HFModelCheckpoint: HF_TOKEN or repo_id missing; checkpoints will NOT be pushed to the Hub."
            )
            return False
        try:
            from huggingface_hub import HfApi

            self._api = HfApi(token=token)
            self._api.create_repo(repo_id=self.repo_id, repo_type="model", private=self.private, exist_ok=True)
            log.info(f"HFModelCheckpoint: pushing checkpoints to https://huggingface.co/{self.repo_id}")
        except Exception as e:  # pylint: disable=broad-except
            log.warning(f"HFModelCheckpoint: could not initialise HF repo ({e}); uploads disabled.")
            self._api = None
        return self._api is not None

    def _upload(self, filepath):
        if not self._ensure_repo():
            return
        try:
            name = os.path.basename(filepath)
            self._api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=f"{self.path_in_repo}/{name}",
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"Add {name}",
            )
            log.info(f"HFModelCheckpoint: uploaded {name} to {self.repo_id}")
        except Exception as e:  # pylint: disable=broad-except
            # Never let a transient Hub error kill a training run.
            log.warning(f"HFModelCheckpoint: failed to upload {filepath} ({e}); continuing training.")

    def _delete(self, filepath):
        """Mirror local checkpoint pruning to the Hub so the repo stays bounded (HF holds the
        same `save_top_k` snapshots + last.ckpt as the local volume)."""
        if not self._ensure_repo():
            return
        try:
            from huggingface_hub.utils import EntryNotFoundError

            name = os.path.basename(filepath)
            try:
                self._api.delete_file(
                    path_in_repo=f"{self.path_in_repo}/{name}",
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=f"Prune {name}",
                )
            except EntryNotFoundError:
                pass  # never uploaded (e.g. pruned same step it was created)
        except Exception as e:  # pylint: disable=broad-except
            log.warning(f"HFModelCheckpoint: failed to prune {filepath} on Hub ({e}); continuing.")

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero:
            self._upload(filepath)

    def _remove_checkpoint(self, trainer, filepath):
        super()._remove_checkpoint(trainer, filepath)
        if trainer.is_global_zero:
            self._delete(filepath)
