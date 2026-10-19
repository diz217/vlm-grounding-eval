"""Backends never receive ground-truth labels or boxes."""

import importlib.metadata
import json
import random
from pathlib import Path

from .io import digest, file_sha256


def versions():
    result = {}
    for package in ("torch", "torchvision", "transformers", "huggingface-hub", "Pillow", "PyYAML", "numpy", "safetensors"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


class ReplayBackend:
    """Recorded fixture responses for integration tests, NOT VLM benchmark results."""
    def __init__(self, config, seed):
        self.responses = json.loads(Path(config["responses"]).read_text(encoding="utf-8"))
        if not isinstance(self.responses, dict):
            raise ValueError("Replay responses must map request IDs to strings")
        self.metadata = {"backend": "replay", "model_id": config["id"], "synthetic": True,
                         "responses_sha256": file_sha256(config["responses"]), "versions": versions()}

    def generate(self, request_id, image, prompt):
        answer = self.responses[request_id]
        if not isinstance(answer, str):
            raise ValueError("Replay answer must be a string")
        return {"text": answer, "generated_tokens": None, "truncated": False}


class TransformersBackend:
    def __init__(self, config, seed):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
        from huggingface_hub import snapshot_download
        from transformers.utils.logging import disable_progress_bar

        disable_progress_bar()

        if config["device"].startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; select cpu explicitly")
        self.torch = torch
        self.config = config
        set_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
        common = {"revision": config["revision"], "local_files_only": config["local_files_only"],
                  "trust_remote_code": config["trust_remote_code"]}
        if config.get("cache_dir"):
            common["cache_dir"] = config["cache_dir"]
        if set(config["processor_kwargs"]) & set(common):
            raise ValueError("processor_kwargs cannot override revision or loading policy")
        # Resolve the snapshot first. In Transformers 5.2 AutoProcessor's initial
        # config probe drops some Hub kwargs; a local snapshot avoids accidental
        # requests to floating main even when local_files_only was requested.
        if Path(config["id"]).is_dir():
            snapshot = Path(config["id"]).resolve()
        else:
            snapshot = Path(snapshot_download(config["id"], revision=config["revision"],
                                              cache_dir=config.get("cache_dir"),
                                              local_files_only=config["local_files_only"]))
        local_common = {"local_files_only": True, "trust_remote_code": config["trust_remote_code"]}
        self.processor = AutoProcessor.from_pretrained(str(snapshot), **local_common, **config["processor_kwargs"])
        if config.get("chat_template"):
            self.processor.chat_template = Path(config["chat_template"]).read_text(encoding="utf-8")
        template = getattr(self.processor, "chat_template", None)
        if not template:
            raise ValueError("Processor has no chat template; configure an explicit frozen template")
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(snapshot), **local_common, dtype=getattr(torch, config["dtype"]),
            attn_implementation=config["attention"])
        self.model.to(config["device"]).eval()
        # Freeze checkpoint-level defaults: do not inherit sampling/penalties silently.
        from transformers import GenerationConfig
        generation = GenerationConfig.from_model_config(self.model.config)
        generation.do_sample = False
        generation.num_beams = 1
        generation.max_new_tokens = config["max_new_tokens"]
        for token_name in ("eos_token_id", "pad_token_id", "bos_token_id"):
            value = getattr(self.model.generation_config, token_name, None)
            if value is not None:
                setattr(generation, token_name, value)
        self.generation = generation
        processor_snapshot = self.processor.to_dict() if hasattr(self.processor, "to_dict") else {}
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            processor_snapshot["image_processor"] = image_processor.to_dict()
        self.metadata = {"backend": "transformers", "model_id": config["id"], "synthetic": False,
                         "requested_revision": config["revision"],
                         "resolved_revision": snapshot.name if len(snapshot.name) == 40 else getattr(self.model.config, "_commit_hash", None),
                         "chat_template": template, "chat_template_sha256": digest(template),
                         "processor": processor_snapshot, "generation_config": generation.to_dict(),
                         "versions": versions(), "batch_size": 1,
                         "gpu": torch.cuda.get_device_name() if config["device"].startswith("cuda") else None,
                         "cuda_runtime": torch.version.cuda}

    def generate(self, request_id, image, prompt):
        torch = self.torch
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                   {"type": "text", "text": prompt}]}]
        template_kwargs = self.config["chat_template_kwargs"]
        forbidden = {"tokenize", "return_dict", "return_tensors", "add_generation_prompt"} & set(template_kwargs)
        if forbidden:
            raise ValueError(f"Reserved chat_template_kwargs: {forbidden}")
        rendered = self.processor.apply_chat_template(messages, add_generation_prompt=True,
                                                       tokenize=False, **template_kwargs)
        inputs = self.processor(text=[rendered], images=[image], return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(self.config["device"])
        if self.config["device"].startswith("cuda"):
            torch.cuda.synchronize()
        with torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=self.generation)
        if self.config["device"].startswith("cuda"):
            torch.cuda.synchronize()
        generated = output[0, inputs["input_ids"].shape[-1]:]
        text = self.processor.decode(generated, skip_special_tokens=True)
        eos = self.generation.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        truncated = len(generated) >= self.config["max_new_tokens"] and int(generated[-1]) not in eos
        return {"text": text, "generated_tokens": len(generated), "truncated": truncated,
                "rendered_chat": rendered, "input_tokens": int(inputs["input_ids"].shape[-1]),
                "input_token_ids_sha256": digest(inputs["input_ids"][0].tolist())}


def create_backend(config, seed):
    return (ReplayBackend if config["backend"] == "replay" else TransformersBackend)(config, seed)
