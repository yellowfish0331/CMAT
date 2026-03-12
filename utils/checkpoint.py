#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import logging
import os
from collections import defaultdict
import torch
import torch.nn as nn

from typing import Any
from typing import Optional, List, Dict, NamedTuple, Tuple, Iterable

from termcolor import colored

def get_missing_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the model but not found in a checkpoint.
    Args:
        keys (list[str]): List of keys that were not found in the checkpoint.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "Some model parameters or buffers are not found in the checkpoint:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "blue") for k, v in groups.items()
    )
    return msg


def get_unexpected_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the checkpoint but not found in the model.
    Args:
        keys (list[str]): List of keys that were not found in the model.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "magenta") for k, v in groups.items()
    )
    return msg


def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> None:
    """
    Strip the prefix in metadata, if any.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix):]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata  # pyre-ignore
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)


def _group_checkpoint_keys(keys: List[str]) -> Dict[str, List[str]]:
    """
    Group keys based on common prefixes. A prefix is the string up to the final
    "." in each key.
    Args:
        keys (list[str]): list of parameter names, i.e. keys in the model
            checkpoint dict.
    Returns:
        dict[list]: keys with common prefixes are grouped into lists.
    """
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1:]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def _group_to_str(group: List[str]) -> str:
    """
    Format a group of parameter name suffixes into a loggable string.
    Args:
        group (list[str]): list of parameter name suffixes.
    Returns:
        str: formated string.
    """
    if len(group) == 0:
        return ""

    if len(group) == 1:
        return "." + group[0]

    return ".{" + ", ".join(group) + "}"


def _named_modules_with_dup(
        model: nn.Module, prefix: str = ""
) -> Iterable[Tuple[str, nn.Module]]:
    """
    The same as `model.named_modules()`, except that it includes
    duplicated modules that have more than one name.
    """
    yield prefix, model
    for name, module in model._modules.items():  # pyre-ignore
        if module is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        yield from _named_modules_with_dup(module, submodule_prefix)


def load_model_from_ckpt(model: nn.Module, ckpt_path: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Load a model from a checkpoint.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Loading checkpoint from {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Handle different checkpoint structures
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "base_model" in checkpoint:
        # Handle Point-MAE checkpoint structure
        state_dict = checkpoint["base_model"]
    else:
        state_dict = checkpoint

    # Remove module. prefix if present
    _strip_prefix_if_present(state_dict, "module.")
    
    # For Point-MAE, we need to extract only the MAE_encoder part
    if any(key.startswith("MAE_encoder.") for key in state_dict.keys()):
        mae_encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("MAE_encoder."):
                # Remove the "MAE_encoder." prefix
                new_key = key[len("MAE_encoder."):]
                mae_encoder_state_dict[new_key] = value
        state_dict = mae_encoder_state_dict
    
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    
    if incompatible_keys.missing_keys:
        logger.warning(f"Missing keys when loading checkpoint: {incompatible_keys.missing_keys}")
    if incompatible_keys.unexpected_keys:
        logger.warning(f"Unexpected keys when loading checkpoint: {incompatible_keys.unexpected_keys}")
    
    logger.info("Checkpoint loaded successfully")