# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("dinov3")


_HPC_PROFILE_ENV_VAR = "DINOV3_HPC_PROFILE"


class ClusterType(Enum):
    CW = "cw"
    SHARK = "shark"
    # Add future HPC profiles here, e.g.:
    # SNELLIUS = "snellius"




_SLURM_PARTITIONS: Dict[ClusterType, str] = {
    ClusterType.CW: "learn",
    ClusterType.SHARK: "PATHgpu",
}

_SLURM_ACCOUNTS: Dict[ClusterType, Optional[str]] = {
    ClusterType.CW: "fair_amaia_cw_explore",
    ClusterType.SHARK: None,
}

_SLURM_QOS: Dict[ClusterType, Optional[str]] = {
    ClusterType.CW: "explore",
    ClusterType.SHARK: None,
}

_CHECKPOINT_DIRNAMES: Dict[ClusterType, str] = {
    ClusterType.CW: "",
    ClusterType.SHARK: "checkpoint",
}

_CPUS_PER_TASK: Dict[ClusterType, int] = {
    ClusterType.CW: 16,
    ClusterType.SHARK: 8,
}

_DEFAULT_GPUS_PER_NODE: Dict[ClusterType, int] = {
    ClusterType.CW: 8,
    ClusterType.SHARK: 3,
}

_NCCL_ENV_OVERRIDES: Dict[ClusterType, Dict[str, str]] = {
    ClusterType.CW: {},
    ClusterType.SHARK: {
        "NCCL_SOCKET_IFNAME": "team0",
    },
}


def _guess_cluster_type() -> ClusterType:
    """Auto-detect the cluster from the environment variable or heuristics."""
    profile = os.environ.get(_HPC_PROFILE_ENV_VAR, "").strip().upper()
    if profile:
        try:
            ct = ClusterType(profile.lower())
            logger.info(f"Cluster profile selected via {_HPC_PROFILE_ENV_VAR}={profile}")
            return ct
        except ValueError:
            logger.warning(
                f"Unknown HPC profile '{profile}' in {_HPC_PROFILE_ENV_VAR}. "
                f"Valid values: {[c.value for c in ClusterType]}. Falling back to auto-detect."
            )

    return ClusterType.CW


def get_cluster_type(
    cluster_type: Optional[ClusterType] = None,
) -> Optional[ClusterType]:
    if cluster_type is None:
        return _guess_cluster_type()
    return cluster_type


def get_slurm_account(cluster_type: Optional[ClusterType] = None) -> Optional[str]:
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type is None:
        return None
    return _SLURM_ACCOUNTS.get(cluster_type)


def get_checkpoint_path(cluster_type: Optional[ClusterType] = None) -> Optional[Path]:
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type is None:
        return None
    dirname = _CHECKPOINT_DIRNAMES.get(cluster_type, "")
    return Path("/") / dirname if dirname else Path("/")


def get_user_checkpoint_path(
    cluster_type: Optional[ClusterType] = None,
) -> Optional[Path]:
    checkpoint_path = get_checkpoint_path(cluster_type)
    if checkpoint_path is None:
        return None
    username = os.environ.get("USER")
    assert username is not None, "USER environment variable must be set"
    return checkpoint_path / username


def get_slurm_qos(cluster_type: Optional[ClusterType] = None) -> Optional[str]:
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type is None:
        return None
    return _SLURM_QOS.get(cluster_type)


def get_slurm_partition(cluster_type: Optional[ClusterType] = None) -> Optional[str]:
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type is None:
        return None
    return _SLURM_PARTITIONS.get(cluster_type)


def get_default_gpus_per_node(cluster_type: Optional[ClusterType] = None) -> int:
    cluster_type = get_cluster_type(cluster_type)
    return _DEFAULT_GPUS_PER_NODE.get(cluster_type, 8)


def get_slurm_executor_parameters(
    nodes: int,
    num_gpus_per_node: int,
    cluster_type: Optional[ClusterType] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Build a dict of executor parameters suitable for ``submitit``."""
    cluster_type = get_cluster_type(cluster_type)

    params: Dict[str, Any] = {
        "mem_gb": 0,  # Request all memory on a node
        "gpus_per_node": num_gpus_per_node,
        "tasks_per_node": num_gpus_per_node,  # one task per GPU
        "cpus_per_task": _CPUS_PER_TASK.get(cluster_type, 10),
        "nodes": nodes,
        "slurm_partition": get_slurm_partition(cluster_type),
    }

    # Cluster-specific QoS / account
    qos = get_slurm_qos(cluster_type)
    if qos is not None:
        params["slurm_qos"] = qos
    account = get_slurm_account(cluster_type)
    if account is not None:
        params["slurm_account"] = account

    # Apply caller overrides
    params.update(kwargs)
    return params


def apply_nccl_env(cluster_type: Optional[ClusterType] = None) -> None:
    """Inject cluster-specific NCCL / networking environment variables.

    Call this early in the training script (before ``dist.init_process_group``).
    """
    cluster_type = get_cluster_type(cluster_type)
    overrides = _NCCL_ENV_OVERRIDES.get(cluster_type, {})
    for key, value in overrides.items():
        os.environ.setdefault(key, value)
        logger.info(f"NCCL env: {key}={os.environ[key]}")
