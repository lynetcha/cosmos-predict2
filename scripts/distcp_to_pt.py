# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

import os

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

checkpoints_list = [
    "Cosmos-Predict2-2B-TextVideo2World-Multiview",
]

for checkpoint_name in checkpoints_list:
    if not os.path.exists(os.path.join("checkpoints", checkpoint_name, "model")):
        print(f"Distributed checkpoint directory for {checkpoint_name} does not exist, skipping.")
        continue

    print(f"Converting {checkpoint_name}...")
    dcp_checkpoint_dir = os.path.join("checkpoints", checkpoint_name, "model")

    torch_save_path_ema_reg = os.path.join("checkpoints", checkpoint_name, "model_ema_reg.pt")
    torch_save_path_ema_only_fp32 = torch_save_path_ema_reg.replace("_ema_reg.pt", "_fp32.pt")
    torch_save_path_ema_only_bf16 = torch_save_path_ema_reg.replace("_ema_reg.pt", ".pt")

    # 1. Convert distributed checkpoint to torch single checkpoint
    if os.path.exists(torch_save_path_ema_reg):
        print(f"{torch_save_path_ema_reg} already exists, skipping.")
    else:
        dcp_to_torch_save(dcp_checkpoint_dir, torch_save_path_ema_reg)
        print(f"Converted {dcp_checkpoint_dir} to {torch_save_path_ema_reg}")

    # 2. drop Reg keys and save EMA only in fp32 precision
    if os.path.exists(torch_save_path_ema_only_fp32):
        print(f"{torch_save_path_ema_only_fp32} already exists, skipping.")
        state_dict_ema_only_fp32 = torch.load(torch_save_path_ema_only_fp32, map_location="cpu", weights_only=False)
    else:
        state_dict_ema_reg = torch.load(torch_save_path_ema_reg, map_location="cpu", weights_only=False)
        keys = list(state_dict_ema_reg.keys())

        n_keys = len(keys)
        net_count = 0
        net_ema_count = 0
        for key in keys:
            if key.startswith("net."):
                net_count += 1
            if key.startswith("net_ema."):
                net_ema_count += 1

        state_dict_ema_only_fp32 = dict()  # ema only
        for key in state_dict_ema_reg:
            if key.startswith("net_ema."):
                key_new = key.replace("net_ema.", "net.")
                state_dict_ema_only_fp32[key_new] = state_dict_ema_reg[key]

        torch.save(state_dict_ema_only_fp32, torch_save_path_ema_only_fp32)
        print(f"Saved EMA weights from {torch_save_path_ema_reg} to {torch_save_path_ema_only_fp32}")

    # 3. save EMA only in bf16 precision
    if os.path.exists(torch_save_path_ema_only_bf16):
        print(f"{torch_save_path_ema_only_bf16} already exists, skipping.")
    else:
        state_dict_ema_only_bf16 = dict()  # ema only
        for key in state_dict_ema_only_fp32:
            if (
                isinstance(state_dict_ema_only_fp32[key], torch.Tensor)
                and state_dict_ema_only_fp32[key].dtype == torch.float32
            ):
                state_dict_ema_only_bf16[key] = state_dict_ema_only_fp32[key].bfloat16()
            else:
                state_dict_ema_only_bf16[key] = state_dict_ema_only_fp32[key]

        torch.save(state_dict_ema_only_bf16, torch_save_path_ema_only_bf16)
        print(f"fp32 -> bf16: {torch_save_path_ema_only_fp32} to {torch_save_path_ema_only_bf16}")
