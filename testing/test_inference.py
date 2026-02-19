# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# End-to-end example script for the inference pipeline:
# This script loads a dataset, runs inference, and computes the minADE.
# It can be used to test the inference pipeline.

import torch
import numpy as np
import mediapy as mp

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1, AlpamayoR1Config
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper


# Example clip ID


clip_id = "3aba15fb-61b2-497a-b38e-8e3bdcc68b75"
print(f"Loading dataset for clip_id: {clip_id}...")
data = load_physical_aiavdataset(clip_id, num_frames=32)
#print(data)
print("Dataset loaded.")

video_tensor = data["image_frames"]
if video_tensor.device.type != 'cpu':
    video_tensor = video_tensor.cpu()

video_permuted = video_tensor.permute(1, 3, 0, 4, 2)
T, H, N_cams, W, C = video_permuted.shape
video_stitched = video_permuted.reshape(T, H, N_cams * W, C)

video_np = video_stitched.numpy()
if video_np.max() <= 1.0:
    video_np = (video_np * 255).astype('uint8')
else:
    video_np = video_np.astype('uint8')

video_filename = "alpamayo_inference.mp4"
mp.write_video(video_filename, video_np, fps=4)
print(f"Number of frames in video_np: {video_np.shape[0]}")
print("video saved:")

messages = helper.create_message(data["image_frames"].flatten(0, 1))

config = AlpamayoR1Config.from_pretrained("nvidia/Alpamayo-R1-10B")
config.attn_implementation = "sdpa"
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16, config=config).to("cuda")
processor = helper.get_processor(model.tokenizer)

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)
model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}

model_inputs = helper.to_device(model_inputs, "cuda")

torch.cuda.manual_seed_all(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,  # Feel free to raise this for more output trajectories and CoC traces.
        max_generation_length=256,
        return_extra=True,
    )

# the size is [batch_size, num_traj_sets, num_traj_samples]
print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
min_ade = diff.min()

import matplotlib.pyplot as plt
def rotate_90cc(xy):
    # Rotate (x, y) by 90 deg CCW -> (y, -x)
    return np.stack([-xy[1], xy[0]], axis=0)


for i in range(pred_xyz.shape[2]):
    pred_xy = pred_xyz.cpu()[0, 0, i, :, :2].T.numpy()
    pred_xy_rot = rotate_90cc(pred_xy)
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    gt_xy_rot = rotate_90cc(gt_xy)
    plt.plot(*pred_xy_rot, "o-", label=f"Predicted Trajectory #{i + 1}")
plt.ylabel("y coordinate (meters)")
plt.xlabel("x coordinate (meters)")
plt.plot(*gt_xy_rot, "r-", label="Ground Truth Trajectory")
plt.legend(loc="best")
plt.axis("equal")
plt.show()

print(gt_xy)
print("minADE:", min_ade, "meters")
# print(
#     "Note: VLA-reasoning models produce nondeterministic outputs due to trajectory sampling, "
#     "hardware differences, etc. With num_traj_samples=1 (set for GPU memory compatibility), "
#     "variance in minADE is expected. For visual sanity checks, see notebooks/inference.ipynb"
# )
