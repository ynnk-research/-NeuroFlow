# > Yannick Schmitt. (2026). EMA-Gated Temporal Sequence Compression in Vision Transformers. Zenodo. https://doi.org/10.5281/zenodo.19337577
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Dataset: DAVIS + Custom Temporal Video
Loss: L2-Normalized Cosine Alignment + MSE

What is being trained:
  The encoder weights are already robust to partial token sequences
  thanks to TIPS pre-training. The bottleneck is the MAP head —
  a cross-attention pooling module that receives [N_active, D] as
  key-value pairs. At 96% skip, N_active ≈ 7-8 tokens. The MAP head
  was never trained to pool from such a sparse K/V set, causing the
  ~51% fidelity floor seen in benchmarks. This fine-tuning teaches
  the MAP head (and residually the encoder) to produce stable
  pooled embeddings from the sparse sequences NeuroFlow generates.
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import cv2
import glob
from transformers import AutoModel

def extract_custom_videos(video_dir="custom_traffic_videos", output_dir="combined_dataset"):
    """Extracts dropped .mp4 files into training-ready frame folders."""
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(video_dir):  os.makedirs(video_dir)

    mp4_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    if not mp4_files:
        print(f"[*] No custom videos found in '{video_dir}'. Moving on...")
        return

    print(f"[*] Found {len(mp4_files)} custom videos. Extracting frames...")
    for video_path in mp4_files:
        vid_name    = os.path.splitext(os.path.basename(video_path))[0]
        vid_out_dir = os.path.join(output_dir, vid_name)
        if os.path.exists(vid_out_dir): continue
        os.makedirs(vid_out_dir)

        cap, count = cv2.VideoCapture(video_path), 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (854, 480))
            cv2.imwrite(os.path.join(vid_out_dir, f"{count:05d}.jpg"), frame)
            count += 1
        cap.release()
    print("[*] Custom video extraction complete.")

class NeuroFlowSiglipStudent(nn.Module):
    def __init__(self, siglip_vision_model, threshold=0.05, ema_decay=0.01):
        super().__init__()
        self.model       = siglip_vision_model
        self.threshold   = threshold
        self.ema_decay   = ema_decay
        self.expectation = None

    def reset_retina(self):
        self.expectation = None

    def forward(self, pixel_values):
        hidden_states = self.model.embeddings(pixel_values)  # [1, N, D]
        N = hidden_states.shape[1]

        if self.expectation is None:
            self.expectation = hidden_states.detach().clone()
            active_indices   = torch.arange(N, device=pixel_values.device)
        else:
            sim      = F.cosine_similarity(hidden_states, self.expectation, dim=-1)
            surprise = 1.0 - sim

            active_indices = (surprise > self.threshold)[0].nonzero(as_tuple=True)[0]

            # MAP head requires at least 4 tokens for stable cross-attention
            MIN_TOKENS = 4
            if len(active_indices) < MIN_TOKENS:
                _, active_indices = torch.topk(surprise[0], MIN_TOKENS)

            active_indices = torch.sort(active_indices)[0]

            self.expectation = (
                (1.0 - self.ema_decay) * self.expectation +
                self.ema_decay * hidden_states.detach()
            )

        hidden_compressed = hidden_states[:, active_indices, :]

        encoder_outputs = self.model.encoder(inputs_embeds=hidden_compressed)
        sequence_output = self.model.post_layernorm(encoder_outputs[0])
        pooled_output   = self.model.head(sequence_output)

        sparsity = 1.0 - (len(active_indices) / float(N))
        return pooled_output, sparsity


class UnifiedVideoDataset(Dataset):
    def __init__(self, davis_root, custom_root="combined_dataset",
                 seq_length=24, frame_stride=3, transform=None):
        self.clips     = []
        self.transform = transform
        video_folders  = []

        if os.path.exists(custom_root):
            video_folders.extend(
                [os.path.join(custom_root, f) for f in os.listdir(custom_root)])
        if davis_root and os.path.exists(
                os.path.join(davis_root, "JPEGImages/Full-Resolution")):
            d_root = os.path.join(davis_root, "JPEGImages/Full-Resolution")
            video_folders.extend(
                [os.path.join(d_root, f) for f in os.listdir(d_root)])

        for folder in video_folders:
            frames = sorted(
                [os.path.join(folder, f)
                 for f in os.listdir(folder) if f.endswith(".jpg")])
            frames = frames[::frame_stride]
            for i in range(0, len(frames) - seq_length + 1, seq_length):
                self.clips.append(frames[i:i + seq_length])

        MAX_CLIPS = 800
        if len(self.clips) > MAX_CLIPS:
            random.shuffle(self.clips)
            self.clips = self.clips[:MAX_CLIPS]

        print(f"[*] Dataset initialized: {len(self.clips)} clips. (Stride={frame_stride})")

    def __len__(self): return len(self.clips)

    def __getitem__(self, idx):
        return torch.stack(
            [self.transform(Image.open(p).convert("RGB")) for p in self.clips[idx]])


#def siglip_distillation_loss(student_embed, teacher_embed):
#    """Aligns student to teacher strictly by angle on the L2 hypersphere."""
#    s_norm = F.normalize(student_embed, p=2, dim=-1)
#    t_norm = F.normalize(teacher_embed, p=2, dim=-1)

#    target      = torch.ones(student_embed.size(0), device=student_embed.device)
#    cosine_loss = nn.CosineEmbeddingLoss()(s_norm, t_norm, target)
#    mse_loss    = F.mse_loss(s_norm, t_norm)

#    return cosine_loss + (10.0 * mse_loss)

def siglip_distillation_loss(student_embed, teacher_embed):
    """
    Train on RAW embeddings. Do NOT normalize.
    This forces the student to learn the exact magnitude the teacher produces.
    """
    # 1. MSE on raw embeddings (Preserves Angle AND Magnitude)
    mse_loss = F.mse_loss(student_embed, teacher_embed)

    # 2. Add a softer constraint on direction if needed
    cos_sim = F.cosine_similarity(student_embed, teacher_embed, dim=-1)
    dir_loss = (1.0 - cos_sim).mean()

    return mse_loss + (0.5 * dir_loss)


def train_siglip2_neuroflow(davis_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing SigLIP 2 Sparse Distillation on {device.type.upper()}...")

    # Model name: SigLIP 2 FixRes
    BASE_MODEL = "google/siglip2-base-patch16-224"
    print(f"[*] Loading {BASE_MODEL}...")

    # Teacher = SigLIP 2 dense (frozen)
    teacher = AutoModel.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32
    ).vision_model.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student = SigLIP 2 (separate load, independent weights)
    student_base = AutoModel.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32
    ).vision_model
    student = NeuroFlowSiglipStudent(student_base).to(device).train()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset    = UnifiedVideoDataset(davis_root=davis_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=2, pin_memory=True)

    optimizer = optim.AdamW(student.parameters(), lr=3e-5, weight_decay=1e-4)

    scaler = torch.amp.GradScaler("cuda")

    EPOCHS = 25

    for epoch in range(EPOCHS):
        total_loss, total_sparsity = 0.0, 0.0

        for step, clip in enumerate(dataloader):
            clip       = clip.squeeze(0).to(device)   # [seq_len, 3, 224, 224]
            seq_length = clip.shape[0]

            # SigLIP 2 zero-shot already reaches 95-97% skip, meaning the
            # MAP head will operate at extreme sparsity in deployment.
            # Training must cover that regime, not just 0-35% thresholds.
            # At threshold=0.55 on a traffic video, expect 97-99% skip —
            # the MAP head will see 2-7 active tokens and must learn to
            # pool a stable embedding from that compressed K/V set.
            student.threshold = random.uniform(0.0001, 0.35)
            student.ema_decay = random.uniform(0.001, 0.1)

            student.reset_retina()
            optimizer.zero_grad()

            clip_loss, clip_sparsity = 0.0, 0.0

            for t in range(seq_length):
                frame = clip[t:t + 1]

                with torch.no_grad():
                    t_embed = teacher(frame).pooler_output

                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    s_embed, sparsity = student(frame)
                    loss = siglip_distillation_loss(s_embed, t_embed)

                clip_loss    += loss
                clip_sparsity += sparsity

            clip_loss = clip_loss / seq_length
            scaler.scale(clip_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss    += clip_loss.item()
            total_sparsity += (clip_sparsity / seq_length)

            if step % 10 == 0:
                avg_skip = (clip_sparsity / seq_length) * 100
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{step}/{len(dataloader)}] "
                      f"Loss: {clip_loss.item():.4f} | Avg Skip: {avg_skip:.1f}%")

        avg_loss = total_loss / len(dataloader)
        print(f"==> Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")

    save_path = "siglip2_finetuned.pth"
    torch.save(student.model.state_dict(), save_path)
    print(f"\n[*] Training Complete. Saved to '{save_path}'")
    print(f"[*] Load in benchmark with: AutoModel.from_pretrained('{BASE_MODEL}').vision_model")
    print(f"[*]   then: model.load_state_dict(torch.load('{save_path}'), strict=False)")


if __name__ == "__main__":
    extract_custom_videos(video_dir="custom_traffic_videos", output_dir="combined_dataset")

    DAVIS_PATH = "/content/DAVIS"
    if os.path.exists(DAVIS_PATH) or os.path.exists("combined_dataset"):
        train_siglip2_neuroflow(
            DAVIS_PATH if os.path.exists(DAVIS_PATH) else None)
    else:
        print("[!] Dataset missing. Place DAVIS at:", DAVIS_PATH)