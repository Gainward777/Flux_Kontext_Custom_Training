# LoRA training for **Flux Kontext** with region token & depth conditioning via **IP-Adapter**

This repository/Colab snippet implements a **minimal-yet-complete** LoRA training loop for
`FluxKontextPipeline` with:
- **Mask-guided editing** via a small *region token* (a learnable projection of a binary mask),
- **Depth conditioning** fed through **IP-Adapter**,
- A loss that simultaneously encourages **target matching inside** the mask and **context preservation outside** the mask,
- Simple **L2 stabilizers** for the region projector and LoRA weights.

> The implementation is intentionally compact and avoids excessive checks/casting.
> It follows a flow-matching training style compatible with `FlowMatchEulerDiscreteScheduler` used for SD3/Flux family models.

---

## Table of Contents

- [Requirements](#requirements)
- [Data Layout](#data-layout)
- [How It Works](#how-it-works)
- [Functions and Classes (API)](#functions-and-classes-api)
  - [`normalize_01_to_m11`](#normalize_01_to_m11)
  - [`to_latents`](#to_latents)
  - [`pack_like_pipeline`](#pack_like_pipeline)
  - [`img_ids_grid`](#img_ids_grid)
  - [`KontextPairs` (Dataset)](#kontextpairs-dataset)
  - [`MaskRegionToken`](#maskregiontoken)
  - [`collect_lora_named_params`](#collect_lora_named_params)
  - [`fm_masked_dual_loss_with_l2`](#fm_masked_dual_loss_with_l2)
- [Training Loop & Shapes](#training-loop--shapes)
- [Notes on Flux / PEFT / IP-Adapter](#notes-on-flux--peft--ip-adapter)
- [Common Pitfalls](#common-pitfalls)
- [Licenses & Model Access](#licenses--model-access)

---

## Requirements

- Python 3.10+
- PyTorch (CUDA recommended)
- Libraries: `diffusers`, `transformers`, `peft`, `safetensors`, `torchvision`, `Pillow`, `tqdm`, `numpy`, `opencv-python` (optional).

Install example (in Colab):
```bash
pip install -qU diffusers transformers peft safetensors torch torchvision pillow tqdm numpy opencv-python
```

---

## Data Layout

```
DATA_ROOT/
  context/  *.png|jpg  # source (context) images
  target/   *.png|jpg  # target images
  masks/    *.png|jpg  # binary masks (white = editable, black = preserve)
  depth/    *.png|jpg  # depth maps (any bit depth; converted to RGB PIL for IP-Adapter)
  prompts.txt          # one line per pair in the order of the sorted file lists
```

---

## How It Works

1) **Flux Kontext + LoRA**  
   Load `FluxKontextPipeline`, freeze VAE/T5/CLIP, and attach a PEFT LoRA adapter to the
   Flux transformer (`add_adapter(...); set_adapter(...)`).

2) **Region token**  
   The binary mask on the **latent token grid** (`H'×W'`) is projected by a tiny linear
   module into a single *region token* (shape `[B,1,Cin]`) concatenated to the visual
   sequence alongside target/context tokens.

3) **Depth via IP-Adapter**  
   Provide a **PIL RGB** depth image to `prepare_ip_adapter_image_embeds(...)`.
   The resulting embeddings are fed to the transformer via
   `joint_attention_kwargs={"ip_adapter_image_embeds": ...}` — the same mechanism the
   pipeline uses internally.

4) **Flow-matching training objective**  
   At a random `timestep` (sigma from the scheduler), we train on the **velocity**:
   - **Inside the mask**: MSE between predicted velocity and the FM target `v_tgt = ε - x_tgt`.
   - **Outside the mask**: identity penalty on the reconstructed latent `x_pred = ε - v_pred`
     to preserve the context.
   Optional **L2** regularizers stabilize the region projector and LoRA parameters.

---

## Functions and Classes (API)

### `normalize_01_to_m11`
```python
def normalize_01_to_m11(img: torch.Tensor) -> torch.Tensor
```
**Purpose**: Map intensities from `[0,1]` to `[-1,1]`.

**Args**
- `img` (`torch.Tensor`, shape `[B,C,H,W]` or `[C,H,W]`, `float` in `[0,1]`)

**Returns**
- `torch.Tensor`, same shape/type, values in `[-1,1]`

---

### `to_latents`
```python
def to_latents(img_bchw: torch.Tensor) -> torch.Tensor
```
**Purpose**: Encode images with the pipeline VAE and scale latents by `vae.config.scaling_factor`.

**Args**
- `img_bchw` (`torch.Tensor`, shape `[B,C,H,W]`, values in `[-1,1]`)

**Returns**
- `torch.Tensor` (latents), shape `[B, C_lat, H_lat, W_lat]`  
  (dimensions depend on the VAE in the loaded Flux pipeline).

---

### `pack_like_pipeline`
```python
def pack_like_pipeline(lat: torch.Tensor) -> tuple[torch.Tensor, tuple[int,int]]
```
**Purpose**: Pack latents with a **2×2** pattern into token sequences.

**Args**
- `lat` (`torch.Tensor`, shape `[B,C,H,W]`, `H` and `W` must be divisible by 2)

**Returns**
- `tokens` (`torch.Tensor`, shape `[B, N, C*4]`), `N = (H/2)*(W/2)`  
- `(H',W')` (`tuple[int,int]`) — token grid size, `H' = H/2`, `W' = W/2`

> This is a compact helper to match the visual sequence shape that the Flux transformer expects.
> It is not a drop-in replacement for all internal packing variants across pipelines but is
> dimensionally compatible with this training loop.

---

### `img_ids_grid`
```python
def img_ids_grid(batch: int, h_tok: int, w_tok: int, mark: float, dtype: torch.dtype) -> torch.Tensor
```
**Purpose**: Produce a grid of image IDs `(type, row, col)` for each visual token.

**Args**
- `batch` (`int`): batch size `B`
- `h_tok`, `w_tok` (`int`): token grid size
- `mark` (`float`): channel 0 “type” ID (`0.0`=target, `1.0`=context, `2.0`=region)
- `dtype` (`torch.dtype`)

**Returns**
- `torch.Tensor`, shape `[B, N, 3]`, `N = h_tok*w_tok`

> Recent Flux versions **deprecate 3D** `[B,N,3]` `img_ids/txt_ids` in favor of **2D** `[N,3]`.
> If you see deprecation warnings, drop the batch dimension and pass `[N,3]`.

---

### `KontextPairs` (Dataset)
```python
class KontextPairs(Dataset):
    def __init__(self, root: str, resolution=1024, ip_resolution=224)
    def __len__(self) -> int
    def __getitem__(self, i: int) -> dict
```
**Purpose**: Load aligned `context/target/mask/depth/prompt` tuples.

**Init Args**
- `root` (`str`): dataset root
- `resolution` (`int`): output resolution for images/masks (BICUBIC for images, NEAREST for masks)
- `ip_resolution` (`int`): reserved (not used in the final code), can be removed safely

**`__getitem__` Returns**
- `context` (`torch.Tensor`, `[3,H,W]`, `float` in `[0,1]`): source frame
- `target`  (`torch.Tensor`, `[3,H,W]`, `float` in `[0,1]`): target frame
- `mask`    (`torch.Tensor`, `[1,H,W]`, `float` in `{0,1}`): edit region
- `depth`   (`torch.Tensor`, `[3,H,W]`, dtype from `pil_to_tensor`): later converted to **PIL RGB**
  via `to_pil_image(...)` in the training loop and passed to IP-Adapter
- `prompt`  (`str`)

---

### `MaskRegionToken`
```python
class MaskRegionToken(nn.Module):
    def __init__(self, h_tok: int, w_tok: int, c_in: int)
    def forward(self, m_small: torch.Tensor) -> torch.Tensor
```
**Purpose**: Map a latent-grid mask to a single *region token* compatible with the visual sequence.

**Init Args**
- `h_tok`, `w_tok` (`int`): latent token grid size
- `c_in` (`int`): transformer input channel size (e.g., `transformer.config.in_channels`)

**Forward**
- **Input** `m_small` (`torch.Tensor`, `[B,1,H',W']`, `float` in `[0,1]`)
- **Output** `region` (`torch.Tensor`, `[B,1,c_in]`)

---

### `collect_lora_named_params`
```python
def collect_lora_named_params(peft_wrapped_module) -> list[tuple[str, torch.nn.Parameter]]
```
**Purpose**: Collect `(name, parameter)` pairs of trainable LoRA parameters from a PEFT-wrapped module.

**Args**
- `peft_wrapped_module`: a model with an active PEFT adapter (e.g., the Flux transformer after `add_adapter`)

**Returns**
- `list[tuple[str, torch.nn.Parameter]]`: trainable parameters (e.g., names containing `lora_`)

---

### `fm_masked_dual_loss_with_l2`
```python
def fm_masked_dual_loss_with_l2(
    pred_v: torch.Tensor,        # [B, N, C]
    tgt_tok: torch.Tensor,       # [B, N, C]
    ctx_tok: torch.Tensor,       # [B, N, C]
    noise: torch.Tensor,         # [B, N, C]
    mask_small: torch.Tensor,    # [B, 1, H', W']
    h_tok: int, w_tok: int,
    w_in: float = 1.0, w_out: float = 1.0,
    l2_region_lambda: float = 1e-4, l2_lora_lambda: float = 1e-4,
    region_token_module: nn.Module | None = None,
    lora_named_params: list[tuple[str, torch.nn.Parameter]] | None = None,
    eps: float = 1e-6,
) -> torch.Tensor
```
**Purpose**: Composite loss:
- **Inside mask**: flow-matching velocity target `v_tgt = ε − x_tgt` → MSE(`pred_v`, `v_tgt`)
- **Outside mask**: identity on reconstructed latent `x_pred = ε − pred_v` → MSE(`x_pred`, `ctx_tok`)
- **L2 stabilizers**: on region projector and LoRA parameters

**Args**
- `pred_v` (`torch.Tensor`, `[B,N,C]`): predicted velocity on target tokens
- `tgt_tok` (`torch.Tensor`, `[B,N,C]`): packed target latents
- `ctx_tok` (`torch.Tensor`, `[B,N,C]`): packed context latents
- `noise` (`torch.Tensor`, `[B,N,C]`): sampled ε at the current timestep
- `mask_small` (`torch.Tensor`, `[B,1,H',W']`): binary/soft mask on the latent grid
- `h_tok`, `w_tok` (`int`): token grid size
- `w_in`, `w_out` (`float`): weights for in-mask vs. out-of-mask terms
- `l2_region_lambda`, `l2_lora_lambda` (`float`): L2 coefficients
- `region_token_module` (`nn.Module|None`): region projector to regularize
- `lora_named_params` (`list[(str,Parameter)]|None`): list of trainable LoRA params to regularize
- `eps` (`float`): numerical stabilizer

**Returns**
- `torch.Tensor` scalar loss (shape `[]`)

---

## Training Loop & Shapes

- **Text encodings** (from `pipe.encode_prompt(prompt=..., prompt_2=...)`):
  - `prompt_embeds`: `[B, T_txt, D_t5]`
  - `pooled_embeds`: `[B, D_clip]`
  - `txt_ids`: `[T_txt, 3]` or `[B, T_txt, 3]` (prefer **2D** as newer Flux expects)  
- **Latent tokens**
  - `tgt_tok`, `ctx_tok`: `[B, N, Cin]` after packing (`N = H'*W'`)
  - `region token`: `[B, 1, Cin]`
  - `hidden_states`: concatenation `[B, 2N+1, Cin]`
- **IDs**
  - `img_ids`: `[B, 2N+1, 3]` (or `[2N+1,3]` preferred by newer Flux)
- **Timestep**
  - `timestep`: `[B]` (1D), sampled from `scheduler.sigmas` and matched in `device/dtype`

> IP-Adapter embeddings are prepared from **PIL RGB** depth and passed to the transformer via
> `joint_attention_kwargs={"ip_adapter_image_embeds": ip_embeds}`.

---

## Notes on Flux / PEFT / IP-Adapter

- **Two prompts** are typical in SDXL-like stacks and Flux Kontext: CLIP pooled (`prompt`) and T5 sequence (`prompt_2`).
- **PEFT**: the Flux transformer class in diffusers is PEFT-enabled; LoRA adapters are added/selected using
  `add_adapter(...)` / `set_adapter(...)`.
- **IP-Adapter**: you may pass either `ip_adapter_image` (PIL/np/torch) or precomputed `ip_adapter_image_embeds`.
  In this training loop we compute embeddings once per step and inject them via `joint_attention_kwargs`.

---

## Common Pitfalls

- **`timestep` must be 1D**: shape `[B]`, on the same device and dtype as the visual tokens.
- **Avoid 3D IDs**: newer Flux transformers prefer `[N,3]` (2D) for `img_ids` and `txt_ids`.
- **`guidance` as float**: pass a tensor or omit it; some transformer variants expect tensor-like for casting.
- **Device/dtype mismatches**: keep `scheduler.sigmas`, indices, and tokens on the same device/dtype.

---

## Licenses & Model Access

- `black-forest-labs/FLUX.1-Kontext-dev` may require accepting license terms on Hugging Face before downloading.
- Check the licenses for the IP-Adapter weights you use (e.g., XLabs/InstantX checkpoints).

---

### Acknowledgements

- Hugging Face `diffusers` for Flux/PEFT/IP-Adapter integrations
- The PEFT library for adapter support
- The SD3/Flux flow-matching schedulers
