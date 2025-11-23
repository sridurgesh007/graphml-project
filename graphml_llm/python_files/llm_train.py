import os
import sys
import gc
import json
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType

# Optional: silence tokenizer fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 1. CONFIGURATION
# ==========================================
TRAIN_JSON = "tox21_instruct_train.json"
VAL_JSON   = "tox21_instruct_val.json"
TRAIN_FEAT = "llm_features/train_features.pt"
VAL_FEAT   = "llm_features/val_features.pt"
OUTPUT_DIR = "checkpoints_labelaux_fixed"

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MAX_SEQ_LENGTH = 2048
GRAPH_DIM = 2304   # 256 (GINE) + 2048 (ECFP)
NUM_GRAPH_TOKENS = 8
UNIQUE_GRAPH_TOKENS = True

BATCH_SIZE = 12          # adjust if VRAM allows
GRAD_ACCUM = 8          # effective batch size = BATCH_SIZE * GRAD_ACCUM
EPOCHS = 4
LR = 2e-4
WARMUP_STEPS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

POS_WEIGHT_GRAPH = 2.0      # >1 to emphasize Toxic in graph-head loss
POS_WEIGHT_LLM   = 2.0      # >1 to emphasize Toxic in LLM label loss
LAMBDA_LLM_LABEL = 0.5      # weight on LLM label loss in total loss

# ==========================================
# 2. DATASET
# ==========================================

class RobustFusionDataset(Dataset):
    """
    For each molecule:
    - Builds a chat-style prompt with system + user + assistant (with graph tokens).
    - Encodes the full text with tokenizer.
    - Extracts:
        * input_ids, attention_mask
        * graph_features (GNN+ECFP)
        * tox_label (0/1)
        * label_pos: token index of the "Toxic"/"Non-Toxic" word in the assistant answer.
    """

    def __init__(self, json_path, feat_path, tokenizer):
        print(f"[DATA] Loading JSON: {json_path}")
        with open(json_path, "r") as f:
            self.data = json.load(f)

        print(f"[DATA] Loading features: {feat_path}")
        self.features = torch.load(feat_path)

        if len(self.data) != len(self.features):
            raise ValueError(f"JSON length {len(self.data)} != features {len(self.features)}")

        self.tokenizer = tokenizer

        # Graph tokens
        if UNIQUE_GRAPH_TOKENS:
            self.graph_tokens = [f"<GRAPH_{i}>" for i in range(NUM_GRAPH_TOKENS)]
            self.graph_token_str = " ".join(self.graph_tokens)
        else:
            self.graph_tokens = ["<GRAPH>"]
            self.graph_token_str = "<GRAPH> " * NUM_GRAPH_TOKENS

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # True label from JSON "output"
        out_text = item.get("output", "")
        is_toxic = 1 if "Prediction: Toxic" in out_text else 0

        # Forced prefix in assistant output
        forced_prefix = f"Graph Structure: {self.graph_token_str}\nAnalysis:\n"
        target_output = forced_prefix + out_text

        # System prompt
        system_prompt = (
            "You are an expert computational toxicologist using a Multi-Modal Graph-Language model.\n"
            "Your goal is to predict molecular toxicity by interpreting the provided Graph Structure Tokens.\n\n"
            "Process:\n"
            "1. READ: List the graph structure tokens explicitly.\n"
            "2. INTERPRET: Relate them to chemical features.\n"
            "3. PREDICT: Conclude with a final toxicity assessment.\n"
            "Use the format:\n"
            "<think> ... </think>\n\n"
            "Prediction: Toxic\n"
            "or\n"
            "Prediction: Non-Toxic\n"
        )

        # Inject graph tokens into the instruction
        instruction = item["instruction"]
        if "<GRAPH>" not in instruction:
            instruction = self.graph_token_str + "\n" + instruction
        else:
            instruction = instruction.replace("<GRAPH>", self.graph_token_str)

        # Construct chat-style text (DeepSeek/LLaMA style)
        full_text = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{target_output}<|eot_id|>"
        )

        # Tokenize full_text
        enc = self.tokenizer(
            full_text,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = enc.input_ids.squeeze(0)      # [T]
        attention_mask = enc.attention_mask.squeeze(0)

        # ---- Find token position of label word ("Toxic" / "Non-Toxic") ----
        label_pos = -1

        # We look for the substring "Prediction: Toxic" or "Prediction: Non-Toxic"
        # and then compute where the label word begins.
        toxic_phrase = "Prediction: Toxic"
        nontoxic_phrase = "Prediction: Non-Toxic"

        label_is_toxic = None
        phrase_to_find = None

        if toxic_phrase in full_text:
            phrase_to_find = toxic_phrase
            label_is_toxic = 1
        elif nontoxic_phrase in full_text:
            phrase_to_find = nontoxic_phrase
            label_is_toxic = 0

        if phrase_to_find is not None:
            # We use rfind in case of multiple occurrences; last one is the prediction line
            start_idx = full_text.rfind(phrase_to_find)
            if start_idx != -1:
                # Character index where label word starts (after "Prediction: ")
                label_char_idx = start_idx + len("Prediction: ")
                prefix_text = full_text[:label_char_idx]

                # Tokenize only the prefix and count tokens
                prefix_enc = self.tokenizer(
                    prefix_text,
                    max_length=MAX_SEQ_LENGTH,
                    truncation=True,
                    return_tensors="pt",
                )
                # The label token is the last token of the prefix
                label_pos = prefix_enc.input_ids.size(1) - 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "graph_features": self.features[idx],
            "labels": input_ids.clone(),                          # full LM loss
            "tox_label": torch.tensor(is_toxic, dtype=torch.float32),
            "label_pos": torch.tensor(label_pos, dtype=torch.long),
        }


# ==========================================
# 3. MODEL: LLM + PROJECTOR + GRAPH HEAD + LLM LABEL HEAD
# ==========================================

class RobustGraphLLM(nn.Module):
    def __init__(self):
        super().__init__()

        print("[MODEL] Loading base DeepSeek-R1 (4-bit)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.llm.gradient_checkpointing_enable()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add graph tokens
        special_tokens = []
        if UNIQUE_GRAPH_TOKENS:
            for i in range(NUM_GRAPH_TOKENS):
                special_tokens.append(f"<GRAPH_{i}>")
        else:
            special_tokens.append("<GRAPH>")

        self.tokenizer.add_tokens(special_tokens)
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # LoRA config
        print("[MODEL] Configuring LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            modules_to_save=["embed_tokens", "lm_head"],
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()

        # Graph token ids
        self.graph_token_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in special_tokens]

        # Projector (float32 for stability)
        llm_dim = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(GRAPH_DIM, 4096),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Linear(2048, llm_dim * NUM_GRAPH_TOKENS),
            nn.Tanh(),
        ).to(torch.float32)

        # Graph-head (aux classifier) – float32
        self.graph_cls_head = nn.Linear(llm_dim, 1).to(torch.float32)
        self.bce_graph = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([POS_WEIGHT_GRAPH], dtype=torch.float32, device=DEVICE)
        )

        # LLM label classifier head – predicts Toxic vs Non-Toxic from hidden state at label token
        self.llm_cls_head = nn.Linear(llm_dim, 1).to(torch.float32)
        self.bce_llm = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([POS_WEIGHT_LLM], dtype=torch.float32, device=DEVICE)
        )

    def to(self, device):
        self.projector.to(device)
        self.graph_cls_head.to(device)
        self.llm_cls_head.to(device)
        return super().to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        tox_label: Optional[torch.Tensor] = None,
        label_pos: Optional[torch.Tensor] = None,
    ):
        """
        input_ids: [B, T]
        attention_mask: [B, T]
        graph_features: [B, GRAPH_DIM]
        labels: [B, T]  (LM targets)
        tox_label: [B]  (0/1)
        label_pos: [B]  (token index of label word, or -1 if not found)
        """

        # 1) Base embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # 2) Project graph features → graph token embeddings
        graph_features = graph_features.to(dtype=torch.float32)
        graph_emb_flat = self.projector(graph_features)           # [B, N*H]
        H = inputs_embeds.size(-1)
        graph_emb_seq = graph_emb_flat.view(-1, NUM_GRAPH_TOKENS, H)  # [B, N, H]
        graph_emb_seq = graph_emb_seq.to(inputs_embeds.dtype).to(inputs_embeds.device)

        # 3) Inject graph embeddings at graph token positions
        B, T = input_ids.shape
        for i in range(B):
            graph_positions = []
            for token_id in self.graph_token_ids:
                positions = (input_ids[i] == token_id).nonzero(as_tuple=True)[0]
                graph_positions.extend(positions.tolist())
            graph_positions = sorted(graph_positions[:NUM_GRAPH_TOKENS])

            for j, pos in enumerate(graph_positions):
                inputs_embeds[i, pos] = graph_emb_seq[i, j]

        # 4) Forward through LLM (LM loss + hidden states)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
        )
        lm_loss = outputs.loss
        logits = outputs.logits                    # [B, T, V]
        hidden_states = outputs.hidden_states[-1]  # [B, T, H] (bfloat16)

        # 5) Graph-head auxiliary loss from graph token hidden states
        hidden_fp32 = hidden_states.to(torch.float32)
        pooled_states = []

        for i in range(B):
            positions = []
            for token_id in self.graph_token_ids:
                pos = (input_ids[i] == token_id).nonzero(as_tuple=True)[0]
                positions.extend(pos.tolist())
            positions = sorted(positions[:NUM_GRAPH_TOKENS])

            if len(positions) == 0:
                pooled = hidden_fp32[i].mean(dim=0)      # [H]
            else:
                pooled = hidden_fp32[i, positions].mean(dim=0)  # [H]
            pooled_states.append(pooled)

        pooled_states = torch.stack(pooled_states, dim=0)           # [B, H]
        graph_logits = self.graph_cls_head(pooled_states).squeeze(-1)  # [B]
        tox_label = tox_label.to(graph_logits.dtype).to(graph_logits.device)
        graph_loss = self.bce_graph(graph_logits, tox_label)

        # 6) LLM label loss – classification at label token position
        llm_label_loss = torch.tensor(0.0, device=logits.device)
        if label_pos is not None:
            label_pos = label_pos.to(torch.long).to(hidden_states.device)
            logits_llm_list = []
            targets_llm_list = []

            for i in range(B):
                pos = label_pos[i].item()
                if pos < 0 or pos >= T:
                    continue

                # hidden state at the label token
                h_i = hidden_fp32[i, pos, :]          # [H]
                logits_i = self.llm_cls_head(h_i)     # [1]
                logits_llm_list.append(logits_i.squeeze(-1))
                targets_llm_list.append(tox_label[i])

            if len(logits_llm_list) > 0:
                logits_llm = torch.stack(logits_llm_list, dim=0)       # [N]
                targets_llm = torch.stack(targets_llm_list, dim=0)     # [N]
                llm_label_loss = self.bce_llm(logits_llm, targets_llm)
            else:
                llm_label_loss = torch.tensor(0.0, device=logits.device)

        # 7) Total loss
        total_loss = lm_loss + graph_loss + LAMBDA_LLM_LABEL * llm_label_loss

        return {
            "loss": total_loss,
            "lm_loss": lm_loss.detach(),
            "graph_loss": graph_loss.detach(),
            "llm_label_loss": llm_label_loss.detach(),
        }


# ==========================================
# 4. UTILS
# ==========================================

def save_checkpoint(model, output_dir, epoch, is_best=False):
    suffix = "best" if is_best else f"epoch_{epoch+1}"
    path = os.path.join(output_dir, f"checkpoint_{suffix}")
    os.makedirs(path, exist_ok=True)

    print(f"[CKPT] Saving checkpoint -> {path}")
    torch.save(model.projector.state_dict(),       os.path.join(path, "projector.pt"))
    torch.save(model.graph_cls_head.state_dict(),  os.path.join(path, "graph_head.pt"))
    torch.save(model.llm_cls_head.state_dict(),    os.path.join(path, "llm_head.pt"))
    model.llm.save_pretrained(path)
    model.tokenizer.save_pretrained(path)


def sanity_check(model, loader, device):
    print("[CHECK] Running sanity check on one batch...")
    model.train()
    try:
        batch = next(iter(loader))
        input_ids = batch["input_ids"].to(device)
        mask      = batch["attention_mask"].to(device)
        feats     = batch["graph_features"].to(device)
        labels    = batch["labels"].to(device)
        tox       = batch["tox_label"].to(device)
        label_pos = batch["label_pos"].to(device)

        out = model(input_ids, mask, feats, labels=labels, tox_label=tox, label_pos=label_pos)
        loss = out["loss"]
        print(
            f"[CHECK] Sanity check passed. "
            f"Total: {loss.item():.4f}, "
            f"LM: {out['lm_loss'].item():.4f}, "
            f"Graph: {out['graph_loss'].item():.4f}, "
            f"LLM-label: {out['llm_label_loss'].item():.4f}"
        )
        loss.backward()
        model.zero_grad(set_to_none=True)
        return True
    except Exception as e:
        print(f"[CHECK] CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==========================================
# 5. MAIN TRAINING LOOP
# ==========================================

if __name__ == "__main__":
    print(f"[MAIN] Starting training on {DEVICE}")

    model = RobustGraphLLM()
    model.to(DEVICE)

    train_ds = RobustFusionDataset(TRAIN_JSON, TRAIN_FEAT, model.tokenizer)
    val_ds   = RobustFusionDataset(VAL_JSON,   VAL_FEAT,   model.tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    if not sanity_check(model, train_loader, DEVICE):
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = total_lm = total_graph = total_llm_label = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(DEVICE)
            mask      = batch["attention_mask"].to(DEVICE)
            feats     = batch["graph_features"].to(DEVICE)
            labels    = batch["labels"].to(DEVICE)
            tox       = batch["tox_label"].to(DEVICE)
            label_pos = batch["label_pos"].to(DEVICE)

            out = model(input_ids, mask, feats, labels=labels, tox_label=tox, label_pos=label_pos)
            loss = out["loss"] / GRAD_ACCUM
            loss.backward()

            total_loss      += loss.item() * GRAD_ACCUM
            total_lm        += out["lm_loss"].item()
            total_graph     += out["graph_loss"].item()
            total_llm_label += out["llm_label_loss"].item()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress.set_postfix({
                "loss":    f"{(loss.item() * GRAD_ACCUM):.3f}",
                "lm":      f"{out['lm_loss'].item():.2f}",
                "graph":   f"{out['graph_loss'].item():.2f}",
                "llm_lbl": f"{out['llm_label_loss'].item():.2f}",
            })

            if step % 50 == 0:
                torch.cuda.empty_cache()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(DEVICE)
                mask      = batch["attention_mask"].to(DEVICE)
                feats     = batch["graph_features"].to(DEVICE)
                labels    = batch["labels"].to(DEVICE)
                tox       = batch["tox_label"].to(DEVICE)
                label_pos = batch["label_pos"].to(DEVICE)

                out = model(input_ids, mask, feats, labels=labels, tox_label=tox, label_pos=label_pos)
                val_loss += out["loss"].item()

        avg_train = total_loss / len(train_loader)
        avg_val   = val_loss / len(val_loader)

        print(
            f"\n[RESULT] Epoch {epoch+1}/{EPOCHS} "
            f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}"
        )

        save_checkpoint(model, OUTPUT_DIR, epoch, is_best=(avg_val < best_val_loss))
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            print("[CKPT] New best model saved.")

        gc.collect()
        torch.cuda.empty_cache()

    print("[MAIN] Training finished.")