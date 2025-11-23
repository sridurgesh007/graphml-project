import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import numpy as np

############################################################
# CONFIG
############################################################

CHECKPOINT = "checkpoints_labelaux_fixed/checkpoint_best"   # <— change if needed
VAL_JSON   = "tox21_instruct_val.json"
VAL_FEAT   = "llm_features/val_features.pt"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MAX_NEW_TOKENS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRAPH_DIM = 2304
NUM_GRAPH_TOKENS = 8
UNIQUE_GRAPH_TOKENS = True

############################################################
# LOAD MODEL (MATCHES TRAINING ARCHITECTURE)
############################################################

class InferenceGraphLLM(nn.Module):
    def __init__(self, base_dir):
        super().__init__()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Graph special tokens
        if UNIQUE_GRAPH_TOKENS:
            self.graph_tokens = [f"<GRAPH_{i}>" for i in range(NUM_GRAPH_TOKENS)]
        else:
            self.graph_tokens = ["<GRAPH>"]

        self.tokenizer.add_tokens(self.graph_tokens)

        # Load 4-bit LLM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            base_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
            quantization_config=bnb_config,
        )

        self.llm.resize_token_embeddings(len(self.tokenizer))

        # Load projector + heads
        H = self.llm.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(GRAPH_DIM, 4096),
            nn.GELU(),
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Linear(2048, H * NUM_GRAPH_TOKENS),
            nn.Tanh(),
        ).to(torch.float32)

        self.graph_head = nn.Linear(H, 1).to(torch.float32)
        self.llm_head   = nn.Linear(H, 1).to(torch.float32)

        # Load weights
        self.projector.load_state_dict(torch.load(os.path.join(base_dir, "projector.pt"), map_location="cpu"))
        self.graph_head.load_state_dict(torch.load(os.path.join(base_dir, "graph_head.pt"), map_location="cpu"))
        self.llm_head.load_state_dict(torch.load(os.path.join(base_dir, "llm_head.pt"), map_location="cpu"))

        self.graph_token_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in self.graph_tokens]

    def inject_graph(self, input_ids, graph_feat):
        """Inject projected graph embeddings into graph token positions."""
        with torch.no_grad():
            B, T = input_ids.shape
            H = self.llm.config.hidden_size

            embeds = self.llm.get_input_embeddings()(input_ids)
            graph_feat = graph_feat.to(torch.float32)

            flat = self.projector(graph_feat)  # [B, N*H]
            seq = flat.view(B, NUM_GRAPH_TOKENS, H).to(embeds.dtype)

            for i in range(B):
                pos = []
                for tid in self.graph_token_ids:
                    loc = (input_ids[i] == tid).nonzero(as_tuple=True)[0]
                    pos.extend(loc.tolist())
                pos = sorted(pos[:NUM_GRAPH_TOKENS])
                for j, p in enumerate(pos):
                    embeds[i, p] = seq[i, j]

        return embeds

    def forward_graph_head(self, hidden_states, input_ids):
        """Graph-head classifier using pooled graph token hidden states."""
        B, T, H = hidden_states.size()
        hs_fp32 = hidden_states.to(torch.float32)
        pooled = []

        for i in range(B):
            pos = []
            for tid in self.graph_token_ids:
                loc = (input_ids[i] == tid).nonzero(as_tuple=True)[0]
                pos.extend(loc.tolist())
            pos = sorted(pos[:NUM_GRAPH_TOKENS])

            if len(pos) == 0:
                pooled.append(hs_fp32[i].mean(dim=0))
            else:
                pooled.append(hs_fp32[i, pos].mean(dim=0))

        pooled = torch.stack(pooled, dim=0)  # [B, H]
        logits = self.graph_head(pooled).squeeze(-1)  # [B]
        return logits

    def forward_llm_head(self, hidden_states, input_ids, raw_text):
        """LLM label head: extract hidden state at Prediction: TOKEN."""
        B, T, H = hidden_states.size()
        hs = hidden_states.to(torch.float32)

        pred_logits = []

        for i in range(B):
            text = raw_text[i]
            # locate Prediction: Toxic / Non-Toxic
            if "Prediction: Toxic" in text:
                phrase = "Prediction: Toxic"
            elif "Prediction: Non-Toxic" in text:
                phrase = "Prediction: Non-Toxic"
            else:
                pred_logits.append(torch.tensor(0.0, device=hs.device))
                continue

            idx = text.rfind(phrase)
            label_char_idx = idx + len("Prediction: ")

            prefix = text[:label_char_idx]
            enc = self.tokenizer(prefix, return_tensors="pt", truncation=True).to(input_ids.device)
            pos = enc.input_ids.size(1) - 1  # last token is the label word

            if pos >= T:
                pos = T - 1

            h = hs[i, pos]
            logit = self.llm_head(h)
            pred_logits.append(logit.squeeze(-1))

        return torch.stack(pred_logits, dim=0)


############################################################
# LOAD DATA
############################################################

print("[INFO] Loading data...")
data = json.load(open(VAL_JSON))
features = torch.load(VAL_FEAT)

assert len(data) == len(features)

############################################################
# LOAD MODEL
############################################################

model = InferenceGraphLLM(CHECKPOINT)
model.eval()
model.to(DEVICE)

############################################################
# RUN INFERENCE
############################################################

results = []

print("[INFO] Running inference...")
for idx in tqdm(range(len(data)), desc="Inference"):
    item = data[idx]
    graph_feat = features[idx].unsqueeze(0).to(DEVICE)

    # Build prompt
    instruction = item["instruction"]
    system = (
        "You are an expert computational toxicologist.\n"
        "Analyze the molecule based on structure and graph tokens.\n"
        "Provide reasoning inside <think>...</think>.\n"
        "End with:\nPrediction: Toxic\nor\nPrediction: Non-Toxic\n"
    )
    graph_tok_str = " ".join(model.graph_tokens)

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{graph_tok_str}\n{instruction}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
        f"Graph Structure: {graph_tok_str}\nAnalysis:\n"
    )

    encoded = model.tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = encoded.input_ids

    # Inject graph embeddings
    embeds = model.inject_graph(input_ids, graph_feat)

    # Generate
    with torch.no_grad():
        output = model.llm.generate(
            inputs_embeds=embeds,
            attention_mask=encoded.attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    decoded = model.tokenizer.decode(output[0], skip_special_tokens=True)
    raw_text = decoded

    # Extract prediction string
    if "Prediction: Toxic" in decoded:
        pred_llm = "Toxic"
    elif "Prediction: Non-Toxic" in decoded:
        pred_llm = "Non-Toxic"
    else:
        pred_llm = "Unknown"

    # Get hidden states for both heads
    with torch.no_grad():
        hs = model.llm(
            inputs_embeds=embeds,
            attention_mask=encoded.attention_mask,
            output_hidden_states=True,
            use_cache=False
        ).hidden_states[-1]

        graph_logit = model.forward_graph_head(hs, input_ids)
        llm_logit   = model.forward_llm_head(hs, input_ids, [decoded])

    p_graph = torch.sigmoid(graph_logit).item()
    p_llm   = torch.sigmoid(llm_logit).item()

    fused = 0.6 * p_graph + 0.4 * p_llm
    pred_fused = "Toxic" if fused >= 0.5 else "Non-Toxic"

    # Save entry
    results.append({
        "index": idx,
        "smiles": item.get("smiles", ""),
        "ground_truth": item["output"],
        "pred_llm": pred_llm,
        "pred_graph": "Toxic" if p_graph >= 0.5 else "Non-Toxic",
        "pred_fused": pred_fused,
        "llm_score": p_llm,
        "p_toxic_graph": p_graph,
        "fused_score": fused,
        "raw_output": decoded,
    })

############################################################
# METRICS
############################################################

def extract_gt(x):
    return 1 if "Toxic" in x else 0

def acc(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)
    return (pred == gt).mean()

def sensitivity(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)
    tp = ((pred==1)&(gt==1)).sum()
    fn = ((pred==0)&(gt==1)).sum()
    return tp / (tp+fn+1e-6)

def specificity(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)
    tn = ((pred==0)&(gt==0)).sum()
    fp = ((pred==1)&(gt==0)).sum()
    return tn / (tn+fp+1e-6)

gt = [extract_gt(r["ground_truth"]) for r in results]
pred_llm = [1 if r["pred_llm"]=="Toxic" else 0 for r in results]
pred_graph = [1 if r["pred_graph"]=="Toxic" else 0 for r in results]
pred_fused = [1 if r["pred_fused"]=="Toxic" else 0 for r in results]

print("\n=== METRICS (LLM Label-Head) ===")
print("Accuracy:", acc(pred_llm, gt))
print("Sensitivity:", sensitivity(pred_llm, gt))
print("Specificity:", specificity(pred_llm, gt))

print("\n=== METRICS (Graph-Head) ===")
print("Accuracy:", acc(pred_graph, gt))
print("Sensitivity:", sensitivity(pred_graph, gt))
print("Specificity:", specificity(pred_graph, gt))

print("\n=== METRICS (Fused) ===")
print("Accuracy:", acc(pred_fused, gt))
print("Sensitivity:", sensitivity(pred_fused, gt))
print("Specificity:", specificity(pred_fused, gt))

# Save JSON
out_path = "glassbox_new_val_results.json"
json.dump(results, open(out_path, "w"), indent=2)
print(f"[INFO] Saved detailed results to: {out_path}")