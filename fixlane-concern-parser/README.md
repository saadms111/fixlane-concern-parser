# Fixlane Concern Parser

Fine-tuning Qwen2.5-3B with QLoRA to parse technician repair notes into structured JSON — evaluated against the same base model (to show fine-tuning delta) and Claude Sonnet (the frontier baseline the assignment asks for).

---

## Repo layout

```
fixlane-concern-parser/
├── data/
│   ├── train.jsonl          # 500 training examples
│   ├── val.jsonl            # 100 validation examples
│   └── test.jsonl           # 100 test examples (held out)
├── outputs/                 # eval_results.json goes here after eval
├── adapter_weights/         # LoRA checkpoint goes here after training
├── explore_data.ipynb       # data inspection — run this first, no GPU needed
├── train.ipynb              # fine-tuning notebook — run in Colab on T4
├── eval.ipynb               # evaluation notebook — run after training
├── requirements.txt
└── README.md
```

---

## How to run

### 1. Look at the data first (no GPU needed)
Open `explore_data.ipynb` locally or in Colab. Run all cells. Takes a few seconds.

### 2. Fine-tune the model
Open `train.ipynb` in [Google Colab](https://colab.research.google.com).
Set the runtime to **GPU → T4** (Runtime → Change runtime type → T4 GPU).
Run all cells from top to bottom. Takes about 25–35 minutes.
At the end, a download cell saves `adapter_weights.zip` to your machine.

### 3. Evaluate
Open `eval.ipynb` in Colab (same T4 runtime).
Upload `test.jsonl` and `adapter_weights.zip` when prompted.
Paste your Anthropic API key in Step 3 (free credits at console.anthropic.com — 100 calls costs under $0.50).
Run all cells. Prints a three-column results table and downloads `eval_results.json`.
If you don't have an API key yet, skip Step 13 — the rest of the notebook still runs.

---

## Approach

### Base model: Qwen2.5-3B-Instruct

I picked Qwen2.5-3B for a few reasons. It's small enough to fit on a free T4 GPU with 4-bit quantization, and the `-Instruct` variant already knows how to follow structured output instructions and produce valid JSON — which matters a lot for this task since malformed JSON is a total failure even if the content is right. I briefly considered Llama-3.2-3B, but Qwen2.5 tends to be more consistent about sticking to a format constraint, which is important here.

### PEFT method: QLoRA

QLoRA loads the base model weights in 4-bit precision and trains small adapter layers (LoRA) on top. The base model weights stay completely frozen — only the adapter weights change. The adapter ends up being a few MB instead of a 6GB full checkpoint, and training on 500 examples finishes in half an hour instead of hours.

I wouldn't reach for full fine-tuning here. With 500 training examples, it would overfit quickly and the memory requirements on a T4 would be painful.

**LoRA settings:**

| Setting | Value | Why |
|---|---|---|
| rank (`r`) | 16 | Gives the adapter enough capacity for a multi-field structured output task without going overboard. r=8 might work, r=32 is overkill for 500 examples. |
| alpha | 32 | Standard practice is alpha = 2× rank. Controls how much influence the adapter has relative to the frozen base weights. |
| target modules | q_proj, k_proj, v_proj, o_proj | These are the attention layers — fine-tuning them teaches the model to focus on the right parts of a technician note. |
| dropout | 0.05 | Small amount of regularization to avoid overfitting on a small dataset. |

**Training settings:**

| Setting | Value | Why |
|---|---|---|
| Epochs | 3 | Enough to converge without overfitting. Val loss started climbing after epoch 3 in early tests. |
| Batch size | 4 | Safe for T4 with 4-bit quantization. |
| Gradient accumulation | 4 steps | Effective batch of 16, which is more stable than 4. |
| Learning rate | 2e-4 | Standard for LoRA. Higher than full fine-tuning because LoRA adapters start from zero. |
| Optimizer | paged_adamw_8bit | 8-bit paging from bitsandbytes saves optimizer memory on the T4. |

### Prompt format

Same instruction at training time and inference time:

```
You are a vehicle repair assistant. Parse the technician note below into structured fields.

Respond with a valid JSON object containing exactly these fields:
- vehicle_system: one of [engine, electrical, brakes, suspension, hvac, drivetrain, body, other]
- primary_symptom: a short noun phrase describing the symptom
- severity: one of [safety_critical, drivability, comfort, cosmetic]
- suggested_diagnostic: a short imperative phrase, or null

Only output the JSON object. No extra text.

Technician note: {input}

JSON:
```

During training the expected JSON follows immediately after `JSON:`. The labels are masked with `-100` on the prompt tokens so the model only learns to predict the answer, not the instruction.

---

## Evaluation design

I used different metrics for different fields because "correct" means different things across them.

- **vehicle_system** and **severity** are fixed label sets, so exact match accuracy is the right call.
- **primary_symptom** and **suggested_diagnostic** are free text. There are a bunch of valid ways to say the same thing, so exact match is too strict. I used ROUGE-L (longest common subsequence overlap) instead — it measures whether the right words are present without penalizing valid rephrasing.
- I also track **JSON validity** separately because a model that produces broken JSON is useless in production regardless of how good its content is.

| Field | Metric |
|---|---|
| vehicle_system | Exact match accuracy |
| severity | Exact match accuracy |
| primary_symptom | ROUGE-L |
| suggested_diagnostic | ROUGE-L |
| Overall | JSON validity % |

---

## Results

*(Fill this in after running eval.ipynb)*

| Metric | Fine-tuned Qwen2.5-3B | Base Qwen2.5-3B | Claude Sonnet |
|---|---|---|---|
| JSON valid (%) | 100.0 | 100.0 | 100.0 |
| vehicle_system accuracy (%) | 77.0 | 37.0 | 73.0 |
| severity accuracy (%) | 65.0 | 56.0 | 79.0 |
| primary_symptom ROUGE-L | 0.588 | 0.346 | 0.567 |
| suggested_diagnostic ROUGE-L | 0.334 | 0.160 | 0.362 |

---

## Data quality

A few things stood out when I looked at the training set:

**Near-duplicate inputs.** There are about 11 groups of inputs that are slightly paraphrased versions of each other — same scenario, different wording ("brakes squeak in the morning" vs "brks squeak in the morning" vs "Per customer, brakes squeak in the morning"). I kept them all. The paraphrase variation might actually help the model handle the messy real-world inputs technicians write.

**Ambiguous severity labels.** There are 8 rows where `vehicle_system = brakes` and `severity = comfort`. They're all variations of the "morning brake squeak that clears after a few stops" scenario. This is genuinely debatable — surface rust squeak is normal and not dangerous, so `comfort` isn't wrong. I kept them and noted them here because the model might pick up on this pattern and get confused on similar inputs.

**No structurally invalid rows.** Every row has valid field values within the defined label sets. No missing fields, no empty inputs.

---

## Failure mode analysis

The main failure patterns I'd expect based on the data:

**Comfort vs drivability confusion.** The boundary between these two is the fuzziest in the definitions. Cases like "ride feels slightly harsh" or "regen braking feels weaker than usual" are legitimately ambiguous, and the model will likely disagree with the labels on some of these.

**Severity on brake cases.** The training set has brake examples labeled as both `safety_critical` and `comfort`. The model has seen conflicting signal on this and may get it wrong when the input doesn't clearly indicate danger.

**Open-text drift on rare categories.** The `other` category has only 16 training examples and `engine` has 40. The model has less to go on for these, so the `primary_symptom` and `suggested_diagnostic` phrases may be generic or slightly off.

**Base model will likely produce off-format output on some inputs.** Without fine-tuning, Qwen2.5-3B understands the instruction but has no exposure to the specific label taxonomy. The fine-tune should win clearly on `vehicle_system` and `severity` accuracy. The base model may also produce prose instead of JSON on some inputs, which is an automatic zero.

**Claude Sonnet will likely win on the open-text fields.** A much larger model with far more world knowledge will write better `suggested_diagnostic` phrases, especially for unusual repair scenarios. The fine-tune may close the gap on `vehicle_system` and `severity` accuracy because it's been trained on the exact label definitions — Claude Sonnet has to infer them from the few-shot examples alone.

---

## What I'd change with more time

- **Severity sweep.** Break severity accuracy down by class (safety_critical vs drivability vs comfort vs cosmetic) to see exactly where the confusion is, rather than averaging it into one number.
- **LoRA rank sweep.** Try r=8, r=16, r=32 to check whether r=16 is actually justified.
- **More data augmentation.** Deliberately expand the duplicate groups with more paraphrase variations — different abbreviation styles, urgency markers, typo patterns.
- **Push weights to HuggingFace Hub** instead of downloading a zip every time.

---

## Adapter weights

Trained LoRA adapter: (https://huggingface.co/Saad09876/fixlane-concern-parser/tree/main)

To load:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

BASE_MODEL   = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "./adapter_weights"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
base      = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant_config, device_map="auto")
model     = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()
```
