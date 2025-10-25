# min_lr & Cosine Scheduler - FAQ

Status: Archived — Superseded by docs/REFERENCE.md (LR scheduler with min_lr)

## Your Questions Answered

### Q1: Does the cosine scheduler perfectly drop the LRs to 0 when max step is achieved?

**Answer: Yes, BY DEFAULT.**

**Without `min_lr` (default):**
- Cosine scheduler decays from initial LR → **exactly 0** at step = T_max
- Formula: `LR(t) = (initial_lr - eta_min) * (1 + cos(π*t/T_max)) / 2 + eta_min`
- With `eta_min=0`: LR drops to exactly **0.0**

**Example:**
```
initial_lr = 5e-4, T_max = 1000 steps, eta_min = 0 (default)

Step 0:    LR = 5.00e-04  ← full power
Step 500:  LR = 2.50e-04  ← halfway decay
Step 900:  LR = 1.22e-05  ← almost there
Step 1000: LR = 0.00e+00  ← complete stop ❌
```

---

### Q2: Where should we reach the `min_lr`? At the end of T_max?

**Answer: The `min_lr` acts as a floor. It doesn't "reach" min_lr at a specific step—it STOPS DECAYING at min_lr.**

**With `min_lr` set:**
- Scheduler decays from initial LR towards min_lr
- Reaches the floor (min_lr) around 99% through training
- **Plateaus at min_lr** for the final 1-2% of steps

**Example:**
```
initial_lr = 5e-4, T_max = 1000 steps, eta_min = 1e-6

Step 0:    LR = 5.00e-04  ← start
Step 500:  LR = 2.51e-04  ← halfway
Step 900:  LR = 1.32e-05  ← approaching floor
Step 950:  LR = 4.07e-06  ← near floor
Step 990:  LR = 1.12e-06  ← ≈ floor
Step 1000: LR = 1.00e-06  ← plateaued at min_lr ✅
```

**Key Point:** The min_lr is a **ceiling on decay**—the scheduler won't go below it.

---

### Q3: What's the point of `min_lr` if we don't have rest steps to be continued on?

**Great question!** This is crucial for understanding when min_lr matters.

#### Scenario A: One-shot Training (ends exactly at T_max, no resume)

```
Initial LR = 5e-4, T_max = 1000, train for exactly 1000 steps, then STOP
```

**Does min_lr matter?**
- ❌ **No, not really.**
- Whether LR goes to 0 or 1e-6 after training ends doesn't matter (training is over)
- The model won't do any more updates anyway
- You could use either `eta_min=0` or `eta_min=1e-6`

#### Scenario B: Training Resume (crash recovery or intentional multi-stage)

```
Initial LR = 5e-4, T_max = 1000, train for 1000 steps, then CRASH
→ Resume from checkpoint at step 1000
```

**Does min_lr matter?**
- ✅ **YES, critical!**

**Without min_lr (eta_min=0):**
```
Step 1000 (end):     LR = 0.0      ← training stops completely
CRASH → Resume from checkpoint
Step 1001 (resume):  LR = 0.0      ← optimizer STUCK, won't update! ❌
```

**With min_lr (eta_min=1e-6):**
```
Step 1000 (end):     LR = 1e-6     ← still has learning rate
CRASH → Resume from checkpoint
Step 1001 (resume):  LR = 1e-6     ← training continues! ✅
```

#### Scenario C: Your Training Setup (Stage 1 → 2 → 3)

```
Stage 1: Train aligner for 1000 steps (LR decays to min_lr)
         ↓
         Save checkpoint
         ↓
Stage 2: Load checkpoint, fine-tune LLM + aligner for 1000 more steps
         Starts at min_lr from stage 1, then decays again
         ↓
         Save checkpoint
         ↓
Stage 3: Load checkpoint, fine-tune vision + LLM for 1000 more steps
         Starts at min_lr from stage 2
```

**Without min_lr in stage 1:**
- Stage 1 ends with LR = 0
- Stage 2 starts with LR = 0
- **No learning happens in stage 2!** ❌

**With min_lr in stage 1:**
- Stage 1 ends with LR = 1e-6
- Stage 2 starts with LR = 1e-6 (you can set new LR for stage 2)
- Learning continues properly ✅

---

## Summary Table

| Setup | min_lr? | Why |
|-------|--------|-----|
| **One-shot training (→ done)** | Optional | Doesn't matter after training ends |
| **Resume from crash** | **Required** | Prevents LR=0 on resume |
| **Multi-stage training** | **Required** | Ensures smooth transition between stages |
| **You want smooth loss curves** | Recommended | Avoids sudden learning stop |

---

## Practical Recommendation

### For Your Qwen3-VL Setup

Since you're doing **multi-stage training** (stage 1→2→3), **always use min_lr**:

```yaml
# Stage 1
training:
  learning_rate: 1.0e-4
  lr_scheduler_type: cosine
  lr_scheduler_kwargs:
    min_lr: 1.0e-6         # ← Add this!

# Stage 2
training:
  learning_rate: 5.0e-4
  aligner_lr: 1.0e-4
  lr_scheduler_type: cosine
  lr_scheduler_kwargs:
    min_lr: 5.0e-6         # ← Add this!

# Stage 3
training:
  learning_rate: 5.0e-4
  vit_lr: 2.0e-4
  lr_scheduler_type: cosine
  lr_scheduler_kwargs:
    min_lr: 1.0e-6         # ← Add this!
```

**Benefits:**
1. ✅ If training crashes, you can resume smoothly
2. ✅ If you add stage 4 later, LR isn't stuck at 0
3. ✅ Loss curves remain smooth throughout training
4. ✅ No harm if you never resume (LR floor doesn't hurt)

---

## Technical Details (Optional)

### The Cosine Annealing Formula

```
LR(t) = eta_min + (initial_lr - eta_min) × (1 + cos(π × t / T_max)) / 2

Where:
  eta_min      = floor (min_lr)
  initial_lr   = starting learning rate
  t            = current step (0 to T_max)
  T_max        = total training steps
```

### Key Insight

The cosine term `cos(π × t / T_max)` goes from:
- `cos(0) = 1` at step 0 → LR = initial_lr
- `cos(π/2) ≈ 0` at step = T_max/2 → LR ≈ (initial_lr + eta_min) / 2
- `cos(π) = -1` at step = T_max → LR = eta_min

So with `eta_min=0`, you get LR → 0. With `eta_min=1e-6`, you get LR → 1e-6.

---

## When Would You NOT Use min_lr?

Only if:
1. You have exactly ONE training run, no crashes, no multi-stage
2. You want aggressive decay all the way to 0 (rare)
3. You're debugging and want to see what happens

**In practice:** Just set min_lr. It's a safety net. Costs nothing.
