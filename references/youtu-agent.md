# Youtu-Agent training_free_GRPO Branch

## Highlights
- Training-Free GRPO extends the base agent stack with semantic advantage learning while leaving weights untouched and API spend low.

```1:11:/data/Qwen3-VL/references/youtu-agent/README.md
# <img src="docs/assets/logo.svg" alt="Youtu-agent Logo" height="24px"> Training-Free GRPO Built on Youtu-Agent
[![arXiv](https://img.shields.io/badge/arXiv-2510.08191-b31b1b.svg)](https://arxiv.org/abs/2510.08191)
`Youtu-Agent` is a flexible, high-performance framework for building, running, and evaluating autonomous agents. Beyond topping the benchmarks, this framework delivers powerful agent capabilities, e.g. data analysis, file processing, and deep research, all with open-source models.
`Training-Free GRPO` is a cost-effective solution that further enhances `Youtu-Agent` performance without any LLM parameter updates. It consumes significantly fewer training data and lower costs on improving the 671B DeepSeek-V3.1-Terminus than fine-tuning a 32B model.
`Training-Free GRPO` leverages the group relative semantic advantage instead of numerical ones of rollouts in vanilla GRPO, iteratively distilling high-quality experiential knowledge during multi-epoch learning. Such knowledge serves as the learned token prior, which is seamlessly integrated during LLM API calls to guide model behavior.
```

## Key Surfaces
- `training_free_grpo/train.py` orchestrates domain-aware data loaders, experience injection, and rollout scheduling.
- `training_free_grpo/main.py` reuses the same rollout engine for evaluation and pass@k scoring.
- `training_free_grpo/math/*` contains math-specific prompts, dataset interfaces, verifiers, and the semantic experience updater.
- `training_free_grpo/web/*` mirrors the math stack while adding group-level consolidation tailored for web search trajectories.

## End-to-End Training Loop

### 1. Domain-aware bootstrap
`train.py` swaps in the correct dataset, verifier, and experience updater for math or web tasks, then builds the chosen agent profile (prompt-only or tool-augmented) before training begins.

```17:44:/data/Qwen3-VL/references/youtu-agent/training_free_grpo/train.py
    if args.domain == "math":
        from training_free_grpo.math.dataset import load_data
        from training_free_grpo.math.verify import verify_func
        from training_free_grpo.math.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE
        from training_free_grpo.math.experience import ExperienceUpdater
        config_name = "simple/math_agent.yaml"
    elif args.domain == "web":
        from training_free_grpo.web.dataset import load_data
        from training_free_grpo.web.verify import verify_func
        from training_free_grpo.web.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE
        from training_free_grpo.web.experience import ExperienceUpdater
        config_name = "simple/base_search.yaml"
    else:
        raise ValueError(f"Unsupported domain: {args.domain}")
    
    # Create experiment directory
    experiment_dir = os.path.join("data", args.domain, "train", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Set up the agent
    if args.mode == "prompt":
        worker_agent = None
    elif args.mode == "agent":
        config = ConfigLoader.load_agent_config(config_name)
        config.model.model_settings.temperature = args.rollout_temperature
        worker_agent = SimpleAgent(config=config)
        await worker_agent.build()
    else:
        raise ValueError(f"Unsupported inference mode: {args.mode}")
```

### 2. Shuffle, prompt augmentation, and GRPO grouping
Samples are shuffled per epoch, reopened if cached, augmented with previously learned experiences, duplicated `grpo_n` times, and rolled out as a batch. Outputs land in step-specific folders alongside stats so runs can resume safely.

```69:140:/data/Qwen3-VL/references/youtu-agent/training_free_grpo/train.py
        # Check if shuffled data already exists for this epoch
        shuffled_filename = os.path.join(cur_epoch_dir, "shuffled_data.jsonl")
        if os.path.exists(shuffled_filename):
            shuffled_data = []
            with open(shuffled_filename) as f:
                for line in f:
                    shuffled_data.append(json.loads(line))
            print(f"Loaded {len(shuffled_data)} records from shuffled data")
        else:
            print(f"Shuffling data ...")
            shuffled_data = copy.deepcopy(train_data)
            random.shuffle(shuffled_data)
            with open(shuffled_filename, "w") as f:
                for each in shuffled_data:
                    f.write(json.dumps(each) + "\n")

        # for each batch
        num_batches = len(shuffled_data) // args.batchsize
        for batch_idx in range(num_batches):
            step = epoch * num_batches + batch_idx
            if f"step_{step}" not in stats:
                stats[f"step_{step}"] = {"epoch": epoch, "batch": batch_idx, "complete": False}
            elif stats[f"step_{step}"]["complete"]:
                continue

            # Init
            print(f"Step {step} (Epoch {epoch}, Batch {batch_idx})")
            cur_step_dir = os.path.join(experiment_dir, f"step_{step}")
            os.makedirs(cur_step_dir, exist_ok=True)
            
            # Get current batch data
            batch_data = copy.deepcopy(shuffled_data[batch_idx * args.batchsize : (batch_idx + 1) * args.batchsize])

            # Load existing rollouts
            rollout_filename = os.path.join(cur_step_dir, "rollout.jsonl")
            rollouts = load_rollouts(rollout_filename)
            
            # Retrieve experiences for this batch (except first step)
            if step > 0:
                experience_filename = os.path.join("data", args.domain, "train", args.experiment_name, f"step_{step}/experiences.json")
                experiences = json.load(open(experience_filename))
            else:
                experiences = {}
            
            # Format the batch data with experiences
            formatted_experiences = "\n".join([ f"[{i}]. {e}" for i, e in experiences.items() ])
            formatted_batch_data = [{
                "prompt": PROBLEM_WITH_EXPERIENCE_TEMPLATE.format(
                    experiences=formatted_experiences if formatted_experiences else "None",
                    problem=each["problem"],
                ) if experiences else each["problem"],
                **each
            } for each in batch_data]
            
            # Duplicate for GRPO
            print(f"GRPO rollout number={args.grpo_n}")
            formatted_batch_data = formatted_batch_data * args.grpo_n

            # Rollout the dataset
            rollouts, rollout_stats = await rollout_dataset(
                worker_agent=worker_agent,
                data=formatted_batch_data,
                rollouts=rollouts,
                verify_func=verify_func,
                rollout_filename=rollout_filename,
                rollout_concurrency=args.rollout_concurrency,
                task_timeout=args.task_timeout,
                temperature=args.rollout_temperature,
                max_tokens=args.rollout_max_tokens,
            )
            stats[f"step_{step}"]["rollout"] = rollout_stats
```

```142:157:/data/Qwen3-VL/references/youtu-agent/training_free_grpo/train.py
            # Generate critiques and update experiences
            next_step_dir = os.path.join(experiment_dir, f"step_{step+1}")
            os.makedirs(next_step_dir, exist_ok=True)
            next_experience_filename = os.path.join(next_step_dir, "experiences.json")
            if os.path.exists(next_experience_filename):
                print(f"Experiences already exist for step {step}, skipping experience update")
            else:
                new_experiences = ExperienceUpdater().run(
                    rollouts=rollouts, 
                    experiences=experiences,
                    save_dir=cur_step_dir,
                    max_workers=args.rollout_concurrency,
                    given_ground_truth=True if args.given_ground_truth=="True" else False,
                    only_partial_correct=True if args.grpo_n > 1 else False,
                )
                json.dump(new_experiences, open(next_experience_filename, "w"), indent=2)
                print(f"Saved {len(new_experiences)} experiences to {next_experience_filename}")

            # Save stats
            stats[f"step_{step}"]["complete"] = True
            json.dump(stats, open(stats_filename, "w"), indent=2)
```

### 3. Rollout scheduler and reward logging
`rollout_dataset` maintains an async worker queue, retries transient failures, persists intermediate trajectories, and computes pass@k statistics per group.

```34:178:/data/Qwen3-VL/references/youtu-agent/training_free_grpo/main.py
async def rollout_dataset(
    worker_agent: SimpleAgent | None,
    data: list[dict],
    rollouts: list[dict],
    rollout_filename: str,
    verify_func: callable,
    rollout_concurrency: int = 5,
    task_timeout: float = 3600,
    max_retries: int = 3,
    temperature: float = 0.3,
    max_tokens: int = 16384,
) -> list[dict]:
    """Rollout the dataset using the worker agent with concurrency control, timeout, error handling, and retries."""

    # examine data and existing rollouts
    if len(rollouts) > 0:
        for each in rollouts:
            assert "runid" in each
        data_problems = [each["problem"] for each in data]
        rollouts_problems = [each["problem"] for each in rollouts]
        assert data_problems == rollouts_problems, (
            f"The problems in data should be the same as existing rollouts {rollout_filename}"
        )
    else:
        for sample in data:
            assert "problem" in sample and "groundtruth" in sample
        rollouts = [{"runid": i, **sample} for i, sample in enumerate(data)]
    save_rollouts(rollouts, rollout_filename)

    # create task queue
    task_queue = asyncio.Queue()
    pending_tasks_count = 0
    for sample in rollouts:
        if "trajectories" not in sample or len(sample["trajectories"]) == 0:
            sample_with_retry = copy.deepcopy(sample)
            sample_with_retry["retry_count"] = 0
            await task_queue.put(sample_with_retry)
            pending_tasks_count += 1
    pbar = tqdm(total=pending_tasks_count, desc="Rolling out")

    async def worker(name: str):
        while not task_queue.empty():
            sample = await task_queue.get()
            task_start_time = time.time()
            try:
                if worker_agent is None:
                    llm = LLM()
                    coro = asyncio.to_thread(llm.chat, sample["prompt"], temperature=temperature, max_tokens=max_tokens)
                    res = await asyncio.wait_for(coro, timeout=task_timeout)                    
                    res = TaskRecorder(
                            final_output=res,
                            trajectories=[{
                                "trajectory": [
                                    {"role": "user", "content": sample["prompt"]},
                                    {"role": "assistant", "content": res}
                                ]
                            }],
                        )
                else:
                    async with worker_agent as agent:
                        async def rollout_streamed(sample) -> TaskRecorder:
                            prompt = sample.get("prompt", sample["problem"])
                            res = agent.run_streamed(prompt)
                            async for _ in res.stream_events(): pass
                            traj = AgentsUtils.get_trajectory_from_agent_result(res)
                            return TaskRecorder(
                                final_output=res.final_output,
                                trajectories=[traj],
                            )
                        res = await asyncio.wait_for(rollout_streamed(sample), timeout=task_timeout)
                
                task_end_time = time.time()
                sample.update(
                    {
                        "response": res.final_output,
                        "trajectories": res.trajectories,
                        "error": None,
                        "rollout_time": task_end_time - task_start_time,
                    }
                )
                sample["reward"] = verify_func(sample, sample["groundtruth"])
                
                # Task succeeded
                rollouts[sample["runid"]] = sample
                save_rollouts(rollouts, rollout_filename)
                pbar.update(1)
            except Exception as e:
                task_end_time = time.time()
                sample["retry_count"] += 1
                error_info = traceback.format_exc()
                print(f"> error: {error_info}")
                
                if sample["retry_count"] <= max_retries:
                    tqdm.write(f"Worker {name}: Task runid={sample['runid']} failed with {type(e).__name__}. Retrying ({sample['retry_count']}/{max_retries})...")
                    await task_queue.put(sample) # Re-queue the task
                else:
                    tqdm.write(f"Worker {name}: Task runid={sample['runid']} failed after {max_retries} retries. Error: {e}. Traceback: {error_info}")
                    sample.update(
                        {
                            "response": f"Error: {str(e)} after {max_retries} retries.",
                            "trajectories": [],
                            "error": error_info,
                            "reward": 0,
                            "rollout_time": task_end_time - task_start_time,
                        }
                    )
                    
                    # Task failed permanently
                    rollouts[sample["runid"]] = sample
                    save_rollouts(rollouts, rollout_filename)
                    pbar.update(1)
            finally:
                task_queue.task_done()
```

### 4. Math semantic experience builder
The math `ExperienceUpdater` summarizes partially correct trajectories, critiques grouped rollouts, and applies batched LLM-driven edits before renumbering experiences for the next step.

```22:52:/data/Qwen3-VL/references/youtu-agent/training_free_grpo/math/experience.py
class ExperienceUpdater:
    def __init__(self):
        self.llm = LLM()

    def run(self, rollouts, experiences, save_dir, max_workers=16, given_ground_truth=True, only_partial_correct=True):
        # 1. Summarize trajectory for each rollout
        problem_to_summarized_rollouts = self._single_rollout_summary(
            rollouts=rollouts, 
            save_dir=save_dir, 
            max_workers=max_workers,
            given_ground_truth=given_ground_truth,
            only_partial_correct=only_partial_correct
        )

        # 2. Generate critique for each query
        critiques = self._single_query_critique(
            problem_to_summarized_rollouts=problem_to_summarized_rollouts, 
            experiences=experiences,
            save_dir=save_dir, 
            max_workers=max_workers,
            given_ground_truth=given_ground_truth,
            only_partial_correct=only_partial_correct
        )

        # 3. batch update experiences
        new_experiences = self._batch_update(
            experiences=experiences, 
            critiques=critiques, 
            save_dir=save_dir
        )

        # 4. assign new experience IDs
        new_experiences = {
            f"G{i}": exp for i, exp in enumerate(new_experiences.values())
        }
        return new_experiences
```

### 5. Prompt template injection
Every math prompt inserts the curated experience list ahead of the problem to act as a learned token prior.

```1:5:/data/Qwen3-VL/references/youtu-agent/training_free_grpo/math/prompts.py
PROBLEM_WITH_EXPERIENCE_TEMPLATE = """Please solve the problem:
{problem}

When solving problems, you MUST first carefully read and understand the helpful instructions and experiences:
{experiences}"""
```

### 6. Web-specific consolidation
The web updater adds an extra group-level reconciliation stage before batching, matching the paper’s semantic advantage workflow for search trajectories.

```26:63:/data/Qwen3-VL/references/youtu-agent/training_free_grpo/web/experience.py
    def run(self, rollouts, experiences, save_dir, max_workers=16, given_ground_truth=True):
        # 1. Summarize trajectory for each rollout
        problem_to_summarized_rollouts = self._single_rollout_summary(
            rollouts=rollouts, 
            save_dir=save_dir, 
            max_workers=max_workers,
            given_ground_truth=given_ground_truth
        )

        # 2. Generate critique for each query
        new_experiences = self._single_query_critique(
            problem_to_summarized_rollouts=problem_to_summarized_rollouts, 
            experiences=experiences,
            save_dir=save_dir, 
            max_workers=max_workers,
            given_ground_truth=given_ground_truth
        )

        # 3. group update experiences
        critiques = self._group_update(
            experiences=experiences, 
            new_experiences=new_experiences, 
            save_dir=save_dir,
            max_workers=max_workers
        )

        # 4. batch update experiences
        new_experiences = self._batch_update(
            experiences=experiences, 
            critiques=critiques, 
            save_dir=save_dir
        )

        # 5. assign new experience IDs
        new_experiences = {
            f"G{i}": exp for i, exp in enumerate(new_experiences.values())
        }
        return new_experiences
```

### 7. Evaluation harness
Evaluation loads saved experiences, duplicates samples for pass@k, and reuses the rollout engine under `data/{domain}/eval`.

```216:248:/data/Qwen3-VL/references/youtu-agent/training_free_grpo/main.py
    # Insert experiences
    if args.experience_file:
        experiences = json.load(open(args.experience_file))
        formatted_experiences = "\n".join([ f"[{i}]. {e}" for i, e in experiences.items() ])
        formatted_test_data = [{
            "prompt": PROBLEM_WITH_EXPERIENCE_TEMPLATE.format(
                experiences=formatted_experiences if formatted_experiences else "None",
                problem=each["problem"],
            ),
            **each
        } for each in test_data]
    else:
        formatted_test_data = [{
            "prompt": each["problem"],
            **each
        } for each in test_data]
    
    # Duplicate for Pass@k evaluation
    formatted_test_data = formatted_test_data * args.pass_k
    print(f"Duplicated to {len(formatted_test_data)} records for Pass@{args.pass_k} evaluation")

    # Load existing rollouts
    os.makedirs(f"data/{args.domain}/eval", exist_ok=True)
    rollout_filename = f"data/{args.domain}/eval/{args.experiment_name}.jsonl"
    rollouts = load_rollouts(rollout_filename)

    # Rollout the dataset
    await rollout_dataset(
        worker_agent=worker_agent,
        data=formatted_test_data,
        rollouts=rollouts,
        verify_func=verify_func,
        rollout_filename=rollout_filename,
        rollout_concurrency=args.rollout_concurrency,
        task_timeout=args.task_timeout,
        max_tokens=args.rollout_max_tokens,
    )
```

## Verification Signals
Math rewards leverage `math_verify` to compare boxed answers against symbolic parses, handling timeouts gracefully.

```6:17:/data/Qwen3-VL/references/youtu-agent/training_free_grpo/math/verify.py
def verify_func(sample: dict, ground_truth: str, timeout_score: float = 0) -> float:
    model_output = sample["response"]
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + str(ground_truth) + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return float(ret_score)
```

## CLI Quick Start
- Math training on 100 DAPO samples (README explains the recipe).

```82:83:/data/Qwen3-VL/references/youtu-agent/README.md
The following command runs a training session on a truncated `DAPO-Math-17k` dataset with only 100 problems.
```

```bash
python train.py \
    --mode agent \
    --domain math \
    --experiment_name DAPO100 \
    --dataset DAPO-Math-17k \
    --dataset_truncate 100 \
    --epochs 3 \
    --batchsize 100 \
    --grpo_n 5 \
    --rollout_concurrency 128 \
    --rollout_temperature 0.7 \
    --task_timeout 1800
```

- Web training with 100 sampled AFM tasks.

```99:101:/data/Qwen3-VL/references/youtu-agent/README.md
For `web` domain, you can run the following command to train on the `AFM_web_RL` dataset (randomly sampled 100 examples by setting the `--dataset` to be `{dataset_name}_{sample_number}`).
```

```bash
python train.py \
    --mode agent \
    --domain web \
    --experiment_name AFM_web_RL_100 \
    --dataset AFM_web_RL_100 \
    --epochs 3 \
    --batchsize 4 \
    --grpo_n 5 \
    --rollout_concurrency 128 \
    --rollout_temperature 0.7 \
    --task_timeout 1800
```

- Pass@32 evaluation with stored experiences.

```129:131:/data/Qwen3-VL/references/youtu-agent/README.md
The following command runs an evaluation on the `AIME24` dataset for the `math` domain, using the experience file saved after 3-step learning.
```

```bash
python main.py \
    --mode agent \
    --domain math \
    --experiment_name AIME24_test_step_3 \
    --dataset AIME24 \
    --experience_file data/math/train/DAPO100/step_3/experiences.json \
    --rollout_concurrency 128 \
    --pass_k 32
```

## One-Sample Iteration (Example)
1. Step 0 batches inject no experiences, so the prompt is the raw problem duplicated `grpo_n` times. After rollouts, one trajectory scores correctly while others fail.
2. `ExperienceUpdater.run` summarizes and critiques the partially-correct group, synthesizing a new heuristic that is written to `step_1/experiences.json`.
3. Step 1 prepends that heuristic via `PROBLEM_WITH_EXPERIENCE_TEMPLATE`, improving the likelihood of success for the next sample.
4. Verification assigns binary rewards, the experience file for `step_2` is updated, and `stats.json` records the step as complete, enabling interruption-free restarts.

```
