import os
os.environ.setdefault('MAX_PIXELS', '147456')
os.environ.setdefault('VIDEO_MAX_PIXELS', '147456')
os.environ.setdefault('FPS_MAX_FRAMES', '2')
os.environ.setdefault('FPS_MIN_FRAMES', '2')
os.environ.setdefault('USE_AUDIO_IN_VIDEO', 'true')
os.environ['VLLM_USE_TRITON_ROPE'] = '0'

import re
import json
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import torch

CASCADE_STAGES_DEFAULT = [
    {'name': 'stage1_T1.0', 'temperature': 1.0, 'top_p': 0.95},
    {'name': 'stage2_T0.5', 'temperature': 0.5, 'top_p': 0.95},
    {'name': 'stage3_T1.3', 'temperature': 1.3, 'top_p': 0.95},
]
CASCADE_STAGE_ARGMAX = {'name': 'stage4_argmax', 'temperature': 0.0, 'top_p': 1.0}

OVERSAMPLE_CAP = 2
INFER_CHUNK_SIZE_DEFAULT = 30
MIN_THINK_LEN = 30


def detect_task(solution):
    has_a = 'Action:' in solution
    has_f = 'Function:' in solution
    if has_a and has_f:
        return 'joint'
    if has_a:
        return 'action'
    if has_f:
        return 'function'
    return None


def parse_axis(text, axis):
    m = re.search(rf'{axis}\s*:\s*(\d)', text, re.IGNORECASE)
    if not m:
        return None
    d = m.group(1)
    return d if d in '012345' else None


def parse_solution_labels(solution, axis):
    m = re.search(rf'{axis}\s*:\s*([\d|]+)', solution)
    if not m:
        return None
    return [d for d in m.group(1).split('|') if d in '012345'] or None


def parse_generation(text):
    if not text:
        return None, None
    think_m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    answer_m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    think = think_m.group(1).strip() if think_m else None
    answer = answer_m.group(1).strip() if answer_m else None
    return think, answer


def is_generation_valid_correct(generation, solution, task):
    think, answer = parse_generation(generation)
    if not think or not answer:
        return False
    if len(think) < MIN_THINK_LEN:
        return False

    think_lower = think.lower()

    def get_pred(axis):
        p = parse_axis(answer, axis)
        if p is None:
            bare = re.match(r'^\s*(\d)\s*$', answer)
            if bare and bare.group(1) in '012345':
                return bare.group(1)
        return p

    action_keywords = {
        '0': ['aac', 'augmentative', 'device', 'tablet', 'screen', 'icon', 'communication tool'],
        '1': ['body', 'head', 'postural', 'lean', 'rocking', 'turn away', 'whole-body', 'nod'],
        '2': ['face', 'facial', 'smile', 'grimace', 'frown', 'widening eyes', 'expression'],
        '3': ['gesture', 'hand', 'arm', 'limb', 'point', 'reach', 'wave', 'hand leading'],
        '4': ['looking', 'gaze', 'eye', 'look', 'orient', 'referential'],
        '5': ['vocal', 'sound', 'grunt', 'squeal', 'laugh', 'cry', 'word approx', 'noise']
    }

    function_keywords = {
        '0': ['comment', 'share', 'observation', 'draw attention', 'interest'],
        '1': ['emotion', 'feeling', 'happy', 'frustrat', 'excite', 'distress', 'valence'],
        '2': ['reject', 'refus', 'push away', 'protest', 'stop', 'turn away', 'no'],
        '3': ['request', 'want', 'need', 'desire', 'ask', 'obtain'],
        '4': ['sdb', 'self-directed', 'stimming', 'sensory', 'self-sooth', 'repetitive'],
        '5': ['social', 'interact', 'greet', 'reciprocity', 'name', 'proximity', 'routine']
    }

    if task == 'action':
        pred = get_pred('Action')
        valid = parse_solution_labels(solution, 'Action')
        if not (pred and valid and pred in valid): return False
        return any(kw in think_lower for kw in action_keywords[pred])

    elif task == 'function':
        pred = get_pred('Function')
        valid = parse_solution_labels(solution, 'Function')
        if not (pred and valid and pred in valid): return False
        return any(kw in think_lower for kw in function_keywords[pred])

    elif task == 'joint':
        pa = get_pred('Action')
        pf = get_pred('Function')
        va = parse_solution_labels(solution, 'Action')
        vf = parse_solution_labels(solution, 'Function')
        if not (pa and pf and va and vf and pa in va and pf in vf): return False
        action_ok = any(kw in think_lower for kw in action_keywords[pa])
        function_ok = any(kw in think_lower for kw in function_keywords[pf])
        return action_ok and function_ok

    return False


def get_base_name_from_videos(videos):
    if not videos:
        return None
    stem = Path(videos[0]).stem
    m = re.match(r'^(.+)_V\d+$', stem)
    return m.group(1) if m else stem


TASK_TYPE_MAP = {
    'action': 'action_interpretation',
    'function': 'function_interpretation',
    'joint': 'joint_interpretation',
}


def build_sft2_sample(grpo_sample, generation, task, action_id, function_id):
    think, _ = parse_generation(generation)
    if not think or len(think) < MIN_THINK_LEN:
        return None

    if task == 'action':
        answer_str = f"Action: {action_id}"
    elif task == 'function':
        answer_str = f"Function: {function_id}"
    elif task == 'joint':
        answer_str = f"Action: {action_id}, Function: {function_id}"
    else:
        return None

    assistant_content = f"<think>\n{think}\n</think>\n<answer>{answer_str}</answer>"

    return {
        "task_type": TASK_TYPE_MAP[task],
        "_action_id": action_id,
        "_function_id": function_id,
        "_base_name": get_base_name_from_videos(grpo_sample.get('videos', [])),
        "messages": [
            grpo_sample['messages'][0],
            {"role": "assistant", "content": assistant_content},
        ],
        "videos": grpo_sample.get('videos', []),
    }


def load_fallback_traces(path):
    traces = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            base_name = get_base_name_from_videos(s.get('videos', []))
            content = s['messages'][-1]['content']
            _, answer = parse_generation(content)
            task = detect_task(answer or '')
            if base_name is None or task is None:
                continue
            traces[(base_name, task)] = content
    return traces


def compute_balanced_factors(class_counts, cap=OVERSAMPLE_CAP):
    if not class_counts:
        return {}
    max_c = max(class_counts.values())
    return {
        cls: min(cap, max(1, round(((max_c / cnt) ** 0.5) / 2)))
        for cls, cnt in class_counts.items()
    }


def oversample_per_task(samples, task_type):
    if task_type == 'joint_interpretation':
        return samples, {'note': 'joint not oversampled'}

    if task_type == 'action_interpretation':
        keys = ['_action_id']
    elif task_type == 'function_interpretation':
        keys = ['_function_id']
    else:
        return samples, {}

    counts = {k: defaultdict(int) for k in keys}
    for s in samples:
        for k in keys:
            v = s.get(k)
            if v is not None:
                counts[k][v] += 1

    factors = {k: compute_balanced_factors(dict(counts[k])) for k in keys}

    balanced = []
    for s in samples:
        repeats = []
        for k in keys:
            v = s.get(k)
            if v is not None:
                repeats.append(factors[k].get(v, 1))
        rep = max(repeats) if repeats else 1
        for _ in range(rep):
            balanced.append(s)

    return balanced, {
        'counts': {k: dict(counts[k]) for k in keys},
        'factors': factors,
        'before': len(samples),
        'after': len(balanced),
    }


def gens_file_path(output_dir, clip_range_str):
    if clip_range_str:
        safe = clip_range_str.replace(':', '-')
        return output_dir / f'generations_{safe}.jsonl'
    return output_dir / 'generations.jsonl'


def load_all_generations(output_dir):
    gens = defaultdict(list)
    for p in sorted(Path(output_dir).glob('generations*.jsonl')):
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if 'clip_idx' in rec and 'stage' in rec:
                        gens[rec['clip_idx']].append(rec)
                except json.JSONDecodeError:
                    continue
    for idx in gens:
        gens[idx].sort(key=lambda r: (r['stage'], r.get('rollout_idx', 0)))
    return gens


def append_generation(gens_path, rec):
    with open(gens_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def clip_status(gens_for_clip, n_stages):
    completed = set()
    first_correct = None
    rescued_stage = None
    for g in gens_for_clip:
        completed.add(g['stage'])
        if g.get('is_correct') and rescued_stage is None:
            rescued_stage = g['stage']
            first_correct = g['generation']
    return rescued_stage, completed, first_correct


def write_jsonl(samples, path):
    with open(path, 'w', encoding='utf-8') as f:
        for s in samples:
            out = {k: v for k, v in s.items() if not k.startswith('_') and k != 'task_type'}
            json.dump(out, f, ensure_ascii=False)
            f.write('\n')


def parse_clip_range(s, total):
    if s is None:
        return 0, total
    if ':' not in s:
        raise ValueError(f"--clip-range must be 'start:end', got {s!r}")
    a, b = s.split(':', 1)
    start = int(a) if a else 0
    end = int(b) if b else total
    return max(0, start), min(total, end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', default='Qwen/Qwen3-Omni-30B-A3B-Thinking')
    parser.add_argument('--adapter', required=True,
                        help='GRPO1 checkpoint, e.g. /path/to/outputs/grpo1/checkpoint-XXX')
    parser.add_argument('--grpo-dataset', required=True,
                        help='GRPO training JSONL, e.g. /path/to/grpo_data/grpo_train.jsonl')
    parser.add_argument('--fallback-sft', required=True,
                        help='SFT1 JSONL with Gemini reference traces used for unrescued clips, '
                             'e.g. /path/to/sft_data/sft_all_combined.jsonl')
    parser.add_argument('--output', required=True,
                        help='Output directory, e.g. /path/to/rej_sampling')
    parser.add_argument('--max-tokens', type=int, default=384)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--chunk-size', type=int, default=INFER_CHUNK_SIZE_DEFAULT,
                        help='Clips per engine.infer() call.')
    parser.add_argument('--clip-range', type=str, default=None,
                        help="Index range for this job, e.g. '0:1325'. "
                             "Each job writes generations_<start>-<end>.jsonl.")
    parser.add_argument('--include-argmax', action='store_true',
                        help='Add stage 4 (T=0.0 argmax). Off by default.')
    parser.add_argument('--build-only', action='store_true',
                        help='Skip inference; merge all generations*.jsonl and build SFT 2.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit total annotated clips (debug).')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cascade_stages = list(CASCADE_STAGES_DEFAULT)
    if args.include_argmax:
        cascade_stages.append(CASCADE_STAGE_ARGMAX)
    n_stages = len(cascade_stages)

    print(f"Loading GRPO dataset: {args.grpo_dataset}")
    grpo_samples = []
    with open(args.grpo_dataset, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            grpo_samples.append(json.loads(line))
    if args.limit:
        grpo_samples = grpo_samples[:args.limit]
    print(f"  Loaded {len(grpo_samples)} GRPO samples")

    annotated = []
    for s in grpo_samples:
        sol = s.get('solution', '')
        task = detect_task(sol)
        action_labels = parse_solution_labels(sol, 'Action') or []
        function_labels = parse_solution_labels(sol, 'Function') or []
        action_id = int(action_labels[0]) if action_labels else None
        function_id = int(function_labels[0]) if function_labels else None
        base_name = get_base_name_from_videos(s.get('videos', []))
        if task is None or base_name is None:
            continue
        annotated.append({
            'sample': s,
            'task': task,
            'action_id': action_id,
            'function_id': function_id,
            'base_name': base_name,
        })
    print(f"  Annotated: {len(annotated)} valid samples")

    all_gens = load_all_generations(output_dir)
    n_rescued_already = sum(
        1 for idx in range(len(annotated))
        if any(g.get('is_correct') for g in all_gens.get(idx, []))
    )
    print(f"  Existing generations: {sum(len(v) for v in all_gens.values())} rollouts, "
          f"{n_rescued_already} clips already rescued")

    range_start, range_end = parse_clip_range(args.clip_range, len(annotated))
    if not args.build_only:
        print(f"  This job processes clips [{range_start}:{range_end}] "
              f"({range_end - range_start} clips)")

    if not args.build_only:
        gens_path = gens_file_path(output_dir, args.clip_range)
        print(f"  Writing this job's generations to: {gens_path.name}")

        print(f"\nInitializing TransformersEngine: {args.adapter}")
        from swift.infer_engine import TransformersEngine, RequestConfig, InferRequest
        from swift.template import TemplateType

        engine = TransformersEngine(
            model=args.adapter,
            template_type=TemplateType.qwen3_omni,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            max_model_len=args.max_model_len,
        )
        print("  Engine ready.\n")

        for stage_zero_idx, stage_cfg in enumerate(cascade_stages):
            stage_num = stage_zero_idx + 1

            pending = []
            for idx in range(range_start, range_end):
                rescued_stage, completed, _ = clip_status(all_gens.get(idx, []), n_stages)
                if rescued_stage is not None:
                    continue
                if stage_num in completed:
                    continue
                pending.append(idx)

            print(f"=== Stage {stage_num}/{n_stages}: {stage_cfg['name']} "
                  f"({len(pending)} pending) ===")
            if not pending:
                continue

            cfg = RequestConfig(
                temperature=stage_cfg['temperature'],
                top_p=stage_cfg['top_p'],
                n=1,
                max_tokens=args.max_tokens,
            )

            chunk_size = args.chunk_size
            n_chunks = (len(pending) + chunk_size - 1) // chunk_size
            stage_t0 = time.time()
            rescued_this_stage = 0

            for ci in range(n_chunks):
                chunk_indices = pending[ci * chunk_size:(ci + 1) * chunk_size]
                infer_requests = []
                for idx in chunk_indices:
                    ann = annotated[idx]
                    req = InferRequest(
                        messages=[{
                            'role': 'user',
                            'content': ann['sample']['messages'][0]['content']
                        }],
                        videos=ann['sample'].get('videos', []),
                    )
                    infer_requests.append(req)

                t0 = time.time()
                try:
                    resp = engine.infer(infer_requests, request_config=cfg)
                except Exception as e:
                    print(f"    chunk {ci+1}/{n_chunks} FAILED: {str(e)[:200]}")
                    resp = [None] * len(infer_requests)
                dt = time.time() - t0

                chunk_rescued = 0
                for i, idx in enumerate(chunk_indices):
                    ann = annotated[idx]
                    response = resp[i] if i < len(resp) else None
                    if response is None:
                        continue
                    try:
                        gen_text = response.choices[0].message.content
                    except Exception:
                        gen_text = None
                    if not gen_text:
                        continue

                    is_correct = is_generation_valid_correct(
                        gen_text, ann['sample']['solution'], ann['task']
                    )
                    rec = {
                        'clip_idx': idx,
                        'stage': stage_num,
                        'rollout_idx': 0,
                        'generation': gen_text,
                        'is_correct': is_correct,
                        'task': ann['task'],
                        'action_id': ann['action_id'],
                        'function_id': ann['function_id'],
                        'timestamp': time.time(),
                    }
                    append_generation(gens_path, rec)
                    all_gens[idx].append(rec)
                    if is_correct:
                        chunk_rescued += 1
                        rescued_this_stage += 1

                elapsed = time.time() - stage_t0
                done = ci + 1
                eta = elapsed * (n_chunks - done) / done if done else 0
                print(f"    chunk {done}/{n_chunks}: {len(resp)} responses in {dt:.0f}s "
                      f"({chunk_rescued} rescued) — ETA {eta/3600:.1f}h")

            print(f"  Stage {stage_num} done: rescued {rescued_this_stage}/{len(pending)} "
                  f"in {(time.time()-stage_t0)/3600:.1f}h\n")

        print(f"=== Job complete for range [{range_start}:{range_end}] ===")
        if args.clip_range is not None:
            print(f"  When all jobs finish, run with --build-only to assemble SFT 2.")
            return

    print(f"\n=== Building SFT 2 samples ===")
    all_gens = load_all_generations(output_dir)

    print(f"  Loading fallback traces: {args.fallback_sft}")
    fallback_traces = load_fallback_traces(args.fallback_sft)
    print(f"  Loaded {len(fallback_traces)} fallback traces")

    task_samples = defaultdict(list)
    rescue_by_stage = Counter()
    clip_outcome = {}
    for idx in range(len(annotated)):
        ann = annotated[idx]
        rescued_stage, _, first_correct = clip_status(all_gens.get(idx, []), n_stages)
        used_fallback = rescued_stage is None or first_correct is None
        if used_fallback:
            generation = fallback_traces.get((ann['base_name'], ann['task']))
        else:
            generation = first_correct

        sft = None
        if generation is not None:
            sft = build_sft2_sample(
                ann['sample'], generation, ann['task'],
                ann['action_id'], ann['function_id']
            )
        if sft:
            task_samples[sft['task_type']].append(sft)
            rescue_by_stage['fallback' if used_fallback else rescued_stage] += 1
        clip_outcome[idx] = (rescued_stage if not used_fallback else None, used_fallback, sft is not None)

    n_missing = sum(1 for _, _, kept in clip_outcome.values() if not kept)
    if n_missing:
        print(f"  WARNING: {n_missing} clips had no rescued generation and no fallback trace")

    for tt, samps in sorted(task_samples.items()):
        print(f"  {tt}: {len(samps)} samples")

    print(f"\n=== Oversampling (sqrt-softened/2, cap={OVERSAMPLE_CAP}x; joint skipped) ===")
    all_unbalanced = []
    all_balanced = []
    for tt in sorted(task_samples.keys()):
        samples = task_samples[tt]
        write_jsonl(samples, output_dir / f'sft2_{tt}.jsonl')
        all_unbalanced.extend(samples)

        balanced, stats = oversample_per_task(samples, tt)
        write_jsonl(balanced, output_dir / f'sft2_{tt}_balanced.jsonl')
        all_balanced.extend(balanced)

        print(f"  {tt}:")
        print(f"    {len(samples)} -> {len(balanced)}")
        if 'factors' in stats:
            print(f"    counts:  {stats['counts']}")
            print(f"    factors: {stats['factors']}")

    write_jsonl(all_unbalanced, output_dir / 'sft2_all.jsonl')
    write_jsonl(all_balanced, output_dir / 'sft2_all_balanced.jsonl')
    random.shuffle(all_balanced)
    write_jsonl(all_balanced, output_dir / 'sft2_all_balanced_shuffled.jsonl')
    print(f"\n  sft2_all_balanced_shuffled.jsonl: {len(all_balanced)} samples")

    difficulty = {}
    for idx in range(len(annotated)):
        ann = annotated[idx]
        rescued_stage, used_fallback, kept = clip_outcome[idx]
        diff_score = ((rescued_stage - 1) / max(1, n_stages - 1)) if rescued_stage else 1.0
        key = f"{ann['base_name']}|{ann['task']}"
        difficulty[key] = {
            'base_name': ann['base_name'],
            'task': ann['task'],
            'action_id': ann['action_id'],
            'function_id': ann['function_id'],
            'rescue_stage': rescued_stage,
            'used_gemini_fallback': used_fallback,
            'difficulty': diff_score,
            'kept_for_sft2': kept,
            'n_rollouts_attempted': len(all_gens.get(idx, [])),
        }
    with open(output_dir / 'clip_difficulty.json', 'w') as f:
        json.dump(difficulty, f, indent=2)

    survival_by_class = defaultdict(lambda: {'rescued': 0, 'fallback': 0, 'attempted': 0})
    for idx, ann in enumerate(annotated):
        if ann['task'] == 'action':
            cls_key = f"action_class_{ann['action_id']}"
        elif ann['task'] == 'function':
            cls_key = f"function_class_{ann['function_id']}"
        else:
            cls_key = f"joint_a{ann['action_id']}_f{ann['function_id']}"
        survival_by_class[cls_key]['attempted'] += 1
        rescued_stage, used_fallback, kept = clip_outcome[idx]
        if rescued_stage is not None:
            survival_by_class[cls_key]['rescued'] += 1
        elif used_fallback and kept:
            survival_by_class[cls_key]['fallback'] += 1

    n_fallback = rescue_by_stage.get('fallback', 0)
    rescued_total = sum(v for k, v in rescue_by_stage.items() if k != 'fallback')
    stats = {
        'total_clips_attempted': len(annotated),
        'rescued': rescued_total,
        'fallback': n_fallback,
        'missing': n_missing,
        'rescue_by_stage': {
            **{f'stage{k}': rescue_by_stage[k] for k in sorted(k for k in rescue_by_stage if k != 'fallback')},
            'fallback': n_fallback,
        },
        'survival_by_class': dict(survival_by_class),
    }
    with open(output_dir / 'rejection_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Final stats ===")
    print(f"  Total attempted: {stats['total_clips_attempted']}")
    print(f"  Rescued:         {stats['rescued']}")
    print(f"  Fallback:        {stats['fallback']}")
    print(f"  Missing:         {stats['missing']}")
    print(f"  Rescue by stage: {stats['rescue_by_stage']}")
    print(f"\n  Per-class outcome:")
    for k, v in sorted(survival_by_class.items()):
        rate = (v['rescued'] / v['attempted'] * 100) if v['attempted'] else 0.0
        print(f"    {k}: {v['rescued']} rescued, {v['fallback']} fallback / {v['attempted']} ({rate:.1f}% rescued)")

    print(f"\n  Output: {output_dir}")


if __name__ == "__main__":
    main()
