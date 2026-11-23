import os
import re
import json
import random
import time
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
from google import genai
from google.genai import types


ACTION_MAP = {
    0: ("AAC", "Alternate and Augmented Communication: Use of high-tech (tablets, speech-generating devices) or low-tech (picture cards, communication boards) AAC systems."),
    1: ("Body", "Use of primarily the body or head. This includes more holistic movements like postural shifts (e.g., leaning in, turning away), or whole-body movements (e.g., rocking, walking away)."),
    2: ("Face", "Use of facial expressions to convey meaning (e.g., smiling, grimacing, frowning) that is not primarily a gaze shift."),
    3: ("Gesture", "Use of the hands, arms, or limbs to communicate. This includes specific, directed movements like pointing, reaching, waving, or hand leading."),
    4: ("Looking", "Use of eye gaze or head orientation to direct another person's attention to a specific subject, person, or location."),
    5: ("Vocalization", "Use of any non-speech or speech-like sound made with the vocal tract to communicate (e.g., grunt, squeal, laugh, word approximation)."),
}

FUNCTION_MAP = {
    0: ("Commenting", "Sharing observations, expressing interest, or drawing attention to objects or events."),
    1: ("Emotion", "Expressing an internal feeling state like happiness, frustration, excitement, or distress."),
    2: ("Reject", "Refusing items, activities, pushing away, head shaking, or vocal protests indicating a desire for an activity to stop."),
    3: ("Request", "Asking for objects, actions, continuation, or assistance through any modality."),
    4: ("SDB", "Self-Directed Behavior: Behaviors that were not perceived as being intentionally communicative or directed at another person, like self-stimulatory behavior."),
    5: ("Social", "Initiating social interaction, maintaining social reciprocity, greeting, responding to a name, or joint attention."),
}

ACTION_DEFS = "ACTION classes:\n" + "\n".join(
    f"  {k}: {v[0]} — {v[1]}" for k, v in ACTION_MAP.items()
)

FUNCTION_DEFS = "FUNCTION classes:\n" + "\n".join(
    f"  {k}: {v[0]} — {v[1]}" for k, v in FUNCTION_MAP.items()
)

ACTION_COGNITION_HINTS = {
    0: "The label is AAC (Alternate and Augmented Communication). Describe how the child is communicating through a device or external communication tool. Focus on: which device or system the child interacts with (tablet, speech-generating device, picture cards, communication board); the specific motor action used to engage the device (pressing, pointing at, handing over, activating); and any observable communicative intent behind the device use, such as selecting a symbol or triggering speech output.",
    1: "The label is Body. Describe how the child uses whole-body or head movements as the primary communicative act. Focus on: postural shifts such as leaning toward or away from the caregiver; whole-body movements like rocking, turning, walking toward or away; head movements such as nodding, shaking, or turning; and how these movements appear directed or purposeful in the communicative context rather than being incidental or self-stimulatory.",
    2: "The label is Face. Describe how the child uses facial expressions to convey communicative meaning. Focus on: the specific expression observed (smiling, grimacing, frowning, widening eyes, lip movements); whether the expression appears directed toward the caregiver or in response to a stimulus; how the expression differs from a neutral resting face; and why this is primarily a facial expression rather than a gaze shift or vocalization.",
    3: "The label is Gesture. Describe how the child uses hand, arm, or limb movements to communicate. Focus on: the specific gesture (pointing, reaching toward an object or person, waving, pulling the caregiver's hand, pushing away, giving or showing an object); which limb is used and the direction/target of the movement; and how the gesture serves a communicative purpose, such as directing the caregiver's attention or indicating a want.",
    4: "The label is Looking. Describe how the child uses eye gaze or head orientation to direct attention communicatively. Focus on: the target of the child's gaze (a specific object, the caregiver's face, a location); any gaze shifts between the caregiver and an object or location (referential gaze); head turning or orientation used to indicate direction of attention; and how the looking behavior appears intentional and communicative rather than incidental scanning.",
    5: "The label is Vocalization. Describe the non-speech or speech-like sounds the child produces to communicate. Focus on: the type of vocalization (grunt, squeal, cry, laugh, hum, babble, word approximation, vowel sound); the volume, intensity, or emotional quality of the sound; whether the vocalization appears directed at the caregiver or produced in response to a stimulus; and any temporal relationship between the vocalization and the interaction (e.g., vocalizing after the caregiver speaks, or while reaching for something).",
}

FUNCTION_COGNITION_HINTS = {
    0: "The function is Commenting. Describe how the child's action serves to share information, draw attention to something, or remark on an event or object without requesting it. Focus on: whether the child is indicating something in the environment for the purpose of sharing interest; any joint attention bids (looking at object then at caregiver); and how this differs from a request — the child is not trying to obtain the object or action, but rather calling attention to it or expressing awareness.",
    1: "The function is Emotion. Describe how the child's action expresses an internal emotional state. Focus on: the observable emotional valence (positive: joy, excitement, contentment; negative: frustration, distress, displeasure); whether the emotion appears to be expressed spontaneously or in reaction to a specific stimulus or event; and how the primary communicative intent is emotional expression rather than making a request or social bid.",
    2: "The function is Reject. Describe how the child's action communicates refusal, protest, or resistance. Focus on: what the child appears to be rejecting (an offered object, an activity, physical proximity, a caregiver's action); the behavioral indicators of rejection (pushing away, turning away, pulling back, shaking head, distress vocalizations); and how the action clearly signals 'no' or 'stop' in the interactional context.",
    3: "The function is Request. Describe how the child's action communicates a want, need, or desire for something. Focus on: what the child appears to be requesting (an object, an action from the caregiver, continuation of an activity, help); the behavioral means of requesting (reaching, leading the caregiver's hand, looking between object and caregiver, vocalizing with intent); and how the action is directed toward obtaining something rather than merely commenting or expressing emotion.",
    4: "The function is Self-Directed Behavior. Describe how the child's action appears self-regulatory or self-directed rather than communicatively intended toward another person. Focus on: whether the behavior serves a sensory or regulatory function (stimming, self-soothing, repetitive movements); the absence of clear social directedness (not oriented toward caregiver); and how this behavior, while observable and potentially informative to a caregiver, does not have an apparent communicative target or social intent.",
    5: "The function is Social. Describe how the child's action serves a social or interactive purpose. Focus on: bids for social interaction (greeting, turn-taking, reciprocal play, seeking proximity); responses to social cues from the caregiver (responding to name, reciprocating a smile, engaging in a social routine); and how the primary function is maintaining or initiating social connection rather than requesting, rejecting, or commenting on a specific object or event.",
}


FORCE_TEST_SESSIONS = {'TALK009', 'TALK015b', 'TALK026b'}

SESSION_TASK_RESTRICTIONS = {
    'TALK012': {'function'},
    'TALK025': {'action'},
    'TALK028': {'joint'},
}

ALL_TASKS = {'action', 'function', 'joint'}

TASK_TO_TYPE = {
    'action': 'action_interpretation',
    'function': 'function_interpretation',
    'joint': 'joint_interpretation',
}


GEMINI_SYSTEM_PREAMBLE = (
    "You are an expert behavioral analyst observing a short video clip "
    "of a non- or minimally-speaking child with profound autism and complex "
    "neurodevelopmental disorders interacting with their caregiver. "
    "You are shown {num_views} camera angle(s) of the same moment. "
    "Focus ONLY on the child — ignore adults except as context for the child's actions. "
    "Integrate observations from all camera views. "
    "Write in a clinical, objective tone. Be concise but specific."
)


def build_gemini_action_prompt(num_views, action_id):
    action_name, action_def = ACTION_MAP[action_id]
    return (
        GEMINI_SYSTEM_PREAMBLE.format(num_views=num_views) + "\n\n"
        f"CONTEXT: The child's communicative ACTION in this clip is:\n"
        f"  Class {action_id}: {action_name} — {action_def}\n\n"
        "For reference, the complete ACTION classification scheme is:\n"
        f"{ACTION_DEFS}\n\n"
        f"GUIDANCE FOR THIS LABEL: {ACTION_COGNITION_HINTS[action_id]}\n\n"
        "YOUR TASK: Produce a reasoning trace that explains why this action label fits "
        "the observable physical behavior in the video. Reference what the child's hands, "
        "gaze, body, and posture are doing. Make the reasoning specific to this clip — "
        "do not write generic descriptions.\n\n"
        "OUTPUT FORMAT — output EXACTLY this structure, with NO text before or after:\n\n"
        "<think>\n"
        "[4-6 sentences of reasoning grounded in observable physical behavior]\n"
        "</think>\n"
        f"<answer>Action: {action_id}</answer>\n\n"
        f'The <answer> tag must contain only "Action: {action_id}" with no other text. '
        "Do not write anything outside the <think> and <answer> tags."
    )


def build_gemini_function_prompt(num_views, function_id):
    function_name, function_def = FUNCTION_MAP[function_id]
    return (
        GEMINI_SYSTEM_PREAMBLE.format(num_views=num_views) + "\n\n"
        f"CONTEXT: The child's communicative FUNCTION in this clip is:\n"
        f"  Class {function_id}: {function_name} — {function_def}\n\n"
        "For reference, the complete FUNCTION classification scheme is:\n"
        f"{FUNCTION_DEFS}\n\n"
        f"GUIDANCE FOR THIS LABEL: {FUNCTION_COGNITION_HINTS[function_id]}\n\n"
        "YOUR TASK: Produce a reasoning trace that explains why this function label fits "
        "the observable communicative intent in the video. Reference the child's apparent "
        "intent and the interaction context (caregiver behavior, objects, timing)."
        "Make the reasoning specific to "
        "this clip — do not write generic descriptions.\n\n"
        "OUTPUT FORMAT — output EXACTLY this structure, with NO text before or after:\n\n"
        "<think>\n"
        "[4-6 sentences of reasoning about communicative intent]\n"
        "</think>\n"
        f"<answer>Function: {function_id}</answer>\n\n"
        f'The <answer> tag must contain only "Function: {function_id}" with no other text. '
        "Do not write anything outside the <think> and <answer> tags."
    )


def build_gemini_joint_prompt(num_views, action_id, function_id):
    action_name, action_def = ACTION_MAP[action_id]
    function_name, function_def = FUNCTION_MAP[function_id]
    return (
        GEMINI_SYSTEM_PREAMBLE.format(num_views=num_views) + "\n\n"
        "CONTEXT: The child's communicative behavior in this clip is:\n"
        f"  ACTION:   class {action_id} — {action_name}: {action_def}\n"
        f"  FUNCTION: class {function_id} — {function_name}: {function_def}\n\n"
        "For reference, the complete classification schemes are:\n\n"
        f"{ACTION_DEFS}\n\n"
        f"{FUNCTION_DEFS}\n\n"
        f"ACTION GUIDANCE: {ACTION_COGNITION_HINTS[action_id]}\n\n"
        f"FUNCTION GUIDANCE: {FUNCTION_COGNITION_HINTS[function_id]}\n\n"
        "YOUR TASK: Produce an integrated reasoning trace that connects the observable "
        "physical behavior to BOTH the action label AND the function label. Describe what "
        "the child is physically doing AND what they appear to be communicating, and how "
        "the two relate. Make the reasoning specific to this clip — do not write generic descriptions.\n\n"
        "OUTPUT FORMAT — output EXACTLY this structure, with NO text before or after:\n\n"
        "<think>\n"
        "[5-8 sentences of integrated reasoning]\n"
        "</think>\n"
        f"<answer>Action: {action_id}, Function: {function_id}</answer>\n\n"
        f'The <answer> tag must contain only "Action: {action_id}, Function: {function_id}". '
        "Do not write anything outside the <think> and <answer> tags."
    )


_STUDENT_PREAMBLE_SINGLE = (
    "You are observing a short video clip of a non- or "
    "minimally-speaking child with profound autism and complex neurodevelopmental "
    "disorders interacting with their caregiver."
)

_STUDENT_PREAMBLE_MULTI = (
    "You are observing a short video clip (multiple camera angles) of a non- or "
    "minimally-speaking child with profound autism and complex neurodevelopmental "
    "disorders interacting with their caregiver."
)


def build_student_action_prompt(num_views):
    preamble = _STUDENT_PREAMBLE_SINGLE if num_views == 1 else _STUDENT_PREAMBLE_MULTI
    return (
        f"{preamble}\n\n"
        "Classify the child's communicative ACTION. Focus only on the child — "
        "ignore any adults present.\n\n"
        f"{ACTION_DEFS}\n\n"
        "First, reason about what you observe inside <think> tags. Then provide your "
        'answer inside <answer> tags in the format "Action: X" where X is a single '
        "digit 0-5.\n\n"
        "Example format:\n"
        "<think>\n[Your reasoning about what the child is physically doing]\n</think>\n"
        "<answer>Action: 3</answer>"
    )


def build_student_function_prompt(num_views):
    preamble = _STUDENT_PREAMBLE_SINGLE if num_views == 1 else _STUDENT_PREAMBLE_MULTI
    return (
        f"{preamble}\n\n"
        "Classify the child's communicative FUNCTION. Focus only on the child — "
        "ignore any adults present.\n\n"
        f"{FUNCTION_DEFS}\n\n"
        "First, reason about what you observe inside <think> tags. Then provide your "
        'answer inside <answer> tags in the format "Function: X" where X is a single '
        "digit 0-5.\n\n"
        "Example format:\n"
        "<think>\n[Your reasoning about why the child is communicating]\n</think>\n"
        "<answer>Function: 3</answer>"
    )


def build_student_joint_prompt(num_views):
    preamble = _STUDENT_PREAMBLE_SINGLE if num_views == 1 else _STUDENT_PREAMBLE_MULTI
    return (
        f"{preamble}\n\n"
        "Classify the child's communicative ACTION and FUNCTION. Focus only on "
        "the child — ignore any adults present.\n\n"
        f"{ACTION_DEFS}\n\n"
        f"{FUNCTION_DEFS}\n\n"
        "First, reason about what you observe inside <think> tags. Then provide your "
        'answer inside <answer> tags in the format "Action: X, Function: Y" where '
        "X and Y are single digits 0-5.\n\n"
        "Example format:\n"
        "<think>\n[Your reasoning about what the child is doing and why]\n</think>\n"
        "<answer>Action: 3, Function: 3</answer>"
    )


def is_valid_value(val):
    if pd.isna(val):
        return False
    if isinstance(val, str) and val.strip().upper() in ['N/A', 'NA', '']:
        return False
    return True


def get_labels(row):
    action_id = int(row['ACTION1_ID']) if is_valid_value(row.get('ACTION1_ID')) else None
    function_id = int(row['FUNCTION1_ID']) if is_valid_value(row.get('FUNCTION1_ID')) else None
    return action_id, function_id


def get_base_name(sample_name):
    stem = Path(sample_name).stem
    match = re.match(r'^(.+)_V\d+$', stem)
    if match:
        return match.group(1)
    match = re.match(r'^(TALK\w+?)_V\d+_(.+)$', stem)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return stem


def get_session(base_name):
    match = re.match(r'^(TALK\d+[a-z]?)', base_name)
    return match.group(1) if match else base_name.split('_')[0]


def group_multiview(df, video_col, input_dir):
    groups = defaultdict(lambda: {'rows': [], 'video_paths': []})

    for _, row in df.iterrows():
        base = get_base_name(row[video_col])
        groups[base]['rows'].append(row)
        vpath = Path(input_dir) / row[video_col]
        if vpath.exists():
            groups[base]['video_paths'].append(str(vpath))

    input_path = Path(input_dir)
    if input_path.exists():
        csv_files = {row[video_col] for _, row in df.iterrows()}
        for mp4 in sorted(input_path.glob('*.mp4')):
            if mp4.name not in csv_files:
                base = get_base_name(mp4.name)
                if base in groups and str(mp4) not in groups[base]['video_paths']:
                    groups[base]['video_paths'].append(str(mp4))

    for base in groups:
        groups[base]['video_paths'] = sorted(groups[base]['video_paths'])

    return dict(groups)


def load_raw_responses(path):
    entries = {}
    if not path or not Path(path).exists():
        return entries
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries[entry['base_name']] = entry
            except (json.JSONDecodeError, KeyError):
                continue
    return entries


def call_gemini(client, model_name, video_paths, prompt_text, fps=2, max_retries=5):
    parts = []
    for vp in video_paths:
        video_bytes = Path(vp).read_bytes()
        parts.append(types.Part(
            inline_data=types.Blob(data=video_bytes, mime_type='video/mp4'),
            video_metadata=types.VideoMetadata(fps=fps)
        ))
    parts.append(types.Part(text=prompt_text))
    content = types.Content(parts=parts)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=content,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="high")
                ),
            )
            if response.text is not None:
                return response.text.strip()
            else:
                reason = "Unknown"
                if hasattr(response, 'candidates') and response.candidates:
                    c = response.candidates[0]
                    if hasattr(c, 'finish_reason'):
                        reason = str(c.finish_reason)
                if 'PROHIBITED_CONTENT' in reason and attempt < max_retries:
                    print(f"      ⚠️ Attempt {attempt}: PROHIBITED_CONTENT, retrying...")
                    time.sleep(2)
                    continue
                print(f"      ⚠️ No text: {reason}")
                return None
        except Exception as e:
            print(f"      ❌ Attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(5)
            continue
    return None


def is_failed(text):
    if not text:
        return True
    return any(m in text for m in ['NO_TEXT', 'BLOCKED', 'Processing failed', 'ERROR:'])


def parse_think_content(text, min_len=30):
    if not text:
        return None
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if not m:
        return None
    content = m.group(1).strip()
    if len(content) < min_len:
        return None
    lowered = content.lower()
    if any(p in lowered for p in [
        "i cannot", "i'm sorry", "i am unable", "i'm unable",
        "i can't help", "as an ai", "i don't have"
    ]):
        return None
    return content


def build_assistant_target(think_content, action_id, function_id, task):
    if task == 'joint':
        answer = f"Action: {action_id}, Function: {function_id}"
    elif task == 'action':
        answer = f"Action: {action_id}"
    elif task == 'function':
        answer = f"Function: {function_id}"
    else:
        raise ValueError(f"Unknown task: {task}")
    return f"<think>\n{think_content}\n</think>\n<answer>{answer}</answer>"


def build_video_tags(video_paths):
    return "".join("<video>" for _ in video_paths)


def build_sft_sample(task, assistant_target, action_id, function_id, video_paths):
    num_views = len(video_paths)
    vtags = build_video_tags(video_paths)

    if task == 'action':
        prompt = build_student_action_prompt(num_views)
    elif task == 'function':
        prompt = build_student_function_prompt(num_views)
    elif task == 'joint':
        prompt = build_student_joint_prompt(num_views)
    else:
        raise ValueError(f"Unknown task: {task}")

    return {
        "task_type": TASK_TO_TYPE[task],
        "_action_id": action_id,
        "_function_id": function_id,
        "messages": [
            {"role": "user", "content": f"{vtags}{prompt}"},
            {"role": "assistant", "content": assistant_target},
        ],
        "videos": video_paths,
    }


def compute_sqrt_factors(class_counts):
    if not class_counts:
        return {}
    max_count = max(class_counts.values())
    return {
        cls: min(2, max(1, round(((max_count / cnt) ** 0.5) / 2)))
        for cls, cnt in class_counts.items()
    }


def oversample_samples(samples, task_type):
    if task_type == 'action_interpretation':
        keys = ['_action_id']
    elif task_type == 'function_interpretation':
        keys = ['_function_id']
    elif task_type == 'joint_interpretation':
        keys = ['_action_id', '_function_id']
    else:
        return samples, None

    counts_by_key = {k: defaultdict(int) for k in keys}
    for s in samples:
        for k in keys:
            v = s.get(k)
            if v is not None:
                counts_by_key[k][v] += 1

    factors_by_key = {k: compute_sqrt_factors(counts_by_key[k]) for k in keys}

    balanced = []
    for s in samples:
        repeats = []
        for k in keys:
            v = s.get(k)
            if v is not None:
                repeats.append(factors_by_key[k].get(v, 1))
        rep = max(repeats) if repeats else 1
        for _ in range(rep):
            balanced.append(s)

    stats = {
        'original_size': len(samples),
        'balanced_size': len(balanced),
        'counts': {k: dict(counts_by_key[k]) for k in keys},
        'factors': factors_by_key,
    }
    return balanced, stats


def write_jsonl(samples_list, path):
    with open(path, 'w', encoding='utf-8') as f:
        for s in samples_list:
            out = {k: v for k, v in s.items() if not k.startswith('_') and k != 'task_type'}
            json.dump(out, f, ensure_ascii=False)
            f.write('\n')


def main():
    parser = argparse.ArgumentParser(description='Stage 0: label-guided Gemini reasoning traces for SFT1.')
    parser.add_argument('--master-csv', '-g', required=True,
                        help='Annotation CSV, e.g. /path/to/annotations.csv')
    parser.add_argument('--input', '-i', required=True,
                        help='Video directory, e.g. /path/to/videos')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory, e.g. /path/to/sft_data')
    parser.add_argument('--model', '-m', default='models/gemini-3-flash-preview')
    parser.add_argument('--api-key', default='', help='Google API key (or set GOOGLE_API_KEY env var)')
    parser.add_argument('--fps', type=int, default=2)
    parser.add_argument('--max-retries', type=int, default=5)
    parser.add_argument('--rate-limit-delay', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true',
                        help='Only do split + audit. No Gemini calls, no SFT writing.')
    parser.add_argument('--skip-gemini', action='store_true',
                        help='Skip Gemini calls; build SFT files from existing raw responses only.')
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input)

    print("=" * 70)
    print("STEP 1: Load CSV and group multi-view clips")
    print("=" * 70)

    df = pd.read_csv(args.master_csv)
    video_col = None
    for col in ['sample_name', 'Video_Name', 'video_name', 'filename']:
        if col in df.columns:
            video_col = col
            break
    if not video_col:
        print(f"❌ No video column. Columns: {list(df.columns)}")
        return

    valid_mask = df['ACTION1_ID'].apply(is_valid_value) & df['FUNCTION1_ID'].apply(is_valid_value)
    df_valid = df[valid_mask].copy()
    print(f"  Total rows: {len(df)}, with valid labels: {len(df_valid)}")

    mv_groups = group_multiview(df_valid, video_col, input_dir)

    all_groups = {}
    for base, info in mv_groups.items():
        if not info['video_paths']:
            continue
        row = info['rows'][0]
        action_id, function_id = get_labels(row)
        if action_id is None or function_id is None:
            continue
        all_groups[base] = {**info, 'action_id': action_id, 'function_id': function_id}

    print(f"  Valid multi-view groups: {len(all_groups)}")

    view_counts = defaultdict(int)
    for base, g in all_groups.items():
        view_counts[len(g['video_paths'])] += 1
    for n, c in sorted(view_counts.items()):
        print(f"    {n}-view clips: {c}")

    session_bases = defaultdict(list)
    for base in all_groups:
        session_bases[get_session(base)].append(base)
    print(f"  Sessions: {len(session_bases)}")

    print("\n" + "=" * 70)
    print("STEP 2: Train/test split with session rules")
    print("=" * 70)

    base_to_split = {}
    split_rng = random.Random(args.seed)

    for session in sorted(session_bases.keys()):
        bases = sorted(session_bases[session])

        if session in FORCE_TEST_SESSIONS:
            for b in bases:
                base_to_split[b] = 'test'
            print(f"  {session}: {len(bases)} → ALL TEST (forced)")
            continue

        split_rng.shuffle(bases)
        n_test = max(1, round(len(bases) * 0.2))
        n_train = len(bases) - n_test

        for b in bases[:n_train]:
            base_to_split[b] = 'train'
        for b in bases[n_train:]:
            base_to_split[b] = 'test'

        print(f"  {session}: {n_train} train / {n_test} test (total {len(bases)})")

    n_train_total = sum(1 for v in base_to_split.values() if v == 'train')
    n_test_total = sum(1 for v in base_to_split.values() if v == 'test')
    print(f"\n  TOTAL: {n_train_total} train / {n_test_total} test")

    print("\n  Forced-test sanity check:")
    for session in sorted(FORCE_TEST_SESSIONS):
        if session in session_bases:
            train_leak = [b for b in session_bases[session] if base_to_split.get(b) == 'train']
            if train_leak:
                print(f"    ❌ {session}: {len(train_leak)} train leaks!")
            else:
                print(f"    ✅ {session}: all {len(session_bases[session])} in test")
        else:
            print(f"    ⚠️  {session}: not present in dataset")

    with open(output_dir / 'base_to_split.json', 'w') as f:
        json.dump(base_to_split, f, indent=2)
    print(f"  ✅ Saved base_to_split.json")

    split_rows = []
    for base, split in sorted(base_to_split.items()):
        session = get_session(base)
        group = all_groups[base]
        row = group['rows'][0]
        action_id, function_id = get_labels(row)
        n_views = len(group['video_paths'])
        split_rows.append({
            'base_name': base, 'session': session, 'split': split,
            'n_views': n_views,
            'ACTION1_ID': action_id, 'FUNCTION1_ID': function_id,
        })
    pd.DataFrame(split_rows).to_csv(output_dir / 'train_test_split.csv', index=False)
    print(f"  ✅ Saved train_test_split.csv")

    test_bases = {b for b, s in base_to_split.items() if s == 'test'}
    test_rows = []
    seen = set()
    for _, row in df_valid.iterrows():
        base = get_base_name(row[video_col])
        if base in test_bases and base not in seen:
            test_rows.append(row)
            seen.add(base)
    pd.DataFrame(test_rows).to_csv(output_dir / 'test_valid_action_function.csv', index=False)
    print(f"  ✅ Saved test_valid_action_function.csv")

    print("\n" + "=" * 70)
    print("STEP 3: Determine pending Gemini calls")
    print("=" * 70)

    raw_path = output_dir / 'gemini_raw_responses.jsonl'
    raw_entries = load_raw_responses(raw_path)
    print(f"  Loaded {len(raw_entries)} existing entries from {raw_path.name}")

    train_bases = {b for b, s in base_to_split.items() if s == 'train'}
    train_groups = {b: all_groups[b] for b in train_bases if b in all_groups}

    pending = []
    per_session_planned = defaultdict(set)
    for base_name in sorted(train_groups.keys()):
        session = get_session(base_name)
        allowed = SESSION_TASK_RESTRICTIONS.get(session, ALL_TASKS)
        per_session_planned[session].update(allowed)
        existing = raw_entries.get(base_name, {})
        existing_tasks = set(existing.get('responses', {}).keys()) if existing else set()
        for task in sorted(allowed):
            if task not in existing_tasks:
                pending.append((base_name, task))

    print(f"\n  Train clips: {len(train_bases)}")
    print(f"  Per-session task plan:")
    for session in sorted(per_session_planned):
        n_clips = sum(1 for b in train_bases if get_session(b) == session)
        tasks = sorted(per_session_planned[session])
        expected = n_clips * len(tasks)
        print(f"    {session}: {n_clips} clips × {tasks} = {expected} expected calls")
    print(f"\n  Pending Gemini calls: {len(pending)}")

    if args.dry_run:
        print("\n  🔍 DRY RUN — exiting before any API calls.")
        return

    if pending and not args.skip_gemini:
        print("\n" + "=" * 70)
        print(f"STEP 4: Run Gemini ({len(pending)} calls)")
        print("=" * 70)

        api_key = args.api_key or os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            print("  ❌ No API key. Set --api-key or GOOGLE_API_KEY.")
            return

        client = genai.Client(api_key=api_key)
        print("  ✅ Gemini client ready.\n")

        by_base = defaultdict(list)
        for base, task in pending:
            by_base[base].append(task)

        for idx, (base_name, tasks) in enumerate(sorted(by_base.items()), 1):
            group = train_groups[base_name]
            video_paths = group['video_paths']
            action_id = group['action_id']
            function_id = group['function_id']
            num_views = len(video_paths)

            existing = raw_entries.get(base_name, {})
            entry = {
                'base_name': base_name,
                'session': get_session(base_name),
                'action_id': action_id,
                'function_id': function_id,
                'video_paths': video_paths,
                'responses': dict(existing.get('responses', {})) if existing else {},
            }

            action_name = ACTION_MAP[action_id][0]
            function_name = FUNCTION_MAP[function_id][0]
            print(f"[{idx}/{len(by_base)}] 🎬 {base_name} ({num_views} views) "
                  f"| {action_name}/{function_name} | tasks={sorted(tasks)}")

            for task in sorted(tasks):
                if task == 'action':
                    prompt = build_gemini_action_prompt(num_views, action_id)
                elif task == 'function':
                    prompt = build_gemini_function_prompt(num_views, function_id)
                elif task == 'joint':
                    prompt = build_gemini_joint_prompt(num_views, action_id, function_id)
                else:
                    print(f"    ❌ Unknown task: {task}")
                    continue

                print(f"  📞 {task}...", end=" ", flush=True)
                resp = call_gemini(client, args.model, video_paths, prompt,
                                   fps=args.fps, max_retries=args.max_retries)

                if resp is None or is_failed(resp):
                    print("❌ no response")
                    time.sleep(args.rate_limit_delay)
                    continue

                think_content = parse_think_content(resp)
                if think_content is None:
                    print("❌ format invalid")
                    time.sleep(args.rate_limit_delay)
                    continue

                target = build_assistant_target(think_content, action_id, function_id, task)
                entry['responses'][task] = target
                print(f"✅ ({len(target)} chars)")

                time.sleep(args.rate_limit_delay)

            with open(raw_path, 'a', encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

            raw_entries[base_name] = entry
    elif args.skip_gemini:
        print("\n  ⏭  Skipping Gemini calls (--skip-gemini set)")
    else:
        print("\n  ✅ No pending Gemini calls — all train (clip, task) pairs already have responses.")

    compact_path = output_dir / 'gemini_raw_responses_compact.jsonl'
    train_entries = {b: raw_entries[b] for b in train_bases if b in raw_entries}
    with open(compact_path, 'w', encoding='utf-8') as f:
        for entry in train_entries.values():
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    print(f"\n  ✅ Compacted train JSONL: {len(train_entries)} entries → {compact_path.name}")

    print("\n" + "=" * 70)
    print("STEP 6: Build SFT samples")
    print("=" * 70)

    task_samples = defaultdict(list)
    for base_name, entry in train_entries.items():
        session = get_session(base_name)
        allowed = SESSION_TASK_RESTRICTIONS.get(session, ALL_TASKS)
        action_id = entry['action_id']
        function_id = entry['function_id']
        video_paths = entry['video_paths']
        responses = entry.get('responses', {})

        for task in sorted(allowed):
            target = responses.get(task)
            if not target or not target.strip():
                continue
            sample = build_sft_sample(task, target, action_id, function_id, video_paths)
            task_samples[sample['task_type']].append(sample)

    for task_type in sorted(task_samples.keys()):
        print(f"  {task_type}: {len(task_samples[task_type])} samples")

    print("\n" + "=" * 70)
    print("STEP 7: Oversample (sqrt-softened) and write JSONLs")
    print("=" * 70)

    all_unbalanced = []
    all_balanced = []

    for task_type in sorted(task_samples.keys()):
        samples = task_samples[task_type]

        path_orig = output_dir / f'sft_{task_type}.jsonl'
        write_jsonl(samples, path_orig)
        all_unbalanced.extend(samples)

        balanced, stats = oversample_samples(samples, task_type)
        path_bal = output_dir / f'sft_{task_type}_balanced.jsonl'
        write_jsonl(balanced, path_bal)
        all_balanced.extend(balanced)

        print(f"\n  {task_type}:")
        print(f"    Unbalanced: {stats['original_size']} → {path_orig.name}")
        print(f"    Balanced:   {stats['balanced_size']} → {path_bal.name}")
        for k, counts in stats['counts'].items():
            print(f"    {k} counts (before): {dict(sorted(counts.items()))}")
        for k, factors in stats['factors'].items():
            print(f"    {k} factors:        {dict(sorted(factors.items()))}")

    write_jsonl(all_unbalanced, output_dir / 'sft_all_combined.jsonl')
    print(f"\n  ✅ sft_all_combined.jsonl: {len(all_unbalanced)} (unbalanced)")

    random.shuffle(all_unbalanced)
    write_jsonl(all_unbalanced, output_dir / 'sft_all_shuffled.jsonl')
    print(f"  ✅ sft_all_shuffled.jsonl: {len(all_unbalanced)} (unbalanced + shuffled)")

    write_jsonl(all_balanced, output_dir / 'sft_all_balanced.jsonl')
    print(f"  ✅ sft_all_balanced.jsonl: {len(all_balanced)} (balanced)")

    random.shuffle(all_balanced)
    write_jsonl(all_balanced, output_dir / 'sft_all_balanced_shuffled.jsonl')
    print(f"  ✅ sft_all_balanced_shuffled.jsonl: {len(all_balanced)} (balanced + shuffled)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Forced-test sessions: {sorted(FORCE_TEST_SESSIONS)}")
    print(f"  Restricted sessions:")
    for s, allowed in sorted(SESSION_TASK_RESTRICTIONS.items()):
        print(f"    {s}: {sorted(allowed)}")
    print(f"  Split: {n_train_total} train / {n_test_total} test")
    print(f"  Train clips with usable responses: {len(train_entries)}")
    print(f"  Total SFT samples (unbalanced): {len(all_unbalanced)}")
    print(f"  Total SFT samples (balanced):   {len(all_balanced)}")
    print(f"  Output directory: {output_dir}")
    print(f"\n  Files:")
    for fpath in sorted(output_dir.glob('*')):
        if fpath.suffix in ('.jsonl', '.csv', '.json'):
            if fpath.suffix == '.jsonl':
                n = sum(1 for _ in open(fpath))
                print(f"    {fpath.name}: {n} lines")
            else:
                print(f"    {fpath.name}")

    final_pending = 0
    for base_name in sorted(train_groups.keys()):
        session = get_session(base_name)
        allowed = SESSION_TASK_RESTRICTIONS.get(session, ALL_TASKS)
        existing_tasks = set(raw_entries.get(base_name, {}).get('responses', {}).keys())
        for task in allowed:
            if task not in existing_tasks:
                final_pending += 1
    if final_pending > 0:
        print(f"\n  ⚠️  Still pending: {final_pending} (clip, task) pairs. Re-run to retry.")
    else:
        print(f"\n  ✅ All planned (clip, task) pairs are complete.")


if __name__ == "__main__":
    main()