import os
import re
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from swift.rewards import ORM, orms

__all__ = ['MultiLabelAccuracyORM']

ACTION_WEIGHTS: Dict[str, float] = {
    '0': 0.7413,
    '1': 0.5314,
    '2': 1.0000,
    '3': 0.3004,
    '4': 0.7771,
    '5': 0.3075,
}

FUNCTION_WEIGHTS: Dict[str, float] = {
    '0': 0.9307,
    '1': 0.4626,
    '2': 0.6886,
    '3': 0.4111,
    '4': 0.7906,
    '5': 1.0000,
}

DIFFICULTY_MULTIPLIERS: Dict[Optional[int], float] = {
    1: 1.0,
    2: 1.2,
    3: 1.4,
    None: 1.7,
}


def _extract_answer_block(completion: str) -> Optional[str]:
    m = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
    return m.group(1).strip() if m else None


def _parse_axis(text: str, axis: str) -> Optional[str]:
    m = re.search(rf'{axis}\s*:\s*(\d)', text, re.IGNORECASE)
    if not m:
        return None
    digit = m.group(1)
    return digit if digit in {'0', '1', '2', '3', '4', '5'} else None


def _parse_solution_axis(solution: str, axis: str) -> Optional[List[str]]:
    m = re.search(rf'{axis}\s*:\s*([\d|]+)', solution, re.IGNORECASE)
    if not m:
        return None
    labels = [d for d in m.group(1).split('|') if d in {'0', '1', '2', '3', '4', '5'}]
    return labels or None


def _detect_task_from_solution(solution: str) -> str:
    has_a = 'Action:' in solution
    has_f = 'Function:' in solution
    if has_a and has_f:
        return 'joint'
    return 'action' if has_a else 'function'


class MultiLabelAccuracyORM(ORM):

    def __init__(self, args=None, **kwargs):
        super().__init__(args=args, **kwargs)

        self.difficulty_map: Dict[str, Dict[str, Any]] = {}
        difficulty_file = os.environ.get('DIFFICULTY_FILE')

        if not difficulty_file:
            print("[ROSCO Reward] DIFFICULTY_FILE not set; using 1.0x difficulty multipliers.")
        elif not Path(difficulty_file).exists():
            raise FileNotFoundError(f"[ROSCO Reward] DIFFICULTY_FILE not found: {difficulty_file}")
        else:
            with open(difficulty_file, 'r', encoding='utf-8') as f:
                self.difficulty_map = json.load(f)
            print(f"[ROSCO Reward] Loaded {len(self.difficulty_map)} difficulty entries from {difficulty_file}.")

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []

        for idx, (completion, sol) in enumerate(zip(completions, solution)):
            base_reward = self._score_one(completion, sol)

            if base_reward <= 0.0:
                rewards.append(0.0)
                continue

            task_type = _detect_task_from_solution(sol)
            base_name = None

            if 'videos' in kwargs and idx < len(kwargs['videos']):
                v_entry = kwargs['videos'][idx]
                if isinstance(v_entry, list) and len(v_entry) > 0:
                    base_name = Path(v_entry[0]).stem
                    base_name = re.sub(r'_V\d+$', '', base_name)

            lookup_key = f"{base_name}|{task_type}" if base_name else None
            multiplier = 1.0

            if lookup_key and lookup_key in self.difficulty_map:
                meta = self.difficulty_map[lookup_key]
                stage = meta.get('rescue_stage')

                if meta.get('used_gemini_fallback', False) or stage is None:
                    multiplier = DIFFICULTY_MULTIPLIERS[None]
                else:
                    multiplier = DIFFICULTY_MULTIPLIERS.get(int(stage), 1.0)

            rewards.append(base_reward * multiplier)

        return rewards

    def _score_one(self, completion: str, solution: str) -> float:
        predicted = _extract_answer_block(completion)
        if predicted is None:
            return 0.0

        sol_action_labels = _parse_solution_axis(solution, 'Action')
        sol_function_labels = _parse_solution_axis(solution, 'Function')

        if sol_action_labels is not None and sol_function_labels is not None:
            return self._score_joint(predicted, sol_action_labels, sol_function_labels)
        elif sol_action_labels is not None:
            return self._score_axis(predicted, 'Action', sol_action_labels, ACTION_WEIGHTS)
        elif sol_function_labels is not None:
            return self._score_axis(predicted, 'Function', sol_function_labels, FUNCTION_WEIGHTS)
        else:
            return 0.0

    @staticmethod
    def _score_axis(predicted: str, axis: str, valid_labels: List[str],
                    weights: Dict[str, float]) -> float:
        pred_digit = _parse_axis(predicted, axis)
        if pred_digit is None:
            return 0.0
        if pred_digit in valid_labels:
            return weights.get(pred_digit, 1.0)
        return 0.0

    @staticmethod
    def _score_joint(predicted: str, valid_actions: List[str],
                     valid_functions: List[str]) -> float:
        reward = 0.0
        pred_a = _parse_axis(predicted, 'Action')
        pred_f = _parse_axis(predicted, 'Function')

        if pred_a is not None and pred_a in valid_actions:
            a_primary = valid_actions[0]
            reward += 0.5 * ACTION_WEIGHTS.get(a_primary, 1.0)

        if pred_f is not None and pred_f in valid_functions:
            f_primary = valid_functions[0]
            reward += 0.5 * FUNCTION_WEIGHTS.get(f_primary, 1.0)

        return reward


orms['external_rosco_accuracy'] = MultiLabelAccuracyORM
