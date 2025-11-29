import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

from random import choice, randint, sample, shuffle, uniform

import re
from dataclasses import dataclass
from typing import Optional, Set, Dict

import torch

from dsl import *


global rng
rng = []


def unifint(
    diff_lb: float,
    diff_ub: float,
    bounds: Tuple[int, int]
) -> int:
    """
    diff_lb: lower bound for difficulty, must be in range [0, diff_ub]
    diff_ub: upper bound for difficulty, must be in range [diff_lb, 1]
    bounds: interval [a, b] determining the integer values that can be sampled
    """
    a, b = bounds
    d = uniform(diff_lb, diff_ub)
    global rng
    rng.append(d)
    return min(max(a, round(a + (b - a) * d)), b)


def is_grid(
    grid: Any
) -> bool:
    """
    returns True if and only if argument is a valid grid
    """
    if not isinstance(grid, tuple):
        return False
    if not 0 < len(grid) <= 30:
        return False
    if not all(isinstance(r, tuple) for r in grid):
        return False
    if not all(0 < len(r) <= 30 for r in grid):
        return False
    if not len(set(len(r) for r in grid)) == 1:
        return False
    if not all(all(isinstance(x, int) for x in r) for r in grid):
        return False
    if not all(all(0 <= x <= 9 for x in r) for r in grid):
        return False
    return True


def strip_prefix(
    string: str,
    prefix: str
) -> str:
    """
    removes prefix
    """
    return string[len(prefix):]


def format_grid(
    grid: List[List[int]]
) -> Grid:
    """
    grid type casting
    """
    return tuple(tuple(row) for row in grid)


def format_example(
    example: dict
) -> dict:
    """
    example data type
    """
    return {
        'input': format_grid(example['input']),
        'output': format_grid(example['output'])
    }


def format_task(
    task: dict
) -> dict:
    """
    task data type
    """
    return {
        'train': [format_example(example) for example in task['train']],
        'test': [format_example(example) for example in task['test']]
    }


def plot_task(
    task: List[dict],
    title: str = None
) -> None:
    """
    displays a task
    """
    cmap = ListedColormap([
        '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = Normalize(vmin=0, vmax=9)
    args = {'cmap': cmap, 'norm': norm}
    height = 2
    width = len(task)
    figure_size = (width * 3, height * 3)
    figure, axes = plt.subplots(height, width, figsize=figure_size)
    for column, example in enumerate(task):
        axes[0, column].imshow(example['input'], **args)
        axes[1, column].imshow(example['output'], **args)
        axes[0, column].axis('off')
        axes[1, column].axis('off')
    if title is not None:
        figure.suptitle(title, fontsize=20)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def fix_bugs(
    dataset: dict
) -> None:
    """
    fixes bugs in the original ARC training dataset
    """
    dataset['a8d7556c']['train'][2]['output'] = fill(dataset['a8d7556c']['train'][2]['output'], 2, {(8, 12), (9, 12)})
    dataset['6cf79266']['train'][2]['output'] = fill(dataset['6cf79266']['train'][2]['output'], 1, {(6, 17), (7, 17), (8, 15), (8, 16), (8, 17)})
    dataset['469497ad']['train'][1]['output'] = fill(dataset['469497ad']['train'][1]['output'], 7, {(5, 12), (5, 13), (5, 14)})
    dataset['9edfc990']['train'][1]['output'] = fill(dataset['9edfc990']['train'][1]['output'], 1, {(6, 13)})
    dataset['e5062a87']['train'][1]['output'] = fill(dataset['e5062a87']['train'][1]['output'], 2, {(1, 3), (1, 4), (1, 5), (1, 6)})
    dataset['e5062a87']['train'][0]['output'] = fill(dataset['e5062a87']['train'][0]['output'], 2, {(5, 2), (6, 3), (3, 6), (4, 7)})


@dataclass
class DSLConstraintState:
    """Tracks high-level syntax state while decoding."""
    config: "DSLConstrainedDecoder"
    line_start: bool = True
    expect_assign: bool = False
    need_value: bool = False
    paren_depth: int = 0
    last_token_type: Optional[str] = None
    seen_content: bool = False

    def copy(self) -> "DSLConstraintState":
        return DSLConstraintState(
            config=self.config,
            line_start=self.line_start,
            expect_assign=self.expect_assign,
            need_value=self.need_value,
            paren_depth=self.paren_depth,
            last_token_type=self.last_token_type,
            seen_content=self.seen_content
        )

    def update(self, token_id: int) -> None:
        token_type = self.config.token_types.get(token_id, "OTHER")

        if token_id == self.config.newline_id:
            self.line_start = True
            self.expect_assign = False
            self.need_value = False
            self.last_token_type = "NEWLINE"
            return

        if token_id == self.config.eos_id:
            self.last_token_type = "EOS"
            return

        # Handle tokens at line start (assignment targets or return)
        if self.line_start:
            if token_id == self.config.return_id:
                self.need_value = True
            elif token_type == "VARIABLE":
                self.expect_assign = True
            self.line_start = False

        if token_id == self.config.eq_id:
            self.expect_assign = False
            self.need_value = True
        elif token_id == self.config.comma_id:
            self.need_value = True
        elif token_id == self.config.lparen_id:
            self.paren_depth += 1
            self.need_value = True
        elif token_id == self.config.rparen_id and self.paren_depth > 0:
            self.paren_depth -= 1
            self.need_value = False
        elif token_id == self.config.return_id:
            self.need_value = True
        elif token_type in {"IDENTIFIER", "VARIABLE", "NUMBER"}:
            self.need_value = False

        if token_type not in {"NEWLINE", "EOS"}:
            self.seen_content = True

        if token_type != "OTHER":
            self.last_token_type = token_type

    def can_end_expression(self) -> bool:
        if self.need_value:
            return False
        return self.last_token_type in {"IDENTIFIER", "VARIABLE", "NUMBER", "RPAREN"}

    def can_terminate(self) -> bool:
        return self.paren_depth == 0 and not self.need_value and self.seen_content

    def allows_lparen(self) -> bool:
        return self.last_token_type in {"IDENTIFIER", "VARIABLE", "RPAREN"}


class DSLConstrainedDecoder:
    """
    Provides lightweight grammar-aware token masking for DSL programs.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        self.variable_pattern = re.compile(r'^x\d+$')
        self.identifier_pattern = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
        self.number_pattern = re.compile(r'^\d+$')

        self.newline_id = tokenizer.newline_token_id
        self.eos_id = tokenizer.eos_token_id
        self.bos_id = tokenizer.bos_token_id
        self.pad_id = tokenizer.pad_token_id
        self.sep_id = tokenizer.sep_token_id
        self.row_id = getattr(tokenizer, 'row_token_id', None)

        token_to_id = tokenizer.token_to_id
        self.eq_id = token_to_id.get('=')
        self.comma_id = token_to_id.get(',')
        self.lparen_id = token_to_id.get('(')
        self.rparen_id = token_to_id.get(')')
        self.return_id = token_to_id.get('return')

        self.variable_ids: Set[int] = set()
        self.identifier_ids: Set[int] = set()
        self.number_ids: Set[int] = set()
        self.token_types: Dict[int, str] = {}

        for token, idx in token_to_id.items():
            if token in {'[PAD]', '[BOS]', '[EOS]', '[SEP]', '[ROW]', '[NEWLINE]'}:
                continue
            if self.variable_pattern.fullmatch(token):
                self.variable_ids.add(idx)
                self.identifier_ids.add(idx)
                self.token_types[idx] = "VARIABLE"
            elif self.number_pattern.fullmatch(token):
                self.number_ids.add(idx)
                self.token_types[idx] = "NUMBER"
            elif self.identifier_pattern.fullmatch(token):
                self.identifier_ids.add(idx)
                self.token_types[idx] = "IDENTIFIER"

        # Special overrides
        if self.return_id is not None:
            self.token_types[self.return_id] = "RETURN"
            self.identifier_ids.discard(self.return_id)

        if self.eq_id is not None:
            self.token_types[self.eq_id] = "ASSIGN"
        if self.comma_id is not None:
            self.token_types[self.comma_id] = "COMMA"
        if self.lparen_id is not None:
            self.token_types[self.lparen_id] = "LPAREN"
        if self.rparen_id is not None:
            self.token_types[self.rparen_id] = "RPAREN"
        self.token_types[self.newline_id] = "NEWLINE"
        self.token_types[self.eos_id] = "EOS"

        self.value_ids = sorted(self.identifier_ids | self.number_ids | self.variable_ids)

        self.base_allowed = torch.ones(self.vocab_size, dtype=torch.bool)
        for tok_id in [self.pad_id, self.bos_id, self.sep_id, self.row_id]:
            if tok_id is not None and tok_id < self.vocab_size:
                self.base_allowed[tok_id] = False

        # Cache for device-local base_allowed to avoid repeated CPU->GPU transfers
        self._base_allowed_cache = {}

    def new_state(self) -> DSLConstraintState:
        return DSLConstraintState(config=self)

    def build_mask(self, state: DSLConstraintState, device) -> torch.Tensor:
        # Use cached device-local tensor to avoid expensive CPU->GPU transfers
        device_key = str(device)
        if device_key not in self._base_allowed_cache:
            self._base_allowed_cache[device_key] = self.base_allowed.to(device)
        mask = self._base_allowed_cache[device_key].clone()

        # Line start: only assignments or return (plus EOS if program already complete)
        if state.line_start:
            allowed = []
            allowed.extend(self.variable_ids)
            if self.return_id is not None:
                allowed.append(self.return_id)
            if state.can_terminate():
                allowed.append(self.eos_id)
            mask[:] = False
            if allowed:
                mask[allowed] = True
            return mask

        # Expecting assignment token
        if state.expect_assign and self.eq_id is not None:
            mask[:] = False
            mask[self.eq_id] = True
            return mask

        # Expecting start of an expression/value
        if state.need_value:
            mask[:] = False
            if self.value_ids:
                mask[self.value_ids] = True
            if self.lparen_id is not None:
                mask[self.lparen_id] = True
            return mask

        # General adjustments
        if self.eq_id is not None:
            mask[self.eq_id] = False

        if self.lparen_id is not None and not state.allows_lparen():
            mask[self.lparen_id] = False

        if self.comma_id is not None:
            allow_comma = (state.paren_depth > 0 and state.can_end_expression())
            if not allow_comma:
                mask[self.comma_id] = False

        if self.rparen_id is not None:
            allow_rparen = (state.paren_depth > 0 and state.can_end_expression())
            if not allow_rparen:
                mask[self.rparen_id] = False

        if self.newline_id is not None:
            allow_newline = (state.paren_depth == 0 and state.can_end_expression())
            if not allow_newline:
                mask[self.newline_id] = False

        if self.return_id is not None:
            mask[self.return_id] = False  # return only allowed at line start

        allow_eos = state.can_terminate()
        if not allow_eos:
            mask[self.eos_id] = False

        if not mask.any():
            # Fallback to base to avoid NaNs if constraints go empty
            print("WARNING: Empty mask fallback triggered!")
            print(f"  State: line_start={state.line_start}, expect_assign={state.expect_assign}, "
                  f"need_value={state.need_value}, paren_depth={state.paren_depth}, "
                  f"last_token_type={state.last_token_type}, seen_content={state.seen_content}")
            # Use cached device-local tensor
            device_key = str(device)
            if device_key not in self._base_allowed_cache:
                self._base_allowed_cache[device_key] = self.base_allowed.to(device)
            mask = self._base_allowed_cache[device_key].clone()

        return mask

    def apply(self, logits: torch.Tensor, state: DSLConstraintState) -> torch.Tensor:
        mask = self.build_mask(state, logits.device)
        if logits.dim() > 1:
            view_shape = [1] * (logits.dim() - 1) + [self.vocab_size]
            mask = mask.view(*view_shape).expand_as(logits)
        return logits.masked_fill(~mask, float('-inf'))
