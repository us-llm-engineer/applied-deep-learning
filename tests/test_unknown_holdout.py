from __future__ import annotations

import random

import pytest

from src.dataset import AUXILIARY_WORDS, holdout_unknown_words

COMMANDS = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"}
V2_VOCABULARY = set(
    "backward bed bird cat dog down eight five follow forward four go happy house learn left "
    "marvin nine no off on one right seven sheila six stop three tree two up visual wow yes zero".split()
)


def test_auxiliary_words_are_the_v2_vocabulary_minus_the_ten_commands() -> None:
    assert len(V2_VOCABULARY) == 35  # guards the fixture itself
    assert isinstance(AUXILIARY_WORDS, tuple)
    assert len(AUXILIARY_WORDS) == 25
    assert len(set(AUXILIARY_WORDS)) == 25
    assert set(AUXILIARY_WORDS) == V2_VOCABULARY - COMMANDS
    assert all(word == word.lower() for word in AUXILIARY_WORDS)


def test_holdout_partitions_the_words_into_disjoint_parts_of_the_requested_sizes() -> None:
    words = tuple(f"w{i:02d}" for i in range(12))
    train, heldout = holdout_unknown_words(words, 4, 0)
    assert isinstance(train, tuple) and isinstance(heldout, tuple)
    assert len(heldout) == 4 and len(train) == 8
    assert set(train).isdisjoint(heldout)
    assert set(train) | set(heldout) == set(words)
    assert len(set(train)) == len(train) and len(set(heldout)) == len(heldout)


def test_auxiliary_vocabulary_splits_into_twenty_train_and_five_heldout_words() -> None:
    train, heldout = holdout_unknown_words(AUXILIARY_WORDS, 5, 17)
    assert (len(train), len(heldout)) == (20, 5)
    assert set(train) | set(heldout) == set(AUXILIARY_WORDS)
    assert set(train).isdisjoint(heldout)


def test_holdout_is_deterministic_for_a_seed() -> None:
    assert holdout_unknown_words(AUXILIARY_WORDS, 5, 3) == holdout_unknown_words(AUXILIARY_WORDS, 5, 3)


def test_holdout_choice_varies_with_the_seed() -> None:
    heldout_sets = {frozenset(holdout_unknown_words(AUXILIARY_WORDS, 5, seed)[1]) for seed in range(8)}
    assert len(heldout_sets) >= 4, "eight seeds should not collapse onto <4 distinct held-out sets"


def test_every_word_can_be_held_out_so_the_selection_is_not_a_fixed_prefix() -> None:
    ever_heldout: set[str] = set()
    for seed in range(200):
        ever_heldout |= set(holdout_unknown_words(AUXILIARY_WORDS, 5, seed)[1])
    assert ever_heldout == set(AUXILIARY_WORDS)


def test_holdout_result_does_not_depend_on_the_order_of_the_input_words() -> None:
    reference = holdout_unknown_words(AUXILIARY_WORDS, 5, 11)
    shuffled = list(AUXILIARY_WORDS)
    random.Random(0).shuffle(shuffled)
    assert shuffled != list(AUXILIARY_WORDS)
    assert holdout_unknown_words(shuffled, 5, 11) == reference
    assert holdout_unknown_words(list(reversed(AUXILIARY_WORDS)), 5, 11) == reference


def test_holdout_does_not_mutate_a_list_input() -> None:
    words = list(AUXILIARY_WORDS)
    holdout_unknown_words(words, 5, 1)
    assert words == list(AUXILIARY_WORDS)


@pytest.mark.parametrize("n_holdout", [0, -1, 25, 26])
def test_holdout_rejects_a_count_that_leaves_either_side_empty_or_is_negative(n_holdout: int) -> None:
    with pytest.raises(ValueError):
        holdout_unknown_words(AUXILIARY_WORDS, n_holdout, 0)


def test_holdout_accepts_the_extreme_legal_counts_one_and_len_minus_one() -> None:
    for n_holdout in (1, 24):
        train, heldout = holdout_unknown_words(AUXILIARY_WORDS, n_holdout, 0)
        assert len(heldout) == n_holdout and len(train) == 25 - n_holdout


def test_holdout_rejects_duplicate_words() -> None:
    words = ["cat", "dog", "bed", "cat", "bird"]
    with pytest.raises(ValueError):
        holdout_unknown_words(words, 2, 0)
