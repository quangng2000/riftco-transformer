from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from labs.conditional_reverse.data import (  # noqa: E402
    TokenCodec,
    encode_example,
    iter_batches,
    make_batch,
)
from labs.conditional_reverse.protocol import (  # noqa: E402
    ProtocolConfig,
    make_example,
)


class ConditionalReverseDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ProtocolConfig(
            sequence_length=3,
            alphabet="abcd",
            reverse_when_first_is="a",
            delimiter="|",
        )
        self.codec = TokenCodec.from_protocol(self.config)

    def test_codec_uses_alphabet_order_and_delimiter_last(self) -> None:
        self.assertEqual(self.codec.vocabulary, ("a", "b", "c", "d", "|"))
        self.assertEqual(self.codec.vocabulary_size, 5)
        self.assertEqual(self.codec.delimiter_id, 4)
        encoded = self.codec.encode("dc|a")
        self.assertEqual(encoded, (3, 2, 4, 0))
        self.assertEqual(self.codec.decode(encoded), "dc|a")
        with self.assertRaisesRegex(ValueError, "outside the vocabulary"):
            self.codec.encode("z")

    def test_teacher_forcing_and_supervised_half_are_exact(self) -> None:
        encoded = encode_example(make_example("abc", self.config), self.codec)
        self.assertEqual(encoded.source_ids, (0, 1, 2))
        self.assertEqual(encoded.output_ids, (2, 1, 0))
        self.assertEqual(encoded.input_ids, (0, 1, 2, 4, 2, 1))
        self.assertEqual(encoded.target_ids, (1, 2, 4, 2, 1, 0))
        self.assertEqual(encoded.supervised_time_range, (3, 6))
        self.assertEqual(encoded.target_ids[3:6], encoded.output_ids)

    def test_batch_is_flattened_row_major_and_retains_pairing(self) -> None:
        examples = (
            make_example("abc", self.config),
            make_example("bcd", self.config),
        )
        batch = make_batch(examples, self.codec, example_indices=(7, 11))
        self.assertEqual(batch.example_indices, (7, 11))
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.sequence_length, 3)
        self.assertEqual(batch.context_length, 6)
        self.assertEqual(batch.supervised_time_range, (3, 6))
        self.assertEqual(batch.input_ids[:6], (0, 1, 2, 4, 2, 1))
        self.assertEqual(batch.output_ids[:3], (2, 1, 0))
        self.assertEqual(batch.reversed, (True, False))
        self.assertEqual(batch.input_rows[0], (0, 1, 2, 4, 2, 1))
        self.assertEqual(batch.target_rows[0], (1, 2, 4, 2, 1, 0))
        self.assertEqual(batch.output_rows[0], (2, 1, 0))

    def test_seeded_epoch_batching_is_deterministic(self) -> None:
        sources = (
            "abc",
            "abd",
            "acb",
            "acd",
            "adb",
            "adc",
            "bca",
            "bcd",
            "cba",
            "cbd",
            "dba",
            "dbc",
        )
        examples = tuple(make_example(source, self.config) for source in sources)

        def order(epoch: int) -> tuple[int, ...]:
            return tuple(
                index
                for batch in iter_batches(
                    examples,
                    self.codec,
                    5,
                    shuffle=True,
                    seed=91,
                    epoch=epoch,
                )
                for index in batch.example_indices
            )

        self.assertEqual(order(3), order(3))
        self.assertNotEqual(order(3), order(4))
        self.assertEqual(set(order(3)), set(range(len(examples))))

    def test_unshuffled_drop_last_keeps_global_indices(self) -> None:
        examples = tuple(
            make_example(source, self.config)
            for source in ("abc", "bca", "cab", "dba", "bcd")
        )
        batches = tuple(
            iter_batches(
                examples,
                self.codec,
                2,
                shuffle=False,
                drop_last=True,
            )
        )
        self.assertEqual(
            tuple(batch.example_indices for batch in batches),
            ((0, 1), (2, 3)),
        )
        self.assertEqual(tuple(iter_batches((), self.codec, 2)), ())

    def test_batch_validation_rejects_index_mismatch(self) -> None:
        example = make_example("abc", self.config)
        with self.assertRaisesRegex(ValueError, "example_indices"):
            make_batch((example,), self.codec, example_indices=(0, 1))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            make_batch((), self.codec)


if __name__ == "__main__":
    unittest.main()
