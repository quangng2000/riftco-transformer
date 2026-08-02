from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "python"))
sys.path.insert(0, str(PROJECT_ROOT))

from labs.conditional_reverse.config import TrainingConfig  # noqa: E402
from labs.conditional_reverse.data import TokenCodec, make_batch  # noqa: E402
from labs.conditional_reverse.evaluation import EvaluationMetrics  # noqa: E402
from labs.conditional_reverse.protocol import (  # noqa: E402
    ProtocolConfig,
    make_example,
)
from labs.conditional_reverse.training import (  # noqa: E402
    target_half_loss,
    train_model,
)
import labs.conditional_reverse.training as training_module  # noqa: E402


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        return None


class _FakeLoss(_Context):
    def __init__(self, events: list[str], value: float = 0.5) -> None:
        self.events = events
        self.value = value

    def item(self) -> float:
        return self.value

    def backward(self) -> None:
        self.events.append("backward")


class _FakeLogits(_Context):
    pass


class _FakeModel:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def parameters(self):
        return _Context()

    def forward(self, rows):
        self.events.append("forward")
        return SimpleNamespace(logits=_FakeLogits())


class _FakeAdam(_Context):
    events: list[str] = []

    def __init__(self, _parameters, **_options) -> None:
        self.backend = "cpu"
        self.step_count = 0

    def zero_grad(self) -> None:
        self.events.append("zero_grad")

    def step(self):
        self.events.append("step")
        self.step_count += 1
        return SimpleNamespace(
            step=self.step_count,
            gradient_norm=1.0,
            clip_scale=1.0,
        )


def _metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        loss=0.25,
        example_count=1,
        target_token_count=3,
        correct_target_token_count=1,
        correct_sequence_count=0,
        target_token_accuracy=1 / 3,
        exact_sequence_accuracy=0.0,
        reverse_example_count=1,
        reverse_target_token_accuracy=1 / 3,
        reverse_exact_sequence_accuracy=0.0,
        copy_example_count=0,
        copy_target_token_accuracy=0.0,
        copy_exact_sequence_accuracy=0.0,
    )


class ConditionalReverseTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = ProtocolConfig(
            sequence_length=3,
            alphabet="abcd",
            reverse_when_first_is="a",
        )
        self.codec = TokenCodec.from_protocol(self.protocol)

    def test_target_loss_selects_exactly_l_to_2l(self) -> None:
        batch = make_batch(
            (make_example("abc", self.protocol),),
            self.codec,
        )
        sentinel = object()
        with patch.object(
            training_module,
            "cross_entropy_time_range",
            return_value=sentinel,
        ) as loss:
            self.assertIs(target_half_loss(object(), batch), sentinel)
        self.assertEqual(loss.call_args.args[1], batch.target_rows)
        self.assertEqual(loss.call_args.args[2:], (3, 3))

    def test_two_steps_zero_grad_before_each_forward_and_backward(self) -> None:
        events: list[str] = []
        _FakeAdam.events = events
        runtime = SimpleNamespace(
            model=_FakeModel(events),
            backend="cpu",
        )
        training = (
            make_example("abc", self.protocol),
            make_example("bca", self.protocol),
        )
        validation = (make_example("cab", self.protocol),)
        config = TrainingConfig(
            epochs=1,
            batch_size=1,
            evaluation_batch_size=1,
            maximum_steps=2,
        )
        fake_evaluation = SimpleNamespace(metrics=_metrics())

        def fake_loss(_logits, _batch):
            return _FakeLoss(events)

        with patch.object(training_module, "Adam", _FakeAdam), patch.object(
            training_module,
            "target_half_loss",
            side_effect=fake_loss,
        ), patch.object(
            training_module,
            "evaluate_model",
            return_value=fake_evaluation,
        ):
            history = train_model(
                runtime,
                training,
                validation,
                self.codec,
                config,
            )
        self.assertEqual(
            events,
            [
                "zero_grad",
                "forward",
                "backward",
                "step",
                "zero_grad",
                "forward",
                "backward",
                "step",
            ],
        )
        self.assertEqual(history.final_step, 2)
        self.assertEqual(len(history.steps), 2)
        self.assertEqual(len(history.epochs), 1)


if __name__ == "__main__":
    unittest.main()
