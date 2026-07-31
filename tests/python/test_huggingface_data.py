from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request

from transformer_lab._atomic_publish import publish_directory_no_replace
from transformer_lab.data import (
    DOLLY_15K,
    HH_RLHF,
    TINY_STORIES,
    DatasetRow,
    DatasetSchemaError,
    DatasetTransportError,
    DollyInstructionAdapter,
    HhRlhfPreferenceAdapter,
    HttpResponse,
    HuggingFaceDatasetClient,
    SplitFractions,
    StableHashSplitter,
    TinyStoriesTextAdapter,
    UrllibHttpTransport,
    prepare_huggingface_dataset,
    verify_prepared_dataset,
)
from transformer_lab.data.client import _RejectRedirectHandler


class FakeDatasetApiTransport:
    def __init__(
        self,
        dataset: str,
        rows: list[dict[str, object]],
        *,
        config: str = "default",
        split: str = "train",
        catalog_splits: tuple[str, ...] | None = None,
        revision: str = "abc123",
        failures: int = 0,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.split = split
        self.catalog_splits = (
            (split,) if catalog_splits is None else catalog_splits
        )
        self.rows = rows
        self.revision = revision
        self.failures = failures
        self.calls: list[tuple[str, dict[str, str], float]] = []
        self.rows_calls = 0
        self.revision_by_rows_call: dict[int, str] = {}

    def get(
        self,
        url: str,
        *,
        headers: object,
        timeout: float,
    ) -> HttpResponse:
        captured_headers = dict(headers)  # type: ignore[arg-type]
        self.calls.append((url, captured_headers, timeout))
        if self.failures > 0:
            self.failures -= 1
            raise URLError("temporary failure")
        parsed = urlsplit(url)
        query = parse_qs(parsed.query)
        if parsed.path == "/splits":
            body = {
                "splits": [
                    {
                        "dataset": self.dataset,
                        "config": self.config,
                        "split": split,
                    }
                    for split in self.catalog_splits
                ],
                "pending": [],
                "failed": [],
            }
        elif parsed.path == "/rows":
            self.rows_calls += 1
            offset = int(query["offset"][0])
            length = int(query["length"][0])
            page_rows = self.rows[offset : offset + length]
            feature_names = (
                tuple(self.rows[0])
                if self.rows
                else ("text",)
            )
            body = {
                "features": [
                    {
                        "feature_idx": index,
                        "name": name,
                        "type": {"dtype": "string", "_type": "Value"},
                    }
                    for index, name in enumerate(feature_names)
                ],
                "rows": [
                    {
                        "row_idx": offset + position,
                        "row": row,
                        "truncated_cells": [],
                    }
                    for position, row in enumerate(page_rows)
                ],
                "num_rows_total": len(self.rows),
                "num_rows_per_page": 100,
                "partial": False,
            }
        else:
            raise AssertionError(f"unexpected path: {parsed.path}")
        revision = self.revision_by_rows_call.get(
            self.rows_calls,
            self.revision,
        )
        return HttpResponse(
            status=200,
            body=json.dumps(body).encode("utf-8"),
            headers={"X-Revision": revision},
        )


class StaticTransport:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response

    def get(
        self,
        _url: str,
        *,
        headers: object,
        timeout: float,
    ) -> HttpResponse:
        del headers, timeout
        return self.response


class HuggingFaceClientTests(unittest.TestCase):
    def test_urllib_transport_never_follows_cross_origin_redirects(self) -> None:
        source_authorizations: list[str | None] = []
        target_authorizations: list[str | None] = []

        class TargetHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                target_authorizations.append(
                    self.headers.get("Authorization")
                )
                self.send_response(200)
                self.end_headers()

            def log_message(self, _format: str, *args: object) -> None:
                del args

        target_server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            TargetHandler,
        )
        target_url = (
            "http://127.0.0.1:"
            f"{target_server.server_address[1]}/capture"
        )

        class RedirectHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                source_authorizations.append(
                    self.headers.get("Authorization")
                )
                self.send_response(302)
                self.send_header("Location", target_url)
                self.end_headers()

            def log_message(self, _format: str, *args: object) -> None:
                del args

        source_server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            RedirectHandler,
        )
        threads = (
            Thread(target=target_server.serve_forever, daemon=True),
            Thread(target=source_server.serve_forever, daemon=True),
        )
        for thread in threads:
            thread.start()
        try:
            source_url = (
                "http://127.0.0.1:"
                f"{source_server.server_address[1]}/redirect"
            )
            with self.assertRaisesRegex(
                DatasetTransportError,
                "redirects are not allowed",
            ):
                UrllibHttpTransport().get(
                    source_url,
                    headers={"Authorization": "Bearer test-secret"},
                    timeout=2.0,
                )
        finally:
            source_server.shutdown()
            target_server.shutdown()
            source_server.server_close()
            target_server.server_close()
            for thread in threads:
                thread.join(timeout=2.0)

        self.assertEqual(source_authorizations, ["Bearer test-secret"])
        self.assertEqual(target_authorizations, [])

    def test_https_to_http_redirect_is_rejected_before_request(self) -> None:
        handler = _RejectRedirectHandler()
        request = Request(
            "https://datasets-server.huggingface.co/rows",
            headers={"Authorization": "Bearer test-secret"},
        )

        with self.assertRaisesRegex(
            DatasetTransportError,
            "redirects are not allowed",
        ):
            handler.redirect_request(
                request,
                None,
                302,
                "Found",
                {},
                "http://datasets-server.huggingface.co/rows",
            )

    def test_rows_pagination_never_requests_more_than_one_hundred(self) -> None:
        rows = [{"text": f"row-{index}"} for index in range(205)]
        transport = FakeDatasetApiTransport("example/demo", rows)
        client = HuggingFaceDatasetClient(transport=transport)

        downloaded = tuple(
            client.iter_rows(
                "example/demo",
                "default",
                "train",
                limit=205,
                page_size=100,
            )
        )

        self.assertEqual([row.index for row in downloaded], list(range(205)))
        row_queries = [
            parse_qs(urlsplit(url).query)
            for url, _, _ in transport.calls
            if urlsplit(url).path == "/rows"
        ]
        self.assertEqual(
            [int(query["length"][0]) for query in row_queries],
            [100, 100, 5],
        )
        self.assertTrue(
            all(int(query["length"][0]) <= 100 for query in row_queries)
        )

    def test_retry_configuration_and_token_are_transport_only(self) -> None:
        secret = "hf_secret_value"
        transport = FakeDatasetApiTransport(
            "example/demo",
            [{"text": "hello"}],
            failures=1,
        )
        sleeps: list[float] = []
        client = HuggingFaceDatasetClient(
            transport=transport,
            token=secret,
            maximum_retries=2,
            retry_backoff_seconds=0.1,
            timeout_seconds=4.5,
            sleep=sleeps.append,
        )

        splits = client.list_splits("example/demo")

        self.assertEqual(len(splits), 1)
        self.assertEqual(sleeps, [0.1])
        self.assertEqual(len(transport.calls), 2)
        for _, headers, timeout in transport.calls:
            self.assertEqual(headers["Authorization"], f"Bearer {secret}")
            self.assertEqual(timeout, 4.5)
        self.assertNotIn(secret, repr(client))

    def test_revision_drift_is_rejected(self) -> None:
        transport = FakeDatasetApiTransport(
            "example/demo",
            [{"text": str(index)} for index in range(3)],
        )
        transport.revision_by_rows_call = {1: "rev-one", 2: "rev-two"}
        client = HuggingFaceDatasetClient(transport=transport)

        with self.assertRaisesRegex(
            DatasetSchemaError,
            "revision changed",
        ):
            tuple(
                client.iter_rows(
                    "example/demo",
                    "default",
                    "train",
                    page_size=2,
                )
            )

    def test_partial_and_truncated_responses_are_rejected(self) -> None:
        base = {
            "features": [
                {
                    "feature_idx": 0,
                    "name": "text",
                    "type": {"dtype": "string"},
                }
            ],
            "rows": [
                {
                    "row_idx": 0,
                    "row": {"text": "value"},
                    "truncated_cells": [],
                }
            ],
            "num_rows_total": 1,
            "num_rows_per_page": 100,
            "partial": True,
        }
        client = HuggingFaceDatasetClient(
            transport=StaticTransport(
                HttpResponse(
                    200,
                    json.dumps(base).encode(),
                    {"x-revision": "rev"},
                )
            )
        )
        with self.assertRaisesRegex(DatasetSchemaError, "partial"):
            client.get_rows_page(
                "example/demo",
                "default",
                "train",
                offset=0,
                length=1,
            )

        base["partial"] = False
        base["rows"][0]["truncated_cells"] = ["text"]
        client = HuggingFaceDatasetClient(
            transport=StaticTransport(
                HttpResponse(
                    200,
                    json.dumps(base).encode(),
                    {"x-revision": "rev"},
                )
            )
        )
        with self.assertRaisesRegex(DatasetSchemaError, "truncated"):
            client.get_rows_page(
                "example/demo",
                "default",
                "train",
                offset=0,
                length=1,
            )


class DatasetAdapterTests(unittest.TestCase):
    def test_dolly_adapter_preserves_category_and_formats_context(self) -> None:
        record = DollyInstructionAdapter().adapt(
            DatasetRow(
                4,
                {
                    "instruction": "Answer this",
                    "context": "Useful facts",
                    "response": "The answer",
                    "category": "closed_qa",
                },
            )
        )
        self.assertEqual(
            record,
            {
                "prompt": "Answer this\n\nContext:\nUseful facts",
                "response": "The answer",
                "category": "closed_qa",
            },
        )

    def test_text_and_preference_adapters_are_strict(self) -> None:
        self.assertEqual(
            TinyStoriesTextAdapter().adapt(
                DatasetRow(0, {"text": " A story. "})
            ),
            {"text": "A story."},
        )
        self.assertEqual(
            HhRlhfPreferenceAdapter().adapt(
                DatasetRow(0, {"chosen": "yes", "rejected": "no"})
            ),
            {"chosen": "yes", "rejected": "no"},
        )
        with self.assertRaisesRegex(DatasetSchemaError, "identical"):
            HhRlhfPreferenceAdapter().adapt(
                DatasetRow(0, {"chosen": "same", "rejected": "same"})
            )


class StableHashSplitterTests(unittest.TestCase):
    def test_assignment_is_order_independent_and_prevents_leakage(self) -> None:
        splitter = StableHashSplitter(seed="experiment-4")
        records = [
            {"prompt": f"p-{index}", "response": f"r-{index}"}
            for index in range(40)
        ]
        records.extend((records[3], records[3], records[17]))

        forward = [
            (splitter.fingerprint(record), splitter.assign(record))
            for record in records
        ]
        reverse = {
            splitter.fingerprint(record): splitter.assign(record)
            for record in reversed(records)
        }

        self.assertTrue(
            all(reverse[fingerprint] == split for fingerprint, split in forward)
        )
        duplicate_splits = {
            splitter.assign(record)
            for record in records
            if record == records[3]
        }
        self.assertEqual(len(duplicate_splits), 1)


class AtomicPublishTests(unittest.TestCase):
    def test_existing_empty_destination_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "staged"
            destination = root / "published"
            source.mkdir()
            (source / "payload.txt").write_text(
                "complete",
                encoding="utf-8",
            )
            destination.mkdir()
            destination_identity = destination.stat()

            with self.assertRaisesRegex(
                FileExistsError,
                "destination already exists",
            ):
                publish_directory_no_replace(source, destination)

            self.assertTrue(source.is_dir())
            self.assertEqual(
                (source / "payload.txt").read_text(encoding="utf-8"),
                "complete",
            )
            self.assertTrue(destination.is_dir())
            self.assertEqual(tuple(destination.iterdir()), ())
            self.assertEqual(
                destination.stat().st_ino,
                destination_identity.st_ino,
            )


class DatasetPreparationTests(unittest.TestCase):
    def test_destination_created_during_preparation_is_not_replaced(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "raced"
            destination_inode: list[int] = []

            class RacingTransport(FakeDatasetApiTransport):
                def get(
                    self,
                    url: str,
                    *,
                    headers: object,
                    timeout: float,
                ) -> HttpResponse:
                    response = super().get(
                        url,
                        headers=headers,
                        timeout=timeout,
                    )
                    if (
                        urlsplit(url).path == "/rows"
                        and self.rows_calls == 2
                    ):
                        destination.mkdir()
                        destination_inode.append(destination.stat().st_ino)
                    return response

            transport = RacingTransport(
                TINY_STORIES.dataset,
                [{"text": "A complete story."}],
            )
            with self.assertRaisesRegex(
                FileExistsError,
                "destination already exists",
            ):
                prepare_huggingface_dataset(
                    TINY_STORIES,
                    destination,
                    client=HuggingFaceDatasetClient(transport=transport),
                    limit=1,
                    selection="sequential",
                )

            self.assertEqual(len(destination_inode), 1)
            self.assertTrue(destination.is_dir())
            self.assertEqual(tuple(destination.iterdir()), ())
            self.assertEqual(
                destination.stat().st_ino,
                destination_inode[0],
            )
            self.assertEqual(tuple(root.glob(".raced.staging-*")), ())

    def test_dolly_is_sampled_deduplicated_hashed_and_reproducible(self) -> None:
        source_rows = [
            {
                "instruction": f"question {index % 10}",
                "context": f"context {index % 10}",
                "response": f"answer {index % 10}",
                "category": (
                    "closed_qa"
                    if (index // 10) % 2 == 0
                    else "open_qa"
                ),
            }
            for index in range(1000)
        ]
        transport = FakeDatasetApiTransport(
            DOLLY_15K.dataset,
            source_rows,
            revision="dolly-revision",
        )
        client = HuggingFaceDatasetClient(transport=transport)
        splitter = StableHashSplitter(seed="sample-11")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = prepare_huggingface_dataset(
                DOLLY_15K,
                root / "first",
                client=client,
                splitter=splitter,
                limit=120,
            )
            second_transport = FakeDatasetApiTransport(
                DOLLY_15K.dataset,
                source_rows,
                revision="dolly-revision",
            )
            second = prepare_huggingface_dataset(
                DOLLY_15K,
                root / "second",
                client=HuggingFaceDatasetClient(
                    transport=second_transport
                ),
                splitter=StableHashSplitter(seed="sample-11"),
                limit=120,
            )

            first_manifest = first.manifest
            counts = first_manifest["counts"]
            self.assertEqual(counts["source_rows"], 120)
            self.assertEqual(counts["adapted_records"], 10)
            self.assertEqual(counts["duplicates_removed"], 110)
            self.assertEqual(
                counts["train"] + counts["validation"] + counts["test"],
                10,
            )
            self.assertEqual(
                first_manifest["source"]["revision"],
                "dolly-revision",
            )
            self.assertEqual(
                first_manifest["source"]["license"],
                "cc-by-sa-3.0",
            )
            selected_ranges = first_manifest["source"]["selection"][
                "row_ranges"
            ]
            self.assertEqual(
                sum(item["length"] for item in selected_ranges),
                120,
            )
            self.assertNotEqual(
                selected_ranges,
                [
                    {"offset": 0, "length": 100},
                    {"offset": 100, "length": 20},
                ],
            )

            all_records: list[dict[str, str]] = []
            for partition in ("train", "validation", "test"):
                path = first.file(partition)
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line:
                        record = json.loads(line)
                        self.assertEqual(
                            set(record),
                            {"prompt", "response", "category"},
                        )
                        all_records.append(record)
                declared = first_manifest["files"][partition]
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    declared["sha256"],
                )
            canonical = {
                json.dumps(record, sort_keys=True)
                for record in all_records
            }
            self.assertEqual(len(canonical), len(all_records))
            identities = {
                (record["prompt"], record["response"])
                for record in all_records
            }
            self.assertEqual(len(identities), len(all_records))
            self.assertEqual(
                first.manifest_path.read_bytes(),
                second.manifest_path.read_bytes(),
            )
            for partition in ("train", "validation", "test"):
                self.assertEqual(
                    first.file(partition).read_bytes(),
                    second.file(partition).read_bytes(),
                )

    def test_tinystories_writes_directly_pretrainable_text(self) -> None:
        stories = [
            {"text": "First story."},
            {"text": "Second story."},
            {"text": "Third story."},
        ]
        transport = FakeDatasetApiTransport(
            TINY_STORIES.dataset,
            stories,
            split="validation",
            catalog_splits=("train", "validation"),
        )
        splitter = StableHashSplitter(
            SplitFractions(train=0.0, validation=1.0, test=0.0),
            seed=3,
        )
        with tempfile.TemporaryDirectory() as temporary:
            prepared = prepare_huggingface_dataset(
                TINY_STORIES,
                Path(temporary) / "tinystories",
                client=HuggingFaceDatasetClient(transport=transport),
                splitter=splitter,
                limit=3,
                selection="sequential",
                source_split="validation",
            )

            self.assertEqual(prepared.file("train").name, "train.txt")
            self.assertEqual(
                prepared.file("validation").read_text(encoding="utf-8"),
                "First story.\n\nSecond story.\n\nThird story.\n\n",
            )
            self.assertEqual(prepared.file("test").name, "test.txt")
            self.assertEqual(
                prepared.manifest["source"]["split"],
                "validation",
            )

    def test_verification_detects_tampering(self) -> None:
        transport = FakeDatasetApiTransport(
            HH_RLHF.dataset,
            [
                {"chosen": "good response", "rejected": "bad response"},
                {"chosen": "better response", "rejected": "worse response"},
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            prepared = prepare_huggingface_dataset(
                HH_RLHF,
                Path(temporary) / "preferences",
                client=HuggingFaceDatasetClient(transport=transport),
                limit=2,
                selection="sequential",
            )
            verify_prepared_dataset(prepared.directory)
            target = prepared.file("train")
            target.write_bytes(target.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "byte count|SHA-256"):
                verify_prepared_dataset(prepared.directory)

    def test_failed_adaptation_never_publishes_partial_directory(self) -> None:
        transport = FakeDatasetApiTransport(
            DOLLY_15K.dataset,
            [
                {
                    "instruction": " ",
                    "context": "",
                    "response": "answer",
                    "category": "open_qa",
                }
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            destination = parent / "broken"
            with self.assertRaises(DatasetSchemaError):
                prepare_huggingface_dataset(
                    DOLLY_15K,
                    destination,
                    client=HuggingFaceDatasetClient(transport=transport),
                    limit=1,
                )
            self.assertFalse(destination.exists())
            self.assertEqual(tuple(parent.glob(".broken.staging-*")), ())


if __name__ == "__main__":
    unittest.main()
