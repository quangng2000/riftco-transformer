"""Strict, dependency-free client for the Hugging Face dataset viewer API."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from types import MappingProxyType
from typing import Callable, Iterator, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


DEFAULT_DATASETS_SERVER_URL = "https://datasets-server.huggingface.co"
MAX_ROWS_PER_REQUEST = 100
_RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class DatasetClientError(RuntimeError):
    """Base error for dataset discovery and download failures."""


class DatasetTransportError(DatasetClientError):
    """The remote service could not be reached successfully."""


class DatasetSchemaError(DatasetClientError):
    """The remote service returned malformed or inconsistent data."""


class _RejectRedirectHandler(HTTPRedirectHandler):
    """Reject redirects before urllib can copy request headers to a new URL."""

    def redirect_request(
        self,
        request: Request,
        file_pointer: object,
        code: int,
        message: str,
        headers: object,
        new_url: str,
    ) -> Request:
        del request, file_pointer, code, message, headers, new_url
        raise DatasetTransportError(
            "dataset API redirects are not allowed"
        )


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Transport-neutral HTTP response used by the injectable client."""

    status: int
    body: bytes
    headers: Mapping[str, str]


class HttpTransport(Protocol):
    """Small adapter boundary that makes network behavior testable."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse:
        """Perform one HTTP GET without retries."""


class UrllibHttpTransport:
    """Bounded standard-library HTTP transport that never follows redirects."""

    __slots__ = ("_maximum_response_bytes", "_opener")

    def __init__(self, maximum_response_bytes: int = 64 * 1024 * 1024) -> None:
        if (
            isinstance(maximum_response_bytes, bool)
            or not isinstance(maximum_response_bytes, int)
        ):
            raise TypeError("maximum_response_bytes must be an int")
        if maximum_response_bytes <= 0:
            raise ValueError("maximum_response_bytes must be positive")
        self._maximum_response_bytes = maximum_response_bytes
        self._opener = build_opener(_RejectRedirectHandler())

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse:
        request = Request(url, headers=dict(headers), method="GET")
        with self._opener.open(request, timeout=timeout) as response:
            body = response.read(self._maximum_response_bytes + 1)
            if len(body) > self._maximum_response_bytes:
                raise DatasetTransportError(
                    "dataset API response exceeded the configured size limit"
                )
            response_headers = {
                name.lower(): value
                for name, value in response.headers.items()
            }
            return HttpResponse(
                status=response.status,
                body=body,
                headers=MappingProxyType(response_headers),
            )


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """One dataset/configuration/split tuple reported by ``/splits``."""

    dataset: str
    config: str
    split: str


@dataclass(frozen=True, slots=True)
class DatasetSplitCatalog:
    """Validated result from the dataset viewer ``/splits`` endpoint."""

    splits: tuple[DatasetSplit, ...]
    revision: str | None


@dataclass(frozen=True, slots=True)
class DatasetRow:
    """One indexed row returned by the dataset viewer."""

    index: int
    values: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class DatasetRowsPage:
    """One validated page from the dataset viewer ``/rows`` endpoint."""

    rows: tuple[DatasetRow, ...]
    total_rows: int
    feature_names: tuple[str, ...]
    revision: str | None


class HuggingFaceDatasetClient:
    """Client for official ``/splits`` and paginated ``/rows`` APIs.

    The optional token is kept only in an authorization header. It is omitted
    from ``repr`` and from all errors constructed by this class.
    """

    __slots__ = (
        "_base_url",
        "_backoff_seconds",
        "_headers",
        "_maximum_retries",
        "_sleep",
        "_timeout_seconds",
        "_transport",
    )

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        maximum_retries: int = 2,
        retry_backoff_seconds: float = 0.25,
        token: str | None = None,
        base_url: str = DEFAULT_DATASETS_SERVER_URL,
        transport: HttpTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._timeout_seconds = _positive_finite(
            timeout_seconds,
            "timeout_seconds",
        )
        if (
            isinstance(maximum_retries, bool)
            or not isinstance(maximum_retries, int)
        ):
            raise TypeError("maximum_retries must be an int")
        if maximum_retries < 0:
            raise ValueError("maximum_retries must not be negative")
        self._maximum_retries = maximum_retries
        self._backoff_seconds = _nonnegative_finite(
            retry_backoff_seconds,
            "retry_backoff_seconds",
        )
        if not callable(sleep):
            raise TypeError("sleep must be callable")
        self._sleep = sleep
        self._base_url = _validated_base_url(base_url)
        if transport is not None and not callable(
            getattr(transport, "get", None)
        ):
            raise TypeError("transport must provide a callable get() method")
        self._transport = (
            UrllibHttpTransport() if transport is None else transport
        )

        headers = {
            "Accept": "application/json",
            "User-Agent": "riftco-transformer/0.6.1 dataset-preparation",
        }
        if token is not None:
            if not isinstance(token, str):
                raise TypeError("token must be a str or None")
            if not token.strip():
                raise ValueError("token must not be blank")
            if any(ord(character) < 33 or ord(character) == 127 for character in token):
                raise ValueError(
                    "token must not contain whitespace or control characters"
                )
            if urlsplit(self._base_url).scheme != "https":
                raise ValueError(
                    "a Hugging Face token may only be sent over HTTPS"
                )
            headers["Authorization"] = f"Bearer {token}"
        self._headers = MappingProxyType(headers)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"base_url={self._base_url!r}, "
            f"timeout_seconds={self._timeout_seconds!r}, "
            f"maximum_retries={self._maximum_retries!r})"
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    def get_split_catalog(self, dataset: str) -> DatasetSplitCatalog:
        """Return all available configurations and splits for a dataset."""

        dataset_name = _nonblank_string(dataset, "dataset")
        value, revision = self._get_json(
            "/splits",
            {"dataset": dataset_name},
        )
        root = _object(value, "/splits response")
        raw_splits = _list(root, "splits", "/splits response")
        _list(root, "pending", "/splits response")
        _list(root, "failed", "/splits response")

        splits: list[DatasetSplit] = []
        seen: set[tuple[str, str, str]] = set()
        for index, raw_split in enumerate(raw_splits):
            context = f"/splits response.splits[{index}]"
            item = _object(raw_split, context)
            entry = DatasetSplit(
                dataset=_string(item, "dataset", context),
                config=_string(item, "config", context),
                split=_string(item, "split", context),
            )
            if entry.dataset != dataset_name:
                raise DatasetSchemaError(
                    f"{context}.dataset does not match the request"
                )
            key = (entry.dataset, entry.config, entry.split)
            if key in seen:
                raise DatasetSchemaError(
                    f"{context} duplicates an earlier split"
                )
            seen.add(key)
            splits.append(entry)
        return DatasetSplitCatalog(tuple(splits), revision)

    def list_splits(self, dataset: str) -> tuple[DatasetSplit, ...]:
        """Convenience view containing only validated split entries."""

        return self.get_split_catalog(dataset).splits

    def iter_pages(
        self,
        dataset: str,
        config: str,
        split: str,
        *,
        offset: int = 0,
        limit: int | None = None,
        page_size: int = MAX_ROWS_PER_REQUEST,
    ) -> Iterator[DatasetRowsPage]:
        """Yield consistent pages, never requesting more than 100 rows."""

        dataset_name = _nonblank_string(dataset, "dataset")
        config_name = _nonblank_string(config, "config")
        split_name = _nonblank_string(split, "split")
        start = _nonnegative_integer(offset, "offset")
        requested_limit = (
            None
            if limit is None
            else _nonnegative_integer(limit, "limit")
        )
        size = _positive_integer(page_size, "page_size")
        if size > MAX_ROWS_PER_REQUEST:
            raise ValueError(
                f"page_size must be at most {MAX_ROWS_PER_REQUEST}"
            )
        if requested_limit == 0:
            return

        next_offset = start
        remaining = requested_limit
        expected_total: int | None = None
        expected_features: tuple[str, ...] | None = None
        expected_revision: str | None = None
        first_page = True
        while remaining is None or remaining > 0:
            length = size if remaining is None else min(size, remaining)
            page = self.get_rows_page(
                dataset_name,
                config_name,
                split_name,
                offset=next_offset,
                length=length,
            )
            if first_page:
                expected_total = page.total_rows
                expected_features = page.feature_names
                expected_revision = page.revision
                first_page = False
            else:
                if page.total_rows != expected_total:
                    raise DatasetSchemaError(
                        "num_rows_total changed during pagination"
                    )
                if page.feature_names != expected_features:
                    raise DatasetSchemaError(
                        "dataset features changed during pagination"
                    )
                if page.revision != expected_revision:
                    raise DatasetSchemaError(
                        "dataset revision changed during pagination"
                    )

            if not page.rows:
                if next_offset < page.total_rows:
                    raise DatasetSchemaError(
                        "the rows endpoint ended before num_rows_total"
                    )
                break
            yield page
            count = len(page.rows)
            next_offset += count
            if remaining is not None:
                remaining -= count
            if next_offset >= page.total_rows:
                break
            if count != length:
                raise DatasetSchemaError(
                    "the rows endpoint returned a short intermediate page"
                )

    def iter_rows(
        self,
        dataset: str,
        config: str,
        split: str,
        *,
        offset: int = 0,
        limit: int | None = None,
        page_size: int = MAX_ROWS_PER_REQUEST,
    ) -> Iterator[DatasetRow]:
        """Yield rows while preserving the service's stable source order."""

        for page in self.iter_pages(
            dataset,
            config,
            split,
            offset=offset,
            limit=limit,
            page_size=page_size,
        ):
            yield from page.rows

    def get_rows_page(
        self,
        dataset: str,
        config: str,
        split: str,
        *,
        offset: int,
        length: int,
    ) -> DatasetRowsPage:
        """Fetch one explicitly bounded page from ``/rows``."""

        dataset_name = _nonblank_string(dataset, "dataset")
        config_name = _nonblank_string(config, "config")
        split_name = _nonblank_string(split, "split")
        page_offset = _nonnegative_integer(offset, "offset")
        page_length = _positive_integer(length, "length")
        if page_length > MAX_ROWS_PER_REQUEST:
            raise ValueError(
                f"length must be at most {MAX_ROWS_PER_REQUEST}"
            )
        value, revision = self._get_json(
            "/rows",
            {
                "dataset": dataset_name,
                "config": config_name,
                "split": split_name,
                "offset": str(page_offset),
                "length": str(page_length),
            },
        )
        root = _object(value, "/rows response")
        raw_features = _list(root, "features", "/rows response")
        raw_rows = _list(root, "rows", "/rows response")
        total_rows = _integer(root, "num_rows_total", "/rows response")
        rows_per_page = _integer(
            root,
            "num_rows_per_page",
            "/rows response",
        )
        partial = _boolean(root, "partial", "/rows response")
        if total_rows < 0:
            raise DatasetSchemaError(
                "/rows response.num_rows_total must not be negative"
            )
        if not 1 <= rows_per_page <= MAX_ROWS_PER_REQUEST:
            raise DatasetSchemaError(
                "/rows response.num_rows_per_page must be between 1 and 100"
            )
        if partial:
            raise DatasetSchemaError(
                "partial dataset responses are not safe for preparation"
            )
        if len(raw_rows) > page_length:
            raise DatasetSchemaError(
                "/rows response contains more rows than requested"
            )

        feature_names: list[str] = []
        seen_features: set[str] = set()
        for index, raw_feature in enumerate(raw_features):
            context = f"/rows response.features[{index}]"
            feature = _object(raw_feature, context)
            feature_index = _integer(feature, "feature_idx", context)
            if feature_index != index:
                raise DatasetSchemaError(
                    f"{context}.feature_idx must equal {index}"
                )
            name = _string(feature, "name", context)
            _object(_required(feature, "type", context), f"{context}.type")
            if name in seen_features:
                raise DatasetSchemaError(
                    f"{context}.name duplicates an earlier feature"
                )
            seen_features.add(name)
            feature_names.append(name)
        if not feature_names:
            raise DatasetSchemaError(
                "/rows response.features must not be empty"
            )

        rows: list[DatasetRow] = []
        feature_set = set(feature_names)
        for position, raw_row in enumerate(raw_rows):
            context = f"/rows response.rows[{position}]"
            item = _object(raw_row, context)
            row_index = _integer(item, "row_idx", context)
            expected_index = page_offset + position
            if row_index != expected_index:
                raise DatasetSchemaError(
                    f"{context}.row_idx must equal {expected_index}"
                )
            values = _object(_required(item, "row", context), f"{context}.row")
            if set(values) != feature_set:
                raise DatasetSchemaError(
                    f"{context}.row fields do not match features"
                )
            truncated = _list(item, "truncated_cells", context)
            if truncated:
                raise DatasetSchemaError(
                    f"{context} contains truncated cells"
                )
            rows.append(
                DatasetRow(
                    index=row_index,
                    values=MappingProxyType(dict(values)),
                )
            )
        if page_offset + len(rows) > total_rows:
            raise DatasetSchemaError(
                "/rows response extends beyond num_rows_total"
            )
        return DatasetRowsPage(
            rows=tuple(rows),
            total_rows=total_rows,
            feature_names=tuple(feature_names),
            revision=revision,
        )

    def _get_json(
        self,
        path: str,
        parameters: Mapping[str, str],
    ) -> tuple[object, str | None]:
        url = f"{self._base_url}{path}?{urlencode(parameters)}"
        response: HttpResponse | None = None
        for attempt in range(self._maximum_retries + 1):
            try:
                response = self._transport.get(
                    url,
                    headers=self._headers,
                    timeout=self._timeout_seconds,
                )
                if not isinstance(response, HttpResponse):
                    raise DatasetTransportError(
                        "HTTP transport returned an invalid response"
                    )
                _validate_http_response(response)
                if 200 <= response.status < 300:
                    break
                if (
                    response.status not in _RETRYABLE_HTTP_STATUS
                    or attempt == self._maximum_retries
                ):
                    raise DatasetTransportError(
                        f"dataset API returned HTTP {response.status}"
                    )
            except HTTPError as error:
                if (
                    error.code not in _RETRYABLE_HTTP_STATUS
                    or attempt == self._maximum_retries
                ):
                    raise DatasetTransportError(
                        f"dataset API returned HTTP {error.code}"
                    ) from None
            except (URLError, TimeoutError, OSError):
                if attempt == self._maximum_retries:
                    raise DatasetTransportError(
                        "dataset API request failed after "
                        f"{attempt + 1} attempt(s)"
                    ) from None
            if attempt < self._maximum_retries:
                self._sleep(self._backoff_seconds * (2**attempt))
        if response is None or not 200 <= response.status < 300:
            raise DatasetTransportError("dataset API request failed")

        try:
            value = _strict_json_loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise DatasetSchemaError(
                "dataset API returned invalid JSON"
            ) from error
        revision = _revision_header(response.headers)
        return value, revision


def _strict_json_loads(body: bytes) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON number: {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate JSON key: {name}")
            result[name] = value
        return result

    text = body.decode("utf-8")
    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _validate_http_response(response: HttpResponse) -> None:
    if isinstance(response.status, bool) or not isinstance(
        response.status,
        int,
    ):
        raise DatasetTransportError("HTTP response status must be an int")
    if not 100 <= response.status <= 599:
        raise DatasetTransportError("HTTP response status is out of range")
    if not isinstance(response.body, bytes):
        raise DatasetTransportError("HTTP response body must be bytes")
    if not isinstance(response.headers, Mapping):
        raise DatasetTransportError("HTTP response headers must be a mapping")
    if any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in response.headers.items()
    ):
        raise DatasetTransportError(
            "HTTP response headers must contain strings"
        )


def _revision_header(headers: Mapping[str, str]) -> str | None:
    normalized = {str(name).lower(): value for name, value in headers.items()}
    raw_revision = normalized.get("x-revision")
    if raw_revision is None:
        return None
    if not isinstance(raw_revision, str) or not raw_revision.strip():
        raise DatasetSchemaError("X-Revision header must be a nonblank string")
    revision = raw_revision.strip()
    if any(character.isspace() for character in revision):
        raise DatasetSchemaError("X-Revision header must not contain whitespace")
    return revision


def _validated_base_url(value: str) -> str:
    url = _nonblank_string(value, "base_url").rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("base_url must not include user information")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not include a query or fragment")
    return url


def _required(
    value: Mapping[str, object],
    name: str,
    context: str,
) -> object:
    if name not in value:
        raise DatasetSchemaError(f"{context} requires {name}")
    return value[name]


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise DatasetSchemaError(f"{context} must be a JSON object")
    if any(not isinstance(name, str) for name in value):
        raise DatasetSchemaError(f"{context} keys must be strings")
    return value


def _list(
    value: Mapping[str, object],
    name: str,
    context: str,
) -> list[object]:
    result = _required(value, name, context)
    if not isinstance(result, list):
        raise DatasetSchemaError(f"{context}.{name} must be a JSON array")
    return result


def _string(
    value: Mapping[str, object],
    name: str,
    context: str,
) -> str:
    result = _required(value, name, context)
    if not isinstance(result, str) or not result.strip():
        raise DatasetSchemaError(
            f"{context}.{name} must be a nonblank string"
        )
    return result


def _integer(
    value: Mapping[str, object],
    name: str,
    context: str,
) -> int:
    result = _required(value, name, context)
    if isinstance(result, bool) or not isinstance(result, int):
        raise DatasetSchemaError(f"{context}.{name} must be an integer")
    return result


def _boolean(
    value: Mapping[str, object],
    name: str,
    context: str,
) -> bool:
    result = _required(value, name, context)
    if not isinstance(result, bool):
        raise DatasetSchemaError(f"{context}.{name} must be a boolean")
    return result


def _nonblank_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str")
    if not value.strip():
        raise ValueError(f"{name} must not be blank")
    return value


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


__all__ = [
    "DEFAULT_DATASETS_SERVER_URL",
    "MAX_ROWS_PER_REQUEST",
    "DatasetClientError",
    "DatasetRow",
    "DatasetRowsPage",
    "DatasetSchemaError",
    "DatasetSplit",
    "DatasetSplitCatalog",
    "DatasetTransportError",
    "HttpResponse",
    "HttpTransport",
    "HuggingFaceDatasetClient",
    "UrllibHttpTransport",
]
