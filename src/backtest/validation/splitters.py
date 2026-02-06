"""Data splitters for validation.

검증을 위한 데이터 분할 유틸리티를 제공합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization, DatetimeIndex
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from datetime import UTC
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandas import DataFrame

from src.backtest.validation.models import SplitInfo
from src.data.market_data import MarketDataSet, MultiSymbolData


def split_is_oos(
    data: MarketDataSet,
    ratio: float = 0.7,
) -> tuple[MarketDataSet, MarketDataSet]:
    """In-Sample / Out-of-Sample 단순 분할.

    데이터를 시간순으로 Train/Test로 분할합니다.
    Look-ahead bias 방지를 위해 시간순 분할만 지원합니다.

    Args:
        data: 원본 MarketDataSet
        ratio: Train 비율 (0.0 ~ 1.0), 기본값 0.7 (70%)

    Returns:
        (train_data, test_data) 튜플

    Raises:
        ValueError: ratio가 0.0 ~ 1.0 범위를 벗어나는 경우

    Example:
        >>> train, test = split_is_oos(data, ratio=0.7)
        >>> print(f"Train: {train.periods}, Test: {test.periods}")
    """
    if not 0.0 < ratio < 1.0:
        msg = f"ratio must be between 0.0 and 1.0, got {ratio}"
        raise ValueError(msg)

    df = data.ohlcv
    n = len(df)
    split_idx = int(n * ratio)

    if split_idx < 1 or split_idx >= n - 1:
        msg = f"Not enough data to split: {n} rows with ratio {ratio}"
        raise ValueError(msg)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # DatetimeIndex 타입 단언
    train_index = train_df.index
    test_index = test_df.index

    train_data = MarketDataSet(
        symbol=data.symbol,
        timeframe=data.timeframe,
        start=train_index[0].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
        end=train_index[-1].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
        ohlcv=train_df,
    )

    test_data = MarketDataSet(
        symbol=data.symbol,
        timeframe=data.timeframe,
        start=test_index[0].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
        end=test_index[-1].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
        ohlcv=test_df,
    )

    return train_data, test_data


def split_walk_forward(
    data: MarketDataSet,
    n_folds: int = 5,
    min_train_ratio: float = 0.5,
    expanding: bool = True,
) -> list[tuple[MarketDataSet, MarketDataSet, SplitInfo]]:
    """Walk-Forward 분할.

    시간순으로 Fold를 생성하여 Walk-Forward 검증을 지원합니다.
    각 Fold에서 Train은 과거 데이터, Test는 바로 다음 기간 데이터입니다.

    Args:
        data: 원본 MarketDataSet
        n_folds: Fold 수 (기본값 5)
        min_train_ratio: 최소 Train 비율 (기본값 0.5)
        expanding: True면 Train이 누적, False면 고정 윈도우

    Returns:
        List of (train_data, test_data, split_info) 튜플

    Example:
        >>> folds = split_walk_forward(data, n_folds=5)
        >>> for train, test, info in folds:
        ...     print(f"Fold {info.fold_id}: Train {info.train_periods}, Test {info.test_periods}")
    """
    min_folds = 2
    if n_folds < min_folds:
        msg = f"n_folds must be >= {min_folds}, got {n_folds}"
        raise ValueError(msg)

    df = data.ohlcv
    n = len(df)

    # Test 크기 계산 (각 Fold의 Test는 동일 크기)
    # 전체 데이터를 (n_folds + 1) 등분하고, 마지막 n_folds개가 Test
    # 첫 번째 부분은 최소 Train
    test_size = n // (n_folds + 1)
    min_train_size = int(n * min_train_ratio)

    min_test_size = 10
    if test_size < min_test_size:
        msg = f"Test size too small: {test_size}. Need more data or fewer folds."
        raise ValueError(msg)

    results: list[tuple[MarketDataSet, MarketDataSet, SplitInfo]] = []

    for fold_id in range(n_folds):
        # 기본값 초기화 (타입 체커용)
        train_start_idx = 0

        if expanding:
            # Expanding window: Train이 점점 커짐
            train_end_idx = min_train_size + (fold_id * test_size)
        else:
            # Sliding window: Train 크기 고정
            train_start_idx = fold_id * test_size
            train_end_idx = train_start_idx + min_train_size

        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_size

        # 범위 체크
        if test_end_idx > n:
            break

        if expanding:
            train_df = df.iloc[:train_end_idx].copy()
        else:
            train_df = df.iloc[train_start_idx:train_end_idx].copy()

        test_df = df.iloc[test_start_idx:test_end_idx].copy()

        # DatetimeIndex 타입 단언
        train_index = train_df.index
        test_index = test_df.index

        train_data = MarketDataSet(
            symbol=data.symbol,
            timeframe=data.timeframe,
            start=train_index[0].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
            end=train_index[-1].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
            ohlcv=train_df,
        )

        test_data = MarketDataSet(
            symbol=data.symbol,
            timeframe=data.timeframe,
            start=test_index[0].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
            end=test_index[-1].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
            ohlcv=test_df,
        )

        split_info = SplitInfo(
            fold_id=fold_id,
            train_start=train_data.start,
            train_end=train_data.end,
            test_start=test_data.start,
            test_end=test_data.end,
            train_periods=len(train_df),
            test_periods=len(test_df),
        )

        results.append((train_data, test_data, split_info))

    return results


def split_cpcv(
    data: MarketDataSet,
    n_splits: int = 5,
    n_test_splits: int = 2,
    purge_periods: int = 5,
    embargo_periods: int = 5,
) -> Iterator[tuple[MarketDataSet, MarketDataSet, SplitInfo]]:
    """CPCV (Combinatorial Purged Cross-Validation) 분할.

    Marcos López de Prado의 방법론을 구현합니다.
    - 데이터를 n_splits개의 그룹으로 나눔
    - n_test_splits개의 그룹을 Test로, 나머지를 Train으로 사용
    - Purge: Train과 Test 사이에 갭을 둠 (look-ahead bias 방지)
    - Embargo: Test 직후 데이터를 제외 (lagged features 영향 제거)

    Args:
        data: 원본 MarketDataSet
        n_splits: 총 그룹 수 (기본값 5)
        n_test_splits: Test로 사용할 그룹 수 (기본값 2)
        purge_periods: Purge 기간 (Train-Test 사이 갭)
        embargo_periods: Embargo 기간 (Test 직후 제외)

    Yields:
        (train_data, test_data, split_info) 튜플

    Note:
        CPCV는 C(n, k) 조합을 생성하므로 계산 비용이 높습니다.
        n_splits=5, n_test_splits=2이면 C(5,2)=10개의 Fold가 생성됩니다.

    Example:
        >>> for train, test, info in split_cpcv(data, n_splits=5, n_test_splits=2):
        ...     print(f"Fold {info.fold_id}: Train {info.train_periods}, Test {info.test_periods}")
    """
    if n_test_splits >= n_splits:
        msg = f"n_test_splits ({n_test_splits}) must be < n_splits ({n_splits})"
        raise ValueError(msg)

    df = data.ohlcv
    n = len(df)

    # 그룹 인덱스 계산
    group_size = n // n_splits
    group_indices = [(i * group_size, min((i + 1) * group_size, n)) for i in range(n_splits)]

    # 테스트 그룹 조합 생성
    test_group_combinations = list(combinations(range(n_splits), n_test_splits))

    for fold_id, test_groups in enumerate(test_group_combinations):
        train_groups = [i for i in range(n_splits) if i not in test_groups]

        # Test 인덱스 수집 (정렬)
        test_indices: list[int] = []
        for group_id in sorted(test_groups):
            start_idx, end_idx = group_indices[group_id]
            test_indices.extend(range(start_idx, end_idx))

        # Train 인덱스 수집 (Purge & Embargo 적용)
        train_indices: list[int] = []
        for group_id in train_groups:
            start_idx, end_idx = group_indices[group_id]

            # Purge: Test 직전 그룹이면 끝에서 purge_periods 제외
            if group_id + 1 in test_groups:
                end_idx = max(start_idx, end_idx - purge_periods)

            # Embargo: Test 직후 그룹이면 시작에서 embargo_periods 제외
            if group_id - 1 in test_groups:
                start_idx = min(end_idx, start_idx + embargo_periods)

            if start_idx < end_idx:
                train_indices.extend(range(start_idx, end_idx))

        if not train_indices or not test_indices:
            continue

        train_df = df.iloc[train_indices].copy()
        test_df = df.iloc[test_indices].copy()

        # DatetimeIndex 타입 단언
        train_index = train_df.index
        test_index = test_df.index

        train_data = MarketDataSet(
            symbol=data.symbol,
            timeframe=data.timeframe,
            start=train_index[0].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
            end=train_index[-1].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
            ohlcv=train_df,
        )

        test_data = MarketDataSet(
            symbol=data.symbol,
            timeframe=data.timeframe,
            start=test_index[0].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
            end=test_index[-1].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
            ohlcv=test_df,
        )

        split_info = SplitInfo(
            fold_id=fold_id,
            train_start=train_data.start,
            train_end=train_data.end,
            test_start=test_data.start,
            test_end=test_data.end,
            train_periods=len(train_df),
            test_periods=len(test_df),
        )

        yield train_data, test_data, split_info


def get_split_info_is_oos(
    data: MarketDataSet,
    ratio: float = 0.7,
) -> SplitInfo:
    """IS/OOS 분할 정보만 반환 (실제 분할 없이).

    Args:
        data: 원본 MarketDataSet
        ratio: Train 비율

    Returns:
        SplitInfo 객체
    """
    n = len(data.ohlcv)
    split_idx = int(n * ratio)
    df = data.ohlcv
    idx = df.index

    return SplitInfo(
        fold_id=0,
        train_start=idx[0].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
        train_end=idx[split_idx - 1].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
        test_start=idx[split_idx].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
        test_end=idx[-1].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
        train_periods=split_idx,
        test_periods=n - split_idx,
    )


# =============================================================================
# Multi-Asset Splitters
# =============================================================================


def split_multi_is_oos(
    data: MultiSymbolData,
    ratio: float = 0.7,
) -> tuple[MultiSymbolData, MultiSymbolData]:
    """멀티에셋 IS/OOS 분할 (동일 시간 경계).

    모든 심볼을 동일한 split 인덱스에서 분할합니다.
    첫 번째 심볼의 인덱스를 기준으로 split 지점을 결정합니다.

    Args:
        data: 원본 MultiSymbolData
        ratio: Train 비율 (0.0 ~ 1.0)

    Returns:
        (train_data, test_data) 튜플
    """
    if not 0.0 < ratio < 1.0:
        msg = f"ratio must be between 0.0 and 1.0, got {ratio}"
        raise ValueError(msg)

    # 첫 번째 심볼 기준으로 split 인덱스 계산
    ref_df = data.ohlcv[data.symbols[0]]
    n = len(ref_df)
    split_idx = int(n * ratio)

    if split_idx < 1 or split_idx >= n - 1:
        msg = f"Not enough data to split: {n} rows with ratio {ratio}"
        raise ValueError(msg)

    train_data = data.slice_iloc(0, split_idx)
    test_data = data.slice_iloc(split_idx, n)

    return train_data, test_data


def split_multi_walk_forward(
    data: MultiSymbolData,
    n_folds: int = 5,
    min_train_ratio: float = 0.5,
    expanding: bool = True,
) -> list[tuple[MultiSymbolData, MultiSymbolData, SplitInfo]]:
    """멀티에셋 Walk-Forward 분할.

    모든 심볼에 동일한 시간 경계를 적용합니다.

    Args:
        data: 원본 MultiSymbolData
        n_folds: Fold 수
        min_train_ratio: 최소 Train 비율
        expanding: True면 누적 Train

    Returns:
        List of (train_data, test_data, split_info) 튜플
    """
    min_folds = 2
    if n_folds < min_folds:
        msg = f"n_folds must be >= {min_folds}, got {n_folds}"
        raise ValueError(msg)

    ref_df = data.ohlcv[data.symbols[0]]
    n = len(ref_df)

    test_size = n // (n_folds + 1)
    min_train_size = int(n * min_train_ratio)

    min_test_size = 10
    if test_size < min_test_size:
        msg = f"Test size too small: {test_size}. Need more data or fewer folds."
        raise ValueError(msg)

    results: list[tuple[MultiSymbolData, MultiSymbolData, SplitInfo]] = []
    ref_index = ref_df.index

    for fold_id in range(n_folds):
        train_start_idx = 0

        if expanding:
            train_end_idx = min_train_size + (fold_id * test_size)
        else:
            train_start_idx = fold_id * test_size
            train_end_idx = train_start_idx + min_train_size

        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_size

        if test_end_idx > n:
            break

        train_data = data.slice_iloc(train_start_idx, train_end_idx)
        test_data = data.slice_iloc(test_start_idx, test_end_idx)

        split_info = SplitInfo(
            fold_id=fold_id,
            train_start=ref_index[train_start_idx].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
            train_end=ref_index[train_end_idx - 1].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
            test_start=ref_index[test_start_idx].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
            test_end=ref_index[test_end_idx - 1].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
            train_periods=train_end_idx - train_start_idx,
            test_periods=test_size,
        )

        results.append((train_data, test_data, split_info))

    return results


def split_multi_cpcv(
    data: MultiSymbolData,
    n_splits: int = 5,
    n_test_splits: int = 2,
    purge_periods: int = 5,
    embargo_periods: int = 5,
) -> Iterator[tuple[MultiSymbolData, MultiSymbolData, SplitInfo]]:
    """멀티에셋 CPCV 분할.

    모든 심볼에 동일한 CPCV 그룹/조합을 적용합니다.

    Args:
        data: 원본 MultiSymbolData
        n_splits: 총 그룹 수
        n_test_splits: Test로 사용할 그룹 수
        purge_periods: Purge 기간
        embargo_periods: Embargo 기간

    Yields:
        (train_data, test_data, split_info) 튜플
    """
    if n_test_splits >= n_splits:
        msg = f"n_test_splits ({n_test_splits}) must be < n_splits ({n_splits})"
        raise ValueError(msg)

    ref_df = data.ohlcv[data.symbols[0]]
    n = len(ref_df)
    ref_index = ref_df.index

    group_size = n // n_splits
    group_indices = [(i * group_size, min((i + 1) * group_size, n)) for i in range(n_splits)]

    test_group_combinations = list(combinations(range(n_splits), n_test_splits))

    for fold_id, test_groups in enumerate(test_group_combinations):
        train_groups = [i for i in range(n_splits) if i not in test_groups]

        # Test 인덱스 수집
        test_indices: list[int] = []
        for group_id in sorted(test_groups):
            start_idx, end_idx = group_indices[group_id]
            test_indices.extend(range(start_idx, end_idx))

        # Train 인덱스 수집 (Purge & Embargo)
        train_indices: list[int] = []
        for group_id in train_groups:
            start_idx, end_idx = group_indices[group_id]

            if group_id + 1 in test_groups:
                end_idx = max(start_idx, end_idx - purge_periods)
            if group_id - 1 in test_groups:
                start_idx = min(end_idx, start_idx + embargo_periods)

            if start_idx < end_idx:
                train_indices.extend(range(start_idx, end_idx))

        if not train_indices or not test_indices:
            continue

        # 멀티에셋 데이터 슬라이싱 (인덱스 기반)
        train_ohlcv: dict[str, DataFrame] = {}
        test_ohlcv: dict[str, DataFrame] = {}
        for symbol in data.symbols:
            df = data.ohlcv[symbol]
            train_ohlcv[symbol] = df.iloc[train_indices].copy()
            test_ohlcv[symbol] = df.iloc[test_indices].copy()

        train_ref_idx = ref_index[train_indices]
        test_ref_idx = ref_index[test_indices]

        train_data = MultiSymbolData(
            symbols=list(data.symbols),
            timeframe=data.timeframe,
            start=train_ref_idx[0].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
            end=train_ref_idx[-1].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
            ohlcv=train_ohlcv,
        )

        test_data = MultiSymbolData(
            symbols=list(data.symbols),
            timeframe=data.timeframe,
            start=test_ref_idx[0].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
            end=test_ref_idx[-1].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
            ohlcv=test_ohlcv,
        )

        split_info = SplitInfo(
            fold_id=fold_id,
            train_start=train_data.start,
            train_end=train_data.end,
            test_start=test_data.start,
            test_end=test_data.end,
            train_periods=len(train_indices),
            test_periods=len(test_indices),
        )

        yield train_data, test_data, split_info
