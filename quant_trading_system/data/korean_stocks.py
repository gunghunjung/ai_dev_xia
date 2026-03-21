# data/korean_stocks.py — 전체 한국/미국 종목 데이터베이스
# KRX 전체 종목: pykrx로 실시간 로드 + JSON 캐시 (24hr TTL)
# 미국/ETF/지수: 정적 목록
from __future__ import annotations
import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("quant.stocks")

# ── 캐시 파일 경로 ────────────────────────────────────────────
_CACHE_FILE = Path(__file__).parent / ".krx_cache.json"
_CACHE_TTL  = 60 * 60 * 24  # 24시간

# ── 정적 DB: 미국/ETF/지수 (pykrx 없어도 항상 사용) ─────────
_STATIC_DB: List[tuple] = [
    # ── KOSPI ETF ────────────────────────────────────────────
    ("069500.KS", "KODEX 200",           "ETF",   "인덱스"),
    ("102110.KS", "TIGER 200",           "ETF",   "인덱스"),
    ("229200.KS", "KODEX 코스닥150",     "ETF",   "인덱스"),
    ("114800.KS", "KODEX 인버스",        "ETF",   "인버스"),
    ("122630.KS", "KODEX 레버리지",      "ETF",   "레버리지"),
    ("252670.KS", "KODEX 200선물인버스2X","ETF",  "인버스"),
    ("278530.KS", "KODEX 200TR",         "ETF",   "인덱스"),
    ("148020.KS", "KINDEX 200",          "ETF",   "인덱스"),
    ("305720.KS", "KODEX 2차전지산업",   "ETF",   "2차전지"),
    ("091160.KS", "KODEX 반도체",        "ETF",   "반도체"),
    ("091170.KS", "KODEX 은행",          "ETF",   "금융"),
    ("157490.KS", "TIGER 200IT",         "ETF",   "IT"),
    ("139230.KS", "TIGER 200 금융",      "ETF",   "금융"),
    ("364980.KS", "TIGER 2차전지테마",   "ETF",   "2차전지"),
    ("287310.KS", "TIGER TOP10",         "ETF",   "인덱스"),
    ("278540.KS", "TIGER 200TR",         "ETF",   "인덱스"),
    ("361580.KS", "TIGER K-미래차액티브","ETF",   "자동차"),
    # ── 미국 주요 주식 ────────────────────────────────────────
    ("AAPL",  "Apple",                   "NASDAQ", "테크"),
    ("MSFT",  "Microsoft",               "NASDAQ", "테크"),
    ("NVDA",  "NVIDIA",                  "NASDAQ", "반도체"),
    ("GOOGL", "Alphabet (Google)",       "NASDAQ", "테크"),
    ("GOOG",  "Alphabet C",              "NASDAQ", "테크"),
    ("AMZN",  "Amazon",                  "NASDAQ", "이커머스"),
    ("META",  "Meta Platforms",          "NASDAQ", "소셜미디어"),
    ("TSLA",  "Tesla",                   "NASDAQ", "전기차"),
    ("AVGO",  "Broadcom",                "NASDAQ", "반도체"),
    ("JPM",   "JPMorgan Chase",          "NYSE",   "금융"),
    ("V",     "Visa",                    "NYSE",   "결제"),
    ("UNH",   "UnitedHealth Group",      "NYSE",   "헬스케어"),
    ("XOM",   "ExxonMobil",              "NYSE",   "에너지"),
    ("BRK-B", "Berkshire Hathaway",      "NYSE",   "금융"),
    ("LLY",   "Eli Lilly",               "NYSE",   "제약"),
    ("JNJ",   "Johnson & Johnson",       "NYSE",   "헬스케어"),
    ("AMD",   "AMD",                     "NASDAQ", "반도체"),
    ("INTC",  "Intel",                   "NASDAQ", "반도체"),
    ("QCOM",  "Qualcomm",                "NASDAQ", "반도체"),
    ("NFLX",  "Netflix",                 "NASDAQ", "미디어"),
    ("DIS",   "Disney",                  "NYSE",   "미디어"),
    ("PYPL",  "PayPal",                  "NASDAQ", "결제"),
    ("ADBE",  "Adobe",                   "NASDAQ", "테크"),
    ("CRM",   "Salesforce",              "NYSE",   "테크"),
    ("ORCL",  "Oracle",                  "NYSE",   "테크"),
    ("IBM",   "IBM",                     "NYSE",   "테크"),
    ("CSCO",  "Cisco",                   "NASDAQ", "네트워크"),
    ("UBER",  "Uber",                    "NYSE",   "플랫폼"),
    ("ABNB",  "Airbnb",                  "NASDAQ", "플랫폼"),
    ("SHOP",  "Shopify",                 "NYSE",   "이커머스"),
    ("SQ",    "Block (Square)",          "NYSE",   "핀테크"),
    ("PLTR",  "Palantir",                "NYSE",   "AI"),
    ("ARM",   "ARM Holdings",            "NASDAQ", "반도체"),
    ("SMCI",  "Super Micro Computer",    "NASDAQ", "서버"),
    ("MU",    "Micron Technology",       "NASDAQ", "반도체"),
    ("AMAT",  "Applied Materials",       "NASDAQ", "반도체장비"),
    ("KLAC",  "KLA Corporation",         "NASDAQ", "반도체장비"),
    ("LRCX",  "Lam Research",            "NASDAQ", "반도체장비"),
    ("ASML",  "ASML Holding",            "NASDAQ", "반도체장비"),
    ("TSM",   "TSMC",                    "NYSE",   "반도체"),
    ("GS",    "Goldman Sachs",           "NYSE",   "금융"),
    ("MS",    "Morgan Stanley",          "NYSE",   "금융"),
    ("BAC",   "Bank of America",         "NYSE",   "금융"),
    ("C",     "Citigroup",               "NYSE",   "금융"),
    ("WFC",   "Wells Fargo",             "NYSE",   "금융"),
    ("CVX",   "Chevron",                 "NYSE",   "에너지"),
    ("PFE",   "Pfizer",                  "NYSE",   "제약"),
    ("MRK",   "Merck",                   "NYSE",   "제약"),
    ("ABBV",  "AbbVie",                  "NYSE",   "제약"),
    ("NVO",   "Novo Nordisk",            "NYSE",   "제약"),
    ("BA",    "Boeing",                  "NYSE",   "항공우주"),
    ("CAT",   "Caterpillar",             "NYSE",   "산업재"),
    ("DE",    "Deere & Company",         "NYSE",   "산업재"),
    ("HON",   "Honeywell",               "NASDAQ", "산업재"),
    ("GE",    "GE Aerospace",            "NYSE",   "항공우주"),
    ("RTX",   "RTX Corporation",         "NYSE",   "방산"),
    ("LMT",   "Lockheed Martin",         "NYSE",   "방산"),
    ("COST",  "Costco",                  "NASDAQ", "유통"),
    ("WMT",   "Walmart",                 "NYSE",   "유통"),
    ("TGT",   "Target",                  "NYSE",   "유통"),
    ("HD",    "Home Depot",              "NYSE",   "유통"),
    ("NKE",   "Nike",                    "NYSE",   "소비재"),
    ("SBUX",  "Starbucks",               "NASDAQ", "F&B"),
    ("MCD",   "McDonald's",              "NYSE",   "F&B"),
    ("KO",    "Coca-Cola",               "NYSE",   "음료"),
    ("PEP",   "PepsiCo",                 "NASDAQ", "음료"),
    ("T",     "AT&T",                    "NYSE",   "통신"),
    ("VZ",    "Verizon",                 "NYSE",   "통신"),
    ("TMUS",  "T-Mobile",                "NASDAQ", "통신"),
    ("COIN",  "Coinbase",                "NASDAQ", "암호화폐"),
    ("MSTR",  "MicroStrategy",           "NASDAQ", "암호화폐"),
    # ── 미국 ETF ──────────────────────────────────────────────
    ("SPY",   "SPDR S&P 500 ETF",        "ETF",    "인덱스"),
    ("QQQ",   "Invesco QQQ ETF",         "ETF",    "인덱스"),
    ("IWM",   "iShares Russell 2000",    "ETF",    "인덱스"),
    ("DIA",   "SPDR Dow Jones",          "ETF",    "인덱스"),
    ("VTI",   "Vanguard Total Market",   "ETF",    "인덱스"),
    ("VOO",   "Vanguard S&P 500",        "ETF",    "인덱스"),
    ("ARKK",  "ARK Innovation ETF",      "ETF",    "테크"),
    ("SOXX",  "iShares Semiconductor",   "ETF",    "반도체"),
    ("GLD",   "SPDR Gold Shares",        "ETF",    "금"),
    ("SLV",   "iShares Silver Trust",    "ETF",    "은"),
    ("TLT",   "iShares 20Y Treasury",    "ETF",    "채권"),
    ("HYG",   "iShares HY Bond",         "ETF",    "채권"),
    ("USO",   "United States Oil Fund",  "ETF",    "에너지"),
    ("SOXS",  "Direxion Semi Bear 3X",   "ETF",    "인버스"),
    ("SOXL",  "Direxion Semi Bull 3X",   "ETF",    "레버리지"),
    # ── 글로벌 지수 ───────────────────────────────────────────
    ("^KS11",  "KOSPI 지수",             "INDEX",  "인덱스"),
    ("^KQ11",  "KOSDAQ 지수",            "INDEX",  "인덱스"),
    ("^GSPC",  "S&P 500",                "INDEX",  "인덱스"),
    ("^IXIC",  "NASDAQ 종합",            "INDEX",  "인덱스"),
    ("^DJI",   "다우존스",               "INDEX",  "인덱스"),
    ("^RUT",   "Russell 2000",           "INDEX",  "인덱스"),
    ("^VIX",   "VIX 공포지수",           "INDEX",  "변동성"),
    ("^N225",  "닛케이 225",             "INDEX",  "인덱스"),
    ("^HSI",   "항셍지수",               "INDEX",  "인덱스"),
    ("^FTSE",  "영국 FTSE 100",          "INDEX",  "인덱스"),
]

# ── KRX 동적 DB (pykrx로 로드) ────────────────────────────────
_krx_db: List[tuple] = []
_krx_loaded: bool = False
_krx_loading: bool = False
_krx_load_lock = threading.Lock()
_krx_callbacks: List[Callable] = []   # 로드 완료 콜백


# ─────────────────────────────────────────────────────────────
# KRX 종목 로드 (pykrx + JSON 캐시)
# ─────────────────────────────────────────────────────────────

def _load_krx_from_cache() -> List[tuple]:
    """캐시 파일에서 KRX 종목 로드"""
    try:
        if _CACHE_FILE.exists():
            age = time.time() - _CACHE_FILE.stat().st_mtime
            if age < _CACHE_TTL:
                with open(_CACHE_FILE, encoding="utf-8") as f:
                    data = json.load(f)
                rows = [tuple(r) for r in data if len(r) == 4]
                if rows:
                    logger.info(f"KRX 캐시 로드: {len(rows)}개 종목")
                    return rows
    except Exception as e:
        logger.warning(f"KRX 캐시 로드 실패: {e}")
    return []


def _save_krx_cache(rows: List[tuple]):
    """KRX 종목 목록을 캐시 파일에 저장"""
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump([list(r) for r in rows], f, ensure_ascii=False)
        logger.info(f"KRX 캐시 저장: {len(rows)}개 종목")
    except Exception as e:
        logger.warning(f"KRX 캐시 저장 실패: {e}")


def _fetch_krx_via_pykrx() -> List[tuple]:
    """
    pykrx StockTicker 싱글톤으로 전체 KRX 상장 종목 조회
    (네트워크 호출 없이 KRX 서버에서 상장종목 목록 로드)

    시장 코드 매핑:
      STK → KOSPI  (.KS suffix)
      KSQ → KOSDAQ (.KQ suffix)
      KNX → KONEX  (.KQ suffix, yfinance 통일)
    """
    try:
        # pykrx 내부 싱글톤: KRX 전체 상장 종목 로드 (날짜 무관)
        from pykrx.website.krx.market.ticker import StockTicker  # type: ignore
    except ImportError:
        logger.warning("pykrx 미설치 — 정적 목록만 사용 (pip install pykrx)")
        return []

    mkt_map = {
        "STK": ("KOSPI",  ".KS"),
        "KSQ": ("KOSDAQ", ".KQ"),
        "KNX": ("KONEX",  ".KQ"),
    }

    rows: List[tuple] = []
    try:
        st = StockTicker()
        df = st.listed  # DataFrame: index=6자리코드, cols=[종목, ISIN, 시장]

        for code, row in df.iterrows():
            name  = str(row["종목"]).strip()
            mkt_k = str(row["시장"]).strip()
            if not name or not mkt_k:
                continue
            mkt_label, suffix = mkt_map.get(mkt_k, ("KOSPI", ".KS"))
            rows.append((f"{code}{suffix}", name, mkt_label, ""))

    except Exception as e:
        logger.warning(f"StockTicker 로드 실패: {e}")
        return []

    # ETF 목록도 추가
    try:
        from pykrx.website.krx.etx.ticker import EtfTicker  # type: ignore
        et = EtfTicker()
        existing = {r[0] for r in rows}
        for code, row in et.listed.iterrows():
            ticker_ks = f"{code}.KS"
            if ticker_ks in existing:
                continue
            name = str(row.get("종목", "")).strip()
            if name:
                rows.append((ticker_ks, name, "ETF", "인덱스"))
    except Exception:
        pass  # ETF 실패해도 주식 목록은 정상 반환

    logger.info(f"pykrx 전체 종목 조회 완료: {len(rows)}개")
    return rows


def _load_krx_worker(callback: Optional[Callable] = None):
    """백그라운드 KRX 로드 워커"""
    global _krx_db, _krx_loaded, _krx_loading

    try:
        # 1. 캐시 시도
        rows = _load_krx_from_cache()

        # 2. 캐시 없으면 pykrx 실시간 조회
        if not rows:
            logger.info("KRX 종목 실시간 조회 중...")
            rows = _fetch_krx_via_pykrx()
            if rows:
                _save_krx_cache(rows)

        # 상태 갱신 + 콜백 목록을 lock 안에서 원자적으로 추출
        with _krx_load_lock:
            _krx_db = rows
            _krx_loaded = True
            _krx_loading = False
            cbs = list(_krx_callbacks)   # lock 안에서 복사
            _krx_callbacks.clear()       # lock 안에서 소비

        # 콜백은 lock 밖에서 실행 (GUI 콜백이 lock 재진입 시 교착 방지)
        count = len(rows)
        for cb in cbs:
            try:
                cb(count)
            except Exception as e:
                logger.debug(f"KRX 콜백 오류 (무시됨): {e}")

        if callback:
            try:
                callback(count)
            except Exception as e:
                logger.debug(f"KRX 완료 콜백 오류: {e}")

    except Exception as e:
        logger.error(f"KRX 로드 오류: {e}")
        with _krx_load_lock:
            _krx_loading = False
            _krx_loaded = True  # 실패해도 완료 표시


def load_krx_async(callback: Optional[Callable[[int], None]] = None):
    """
    백그라운드에서 KRX 전체 종목 로드
    callback(count: int) — 로드 완료 시 호출
    """
    global _krx_loading

    with _krx_load_lock:
        if _krx_loaded:
            if callback:
                try:
                    callback(len(_krx_db))
                except Exception:
                    pass
            return
        if _krx_loading:
            if callback:
                _krx_callbacks.append(callback)
            return
        _krx_loading = True
        if callback:
            _krx_callbacks.append(callback)

    t = threading.Thread(target=_load_krx_worker, daemon=True)
    t.start()


def refresh_krx(callback: Optional[Callable[[int], None]] = None):
    """
    캐시를 무효화하고 KRX 종목 강제 갱신
    """
    global _krx_loaded, _krx_loading, _krx_db
    try:
        if _CACHE_FILE.exists():
            _CACHE_FILE.unlink()
    except Exception:
        pass
    with _krx_load_lock:
        _krx_db = []
        _krx_loaded = False
        _krx_loading = False
    load_krx_async(callback)


def is_krx_loaded() -> bool:
    return _krx_loaded


def get_krx_count() -> int:
    return len(_krx_db)


# ─────────────────────────────────────────────────────────────
# 전체 DB 접근
# ─────────────────────────────────────────────────────────────

def _get_full_db() -> List[tuple]:
    """정적 + KRX 동적 목록 반환 (중복 제거)"""
    # KRX DB에 있는 ticker set
    krx_tickers = {r[0] for r in _krx_db}
    # 정적 목록 중 KRX에 없는 것만 (US/ETF/INDEX)
    static_only = [r for r in _STATIC_DB if r[0] not in krx_tickers]
    return _krx_db + static_only


def _make_row_dict(row: tuple) -> Dict:
    return {
        "ticker": row[0],
        "name":   row[1],
        "market": row[2],
        "sector": row[3],
    }


# ─────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────

MARKETS = ["전체", "KOSPI", "KOSDAQ", "KONEX", "ETF", "NASDAQ", "NYSE", "INDEX"]


def search(query: str, market: str = "전체") -> List[Dict]:
    """
    종목 검색 (코드 / 이름 / 섹터 부분 매칭)
    - KRX 로드 전이면 정적 목록 + 이미 로드된 KRX 데이터로 검색
    - 검색어 없으면 전체 반환
    """
    q = query.strip().upper()
    results = []
    seen: set = set()

    for row in _get_full_db():
        ticker, name, mkt, sector = row

        if market != "전체" and mkt.upper() != market.upper():
            continue

        if q and not (
            q in ticker.upper() or
            q in name.upper() or
            q in sector.upper()
        ):
            continue

        if ticker in seen:
            continue
        seen.add(ticker)
        results.append(_make_row_dict(row))

    return results


def get_info(ticker: str) -> Optional[Dict]:
    """단일 종목 정보 반환"""
    t = ticker.strip()
    for row in _get_full_db():
        if row[0] == t:
            return _make_row_dict(row)
    return None


def get_name(ticker: str) -> str:
    """종목코드 → 종목명 (없으면 ticker 반환)"""
    info = get_info(ticker)
    return info["name"] if info else ticker


# ─────────────────────────────────────────────────────────────
# yfinance 실시간 유효성 검사
# ─────────────────────────────────────────────────────────────

def validate_ticker_yfinance(ticker: str, timeout: float = 8.0) -> Dict:
    """
    yfinance로 실시간 종목 유효성 검사
    Returns: {'valid': bool, 'name': str, 'price': float,
              'change_pct': float, 'market_cap': str, 'error': str}
    """
    result: Dict = {
        "valid": False, "name": "", "price": 0.0,
        "change_pct": 0.0, "market_cap": "—", "error": "",
    }
    try:
        import yfinance as yf  # type: ignore
        t = yf.Ticker(ticker)
        info = t.info

        price = (
            info.get("regularMarketPrice") or
            info.get("currentPrice") or
            info.get("previousClose") or 0
        )
        if price and price > 0:
            result["valid"] = True
            result["name"] = (
                info.get("longName") or
                info.get("shortName") or
                get_name(ticker)
            )
            result["price"] = float(price)
            prev = info.get("regularMarketPreviousClose",
                            info.get("previousClose", price))
            if prev and prev > 0:
                result["change_pct"] = (price - prev) / prev * 100
            mc = info.get("marketCap", 0)
            if mc:
                if mc >= 1e12:
                    result["market_cap"] = f"{mc/1e12:.1f}조"
                elif mc >= 1e8:
                    result["market_cap"] = f"{mc/1e8:.0f}억"
                else:
                    result["market_cap"] = f"{mc:,.0f}"
        else:
            result["error"] = "가격 정보 없음 (상장폐지 또는 잘못된 코드)"
    except Exception as e:
        result["error"] = str(e)[:80]
    return result


# ─────────────────────────────────────────────────────────────
# 모듈 로드 시 자동으로 KRX 백그라운드 로딩 시작
# ─────────────────────────────────────────────────────────────
def _auto_start():
    """캐시가 있으면 즉시 로드, 없으면 백그라운드에서 pykrx 조회"""
    # 캐시 즉시 동기 로드 시도 (빠름)
    rows = _load_krx_from_cache()
    if rows:
        global _krx_db, _krx_loaded
        _krx_db = rows
        _krx_loaded = True
    else:
        # 캐시 없으면 백그라운드에서 pykrx 로드
        load_krx_async()


_auto_start()
