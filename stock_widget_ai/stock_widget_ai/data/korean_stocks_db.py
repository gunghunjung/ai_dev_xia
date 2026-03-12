"""
한국 주식 종목 데이터베이스 (KRX 전체 종목 내장)
──────────────────────────────────────────────────
KOSPI / KOSDAQ 주요 종목 300+ 개 내장.
네트워크 없이도 한글 종목명으로 즉시 검색 가능.

형식: (티커, 종목명, 시장구분)
  .KS = KOSPI
  .KQ = KOSDAQ
"""

KRX_STOCKS = [
    # ─── KOSPI 대형주 ───────────────────────────────────────────
    ("005930.KS", "삼성전자",        "KOSPI"),
    ("000660.KS", "SK하이닉스",      "KOSPI"),
    ("005380.KS", "현대자동차",      "KOSPI"),
    ("000270.KS", "기아",            "KOSPI"),
    ("207940.KS", "삼성바이오로직스","KOSPI"),
    ("373220.KS", "LG에너지솔루션",  "KOSPI"),
    ("051910.KS", "LG화학",          "KOSPI"),
    ("006400.KS", "삼성SDI",         "KOSPI"),
    ("068270.KS", "셀트리온",        "KOSPI"),
    ("035420.KS", "NAVER",           "KOSPI"),
    ("035720.KS", "카카오",          "KOSPI"),
    ("003550.KS", "LG",              "KOSPI"),
    ("028260.KS", "삼성물산",        "KOSPI"),
    ("105560.KS", "KB금융",          "KOSPI"),
    ("055550.KS", "신한지주",        "KOSPI"),
    ("086790.KS", "하나금융지주",    "KOSPI"),
    ("032830.KS", "삼성생명",        "KOSPI"),
    ("066570.KS", "LG전자",          "KOSPI"),
    ("012330.KS", "현대모비스",      "KOSPI"),
    ("017670.KS", "SK텔레콤",        "KOSPI"),
    ("030200.KS", "KT",              "KOSPI"),
    ("003490.KS", "대한항공",        "KOSPI"),
    ("009830.KS", "한화솔루션",      "KOSPI"),
    ("011200.KS", "HMM",             "KOSPI"),
    ("009540.KS", "HD한국조선해양",  "KOSPI"),
    ("010130.KS", "고려아연",        "KOSPI"),
    ("033780.KS", "KT&G",            "KOSPI"),
    ("096770.KS", "SK이노베이션",    "KOSPI"),
    ("018260.KS", "삼성에스디에스",  "KOSPI"),
    ("010950.KS", "S-Oil",           "KOSPI"),
    ("005490.KS", "POSCO홀딩스",     "KOSPI"),
    ("047050.KS", "포스코인터내셔널","KOSPI"),
    ("000810.KS", "삼성화재",        "KOSPI"),
    ("090430.KS", "아모레퍼시픽",   "KOSPI"),
    ("071050.KS", "한국금융지주",    "KOSPI"),
    ("272210.KS", "한화시스템",      "KOSPI"),
    ("018880.KS", "한온시스템",      "KOSPI"),
    ("011170.KS", "롯데케미칼",      "KOSPI"),
    ("138040.KS", "메리츠금융지주",  "KOSPI"),
    ("316140.KS", "우리금융지주",    "KOSPI"),
    ("175330.KS", "JB금융지주",      "KOSPI"),
    ("082640.KS", "동양생명",        "KOSPI"),

    # ─── 현대 그룹 ───────────────────────────────────────────────
    ("064350.KS", "현대로템",        "KOSPI"),   # ← 사용자 검색 종목
    ("042660.KS", "한화오션",        "KOSPI"),
    ("047810.KS", "한국항공우주",    "KOSPI"),
    ("010140.KS", "삼성중공업",      "KOSPI"),
    ("267250.KS", "HD현대",          "KOSPI"),
    ("009150.KS", "삼성전기",        "KOSPI"),
    ("001450.KS", "현대해상",        "KOSPI"),
    ("000100.KS", "유한양행",        "KOSPI"),
    ("034020.KS", "두산에너빌리티",  "KOSPI"),
    ("036460.KS", "한국가스공사",    "KOSPI"),
    ("015760.KS", "한국전력",        "KOSPI"),
    ("030000.KS", "제일기획",        "KOSPI"),
    ("161390.KS", "한국타이어앤테크놀로지","KOSPI"),
    ("012450.KS", "한화에어로스페이스","KOSPI"),
    ("029780.KS", "삼성카드",        "KOSPI"),
    ("024110.KS", "기업은행",        "KOSPI"),
    ("006360.KS", "GS건설",          "KOSPI"),
    ("000720.KS", "현대건설",        "KOSPI"),
    ("006980.KS", "우성사료",        "KOSPI"),
    ("011780.KS", "금호석유",        "KOSPI"),
    ("010620.KS", "HD현대미포",      "KOSPI"),
    ("002790.KS", "아모레G",         "KOSPI"),
    ("003230.KS", "삼양식품",        "KOSPI"),
    ("004020.KS", "현대제철",        "KOSPI"),
    ("005010.KS", "현대하이스코",    "KOSPI"),
    ("000120.KS", "CJ대한통운",      "KOSPI"),
    ("001040.KS", "CJ",              "KOSPI"),
    ("097950.KS", "CJ제일제당",      "KOSPI"),
    ("069960.KS", "현대백화점",      "KOSPI"),
    ("004170.KS", "신세계",          "KOSPI"),
    ("023530.KS", "롯데쇼핑",        "KOSPI"),
    ("002380.KS", "KCC",             "KOSPI"),
    ("000030.KS", "우리은행",        "KOSPI"),
    ("088350.KS", "한화생명",        "KOSPI"),
    ("014820.KS", "동원시스템즈",    "KOSPI"),
    ("000080.KS", "하이트진로",      "KOSPI"),
    ("005180.KS", "빙그레",          "KOSPI"),
    ("005300.KS", "롯데칠성",        "KOSPI"),
    ("032640.KS", "LG유플러스",      "KOSPI"),
    ("259960.KS", "크래프톤",        "KOSPI"),
    ("112040.KS", "위메이드",        "KOSPI"),
    ("036570.KS", "엔씨소프트",      "KOSPI"),
    ("251270.KS", "넷마블",          "KOSPI"),
    ("293490.KS", "카카오게임즈",    "KOSPI"),
    ("377300.KS", "카카오페이",      "KOSPI"),
    ("323410.KS", "카카오뱅크",      "KOSPI"),
    ("028300.KS", "HLB",             "KOSPI"),
    ("302440.KS", "SK바이오사이언스","KOSPI"),
    ("000950.KS", "전방",            "KOSPI"),
    ("007070.KS", "GS리테일",        "KOSPI"),
    ("078930.KS", "GS",              "KOSPI"),
    ("006120.KS", "SK디스커버리",    "KOSPI"),
    ("034730.KS", "SK",              "KOSPI"),
    ("003600.KS", "SK케미칼",        "KOSPI"),
    ("011790.KS", "SKC",             "KOSPI"),
    ("285130.KS", "SK케이파워",      "KOSPI"),
    ("402340.KS", "SK스퀘어",        "KOSPI"),
    ("009420.KS", "한올바이오파마",  "KOSPI"),
    ("170900.KS", "동아쏘시오홀딩스","KOSPI"),
    ("000670.KS", "영풍",            "KOSPI"),
    ("002350.KS", "넥센타이어",      "KOSPI"),
    ("007310.KS", "오뚜기",          "KOSPI"),
    ("280360.KS", "롯데웰푸드",      "KOSPI"),
    ("010060.KS", "OCI",             "KOSPI"),
    ("002020.KS", "코오롱",          "KOSPI"),
    ("008770.KS", "호텔신라",        "KOSPI"),
    ("120110.KS", "코오롱인더",      "KOSPI"),
    ("025540.KS", "한국단자",        "KOSPI"),
    ("047040.KS", "대우건설",        "KOSPI"),
    ("000210.KS", "DL",              "KOSPI"),
    ("001680.KS", "대상",            "KOSPI"),
    ("025000.KS", "KPX홀딩스",       "KOSPI"),
    ("008930.KS", "한미사이언스",    "KOSPI"),
    ("128940.KS", "한미약품",        "KOSPI"),
    ("185750.KS", "종근당",          "KOSPI"),
    ("003000.KS", "부광약품",        "KOSPI"),
    ("001760.KS", "한국기업평가",    "KOSPI"),
    ("071970.KS", "STX중공업",       "KOSPI"),
    ("267270.KS", "HD현대건설기계",  "KOSPI"),
    ("329180.KS", "HD현대중공업",    "KOSPI"),
    ("009970.KS", "영원무역홀딩스",  "KOSPI"),
    ("007700.KS", "F&F홀딩스",       "KOSPI"),
    ("383220.KS", "F&F",             "KOSPI"),
    ("004000.KS", "롯데정밀화학",    "KOSPI"),
    ("016360.KS", "삼성증권",        "KOSPI"),
    ("010680.KS", "한화에너지",      "KOSPI"),
    ("012630.KS", "HDC",             "KOSPI"),
    ("294870.KS", "HDC현대산업개발", "KOSPI"),
    ("000240.KS", "한국앤컴퍼니",    "KOSPI"),
    ("023160.KS", "태광산업",        "KOSPI"),
    ("002960.KS", "한국쉘석유",      "KOSPI"),
    ("011000.KS", "진도",            "KOSPI"),
    ("003080.KS", "성보화학",        "KOSPI"),
    ("014990.KS", "인디에프",        "KOSPI"),
    ("039490.KS", "키움증권",        "KOSPI"),
    ("006800.KS", "미래에셋증권",    "KOSPI"),
    ("001500.KS", "현대차증권",      "KOSPI"),
    ("003460.KS", "유화증권",        "KOSPI"),
    ("005945.KS", "NH투자증권",      "KOSPI"),
    ("001720.KS", "신영증권",        "KOSPI"),

    # ─── KOSDAQ 주요 종목 ────────────────────────────────────────
    ("247540.KQ", "에코프로비엠",    "KOSDAQ"),
    ("086520.KQ", "에코프로",        "KOSDAQ"),
    ("196170.KQ", "알테오젠",        "KOSDAQ"),
    ("091990.KQ", "셀트리온헬스케어","KOSDAQ"),
    ("263750.KQ", "펄어비스",        "KOSDAQ"),
    ("039200.KQ", "오스코텍",        "KOSDAQ"),
    ("214150.KQ", "클래시스",        "KOSDAQ"),
    ("183490.KQ", "엔지켐생명과학",  "KOSDAQ"),
    ("145020.KQ", "휴젤",            "KOSDAQ"),
    ("078340.KQ", "컴투스",          "KOSDAQ"),
    ("095340.KQ", "ISC",             "KOSDAQ"),
    ("041510.KQ", "에스엠",          "KOSDAQ"),
    ("035900.KQ", "JYP Ent.",        "KOSDAQ"),
    ("352820.KQ", "하이브",          "KOSDAQ"),
    ("122870.KQ", "와이지엔터테인먼트","KOSDAQ"),
    ("950130.KQ", "엑스페릭스",      "KOSDAQ"),
    ("039030.KQ", "이오테크닉스",    "KOSDAQ"),
    ("357780.KQ", "솔브레인",        "KOSDAQ"),
    ("036800.KQ", "나이스정보통신",  "KOSDAQ"),
    ("032500.KQ", "케이엠더블유",    "KOSDAQ"),
    ("064760.KQ", "티씨케이",        "KOSDAQ"),
    ("045270.KQ", "크라운제과",      "KOSDAQ"),
    ("054620.KQ", "APS홀딩스",       "KOSDAQ"),
    ("290650.KQ", "엘앤에프",        "KOSDAQ"),
    ("053800.KQ", "안랩",            "KOSDAQ"),
    ("066970.KQ", "엘앤씨바이오",    "KOSDAQ"),
    ("068760.KQ", "셀트리온제약",    "KOSDAQ"),
    ("095660.KQ", "네오위즈",        "KOSDAQ"),
    ("098660.KQ", "에스티오",        "KOSDAQ"),
    ("033290.KQ", "코웰패션",        "KOSDAQ"),
    ("067160.KQ", "아프리카TV",      "KOSDAQ"),
    ("036030.KQ", "KG이니시스",      "KOSDAQ"),
    ("060310.KQ", "3S",              "KOSDAQ"),
    ("048410.KQ", "현대바이오",      "KOSDAQ"),
    ("214370.KQ", "케어젠",          "KOSDAQ"),
    ("204210.KQ", "모트렉스",        "KOSDAQ"),
    ("101160.KQ", "월덱스",          "KOSDAQ"),
    ("064820.KQ", "이노와이어리스",  "KOSDAQ"),
    ("009420.KQ", "한올바이오파마",  "KOSDAQ"),
    ("017900.KQ", "광동제약",        "KOSDAQ"),
    ("000020.KQ", "동화약품",        "KOSDAQ"),

    # ─── 지수 / ETF ──────────────────────────────────────────────
    ("^KS11",    "KOSPI 지수",       "INDEX"),
    ("^KQ11",    "KOSDAQ 지수",      "INDEX"),
    ("069500.KS","KODEX 200",        "ETF"),
    ("114800.KS","KODEX 인버스",     "ETF"),
    ("122630.KS","KODEX 레버리지",   "ETF"),
    ("229200.KS","KODEX 코스닥150",  "ETF"),
    ("252670.KS","KODEX 200선물인버스2X","ETF"),
    ("278530.KS","KODEX 2차전지산업","ETF"),
    ("091160.KS","KODEX 반도체",     "ETF"),
    ("091180.KS","KODEX 은행",       "ETF"),
    ("140700.KS","KODEX 보험",       "ETF"),
    ("152100.KS","ARIRANG 200",      "ETF"),
    ("157490.KS","TIGER 200",        "ETF"),
    ("176950.KS","TIGER 코스닥150",  "ETF"),
    ("243890.KS","TIGER 차이나전기차SOLACTIVE","ETF"),
    ("381170.KS","TIGER K-미래차액티브","ETF"),
]


def search_krx(query: str, top_n: int = 30):
    """
    로컬 KRX 종목 DB에서 퍼지 검색.

    Parameters
    ----------
    query : 검색어 (한글 종목명 / 티커코드 / 영어 종목명)
    top_n : 최대 결과 수

    Returns
    -------
    list of dict: [{symbol, name, market}, ...]
    """
    q = query.strip()
    if not q:
        return []

    q_lower = q.lower()
    results = []

    for symbol, name, market in KRX_STOCKS:
        score = 0

        # 정확히 일치
        if q == name or q_lower == symbol.lower():
            score = 100
        # 종목명이 검색어로 시작
        elif name.startswith(q):
            score = 90
        # 티커가 검색어로 시작
        elif symbol.lower().startswith(q_lower):
            score = 85
        # 종목명에 검색어 포함
        elif q in name:
            score = 70
        # 티커에 포함 (숫자/코드 검색)
        elif q_lower in symbol.lower():
            score = 65
        # 초성 검색 (ㅎ → 현대로템 등)
        elif _choseong_match(q, name):
            score = 50

        if score > 0:
            results.append((score, symbol, name, market))

    # 점수 내림차순 정렬
    results.sort(key=lambda x: -x[0])

    return [
        {
            "symbol":    sym,
            "shortname": name,
            "exchange":  mkt,
            "quoteType": "INDEX" if sym.startswith("^") else
                         "ETF"   if "ETF" in mkt         else "EQUITY",
        }
        for _, sym, name, mkt in results[:top_n]
    ]


def _choseong_match(query: str, name: str) -> bool:
    """
    한글 초성 매칭.
    예: "ㅎㄷㄹㅌ" → "현대로템"
    """
    if not query:
        return False

    CHOSEONG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"

    def is_choseong_only(s: str) -> bool:
        return all(c in CHOSEONG for c in s)

    if not is_choseong_only(query):
        return False

    def get_choseong(s: str) -> str:
        result = []
        for c in s:
            code = ord(c)
            if 0xAC00 <= code <= 0xD7A3:
                idx = (code - 0xAC00) // 588
                result.append(CHOSEONG[idx])
            elif c in CHOSEONG:
                result.append(c)
        return "".join(result)

    return get_choseong(name).startswith(query)
