#!/usr/bin/env python3
"""
주식 미니 위젯 — 최상위창 (Always-on-Top)
설치: pip install yfinance pykrx requests pandas lxml
"""
import tkinter as tk
import threading, time, json, os, logging, re, traceback

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ── 로그 파일 설정 ────────────────────────────────────────────────
_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_widget_debug.log")
_fh = logging.FileHandler(_LOG_FILE, encoding="utf-8", mode="w")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_log = logging.getLogger("stock_widget")
_log.setLevel(logging.DEBUG)
_log.addHandler(_fh)
_log.addHandler(logging.StreamHandler())
_log.info(f"=== stock_widget 시작 ===  로그파일: {_LOG_FILE}")

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from pykrx import stock as krx
except ImportError:
    krx = None

# ── 설정 ────────────────────────────────────────────────────────
SAVE_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_watch.json")
REFRESH_MS = 10_000
WIDTH_L    = 300   # 왼쪽 종목 패널
WIDTH_R    = 320   # 오른쪽 뉴스 패널
TOTAL_W    = WIDTH_L + WIDTH_R
ROW_H      = 26   # 종목명/가격 행
SUB_H      = 22   # 평단/수량/평가 행
STOCK_H    = ROW_H + SUB_H
HANDLE_H   = 22
SEARCH_H   = 38
DROP_ROW   = 24
DROP_MAX   = 8

BG        = "#0f0f0f"
BG_A      = "#181818"
BG_B      = "#131313"
BG_HDL    = "#090909"
BG_DROP   = "#1a1a1a"
BG_ENT    = "#202020"
BG_SEL    = "#2a3a4a"
BG_NEWS   = "#0d1117"
FG        = "#dedede"
FG_DIM    = "#404040"
FG_MED    = "#777777"
UP        = "#d44f4f"
DN        = "#4a90d9"
NC        = "#888888"
NEWS_FG   = "#8ab4cc"
NEWS_FG2  = "#c9d1d9"
FONT      = ("Malgun Gothic", 9)
FONTB     = ("Malgun Gothic", 9, "bold")
FONTS     = ("Malgun Gothic", 8)
FONTSS    = ("Malgun Gothic", 7)

# ── 뉴스 감성 분석 — KR-FinBert-SC + 확장 키워드 혼합 ─────────────
_POS_KW = {
    # 수주·계약
    '수주','수주잔고','수주총액','대형수주','수주성공','수주확보',
    '계약','계약체결','계약확정','장기계약','독점계약','공급계약',
    '체결','MOU','MOA','협약','협정','업무협약','전략적협약',
    '공급','납품','독점공급','우선공급','장기공급',
    # 실적·수익
    '흑자','흑자전환','흑자기조','실적호조','실적개선','최대실적',
    '어닝서프라이즈','컨센서스상회','기대이상','예상상회',
    '영업이익','순이익','영업익','이익','수익','수익성',
    '영업이익증가','매출증가','판매증가','판매호조','수요증가',
    '분기최고','사상최대','역대최고','신기록','최고치','최대치',
    # 주가·투자의견
    '상승','급등','강세','신고가','52주신고가','역대최고가',
    '반등','반등세','회복','회복세','상승세',
    '목표가상향','투자의견상향','매수의견','강력매수','적극매수',
    '강추','유망','주목','관심','매력적','저평가',
    # 성장·확장
    '성장','성장세','고성장','확대','증가','증가세',
    '증설','확장','신공장','생산능력확대','시장확대','점유율확대',
    '해외진출','글로벌','수출','수출확대','수출증가','수출호조',
    '신사업','신제품','신서비스','신기술','혁신','기술혁신',
    # 투자·M&A
    '투자유치','지분투자','전략적투자','외국인투자','대규모투자',
    '합병','인수','M&A','지분취득','기업가치상승',
    '상장','IPO','코스피편입','코스닥편입','유가증권편입',
    # 주주환원
    '배당','배당확대','배당금증가','특별배당','자사주매입','주주환원','자사주소각',
    # 개발·연구
    '개발','개발성공','개발완료','연구','연구성과','기술개발',
    '특허','특허취득','특허등록','특허획득','지식재산',
    '임상성공','임상통과','신약승인','식약처승인','허가','인허가',
    # 수상·선정
    '수상','표창','선정','인증','우수기업','기술력인정','품질인증',
    # 기타 긍정
    '수혜','수혜주','협력','협업','파트너십','전략적파트너',
    '개선','호전','강화','긍정','긍정적','낙관','낙관적',
    '청신호','기대','호재','호조','양호','확정','안정','안정적',
    '선도','독보적','경쟁우위','차별화','글로벌선두',
}
_NEG_KW = {
    # 실적·손실
    '적자','적자전환','적자지속','영업손실','순손실','당기순손실',
    '실적부진','어닝쇼크','컨센서스하회','기대이하','예상하회',
    '매출감소','판매부진','수요감소','영업손실확대','수익성악화',
    # 주가·투자의견
    '하락','급락','폭락','약세','저점','하락세','하향세',
    '신저가','52주신저가','역대최저가',
    '목표가하향','투자의견하향','매도의견','비중축소','중립하향',
    '공매도','외인매도','기관매도',
    # 법적·제재
    '소송','피소','기소','고발','고소','피고','법적분쟁',
    '제재','행정처분','시정명령','과징금','벌금','과태료','제재금',
    '검찰','수사','조사','압수수색','경찰수사','금융당국조사',
    '횡령','배임','사기','불법','위반','탈세','탈루',
    '주가조작','불공정거래','내부자거래','시세조종',
    '패소','패배','유죄','판결','손해배상','징역',
    # 리콜·결함
    '리콜','반품','제품결함','품질문제','불량','클레임','결함',
    # 파산·위기
    '파산','법정관리','워크아웃','부도','디폴트','채무불이행',
    '자본잠식','감자','완전자본잠식','분식회계',
    '상장폐지','관리종목','투자경고','거래정지','불성실공시',
    '자금난','유동성위기','현금부족','재무위기',
    '부채','차입금증가','이자부담','신용강등','신용등급강등',
    # 구조조정
    '구조조정','감원','해고','인력감축','구조개편','희망퇴직',
    '공장폐쇄','사업철수','사업중단','서비스종료','사업포기',
    '계약해지','수주취소','주문취소','납품취소',
    # 경기·업황
    '불황','침체','경기침체','경기둔화','위축','경기악화',
    '역성장','감소','감소세','부진','저조','하락','축소',
    '경쟁심화','점유율하락','시장잠식','시장축소','점유율감소',
    # 비용·압박
    '원가상승','비용증가','마진압박','수익성악화','원자재상승',
    '금리인상','금리부담','이자비용증가',
    # 악재·위험
    '악재','리스크','위기','위험','우려','불안','불확실','불확실성',
    '경고','주의','위협','경계','리스크확대',
    '부담','부정','부정적','비관','비관적','암울',
    # 사고·환경
    '사고','화재','폭발','안전사고','인명사고','폭발사고',
    '환경오염','환경위반','환경제재','환경과징금',
    # 신뢰·경영
    '허위공시','정정공시','감사의견거절','한정의견',
    '신뢰하락','브랜드훼손','이미지손상','평판악화',
    '경영위기','리더십위기','대표사임','경영진교체','내분',
}

_bert_pipe   = None
_bert_ready  = False
_bert_failed = False
_BERT_LOCAL  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "models", "KR-FinBert-SC")

def _load_bert_model():
    global _bert_pipe, _bert_ready, _bert_failed
    try:
        from transformers import pipeline
        src = _BERT_LOCAL if os.path.isfile(os.path.join(_BERT_LOCAL, "config.json")) \
              else "snunlp/KR-FinBert-SC"
        _log.info(f"KR-FinBert-SC 로딩: {src}")
        _bert_pipe = pipeline("text-classification", model=src,
                              device=-1, truncation=True, max_length=512)
        _bert_ready = True
        _log.info("KR-FinBert-SC 로드 완료")
    except Exception as e:
        _bert_failed = True
        _log.warning(f"KR-FinBert-SC 로드 실패 (키워드 폴백): {e}")

def _parse_pubdate(s: str) -> str:
    """RSS pubDate → 'HH:MM' 변환"""
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(s).strftime("%H:%M")
    except Exception:
        return ""

def _kw_sentiment(text: str) -> str:
    pos = sum(1 for kw in _POS_KW if kw in text)
    neg = sum(1 for kw in _NEG_KW if kw in text)
    if pos > neg: return 'pos'
    if neg > pos: return 'neg'
    return 'neu'

def _news_sentiment(text: str) -> str:
    """BERT + 키워드 혼합 감성 분류
    - 둘 다 동의 → 그 결과
    - BERT 신뢰도 ≥0.80 → BERT 우선
    - BERT 중립 + 키워드 명확 → 키워드
    - 그 외 → BERT
    """
    kw = _kw_sentiment(text)
    if _bert_ready and _bert_pipe is not None:
        try:
            res   = _bert_pipe(text[:512])[0]
            lbl   = res["label"].lower()
            score = res["score"]
            if any(x in lbl for x in ("pos", "positive", "label_2")):   bert = 'pos'
            elif any(x in lbl for x in ("neg", "negative", "label_0")): bert = 'neg'
            else:                                                          bert = 'neu'
            if bert == kw:       return bert          # 합의
            if score >= 0.80:    return bert          # BERT 고신뢰
            if bert == 'neu':    return kw            # BERT 중립 → 키워드
            return bert                               # 나머지 BERT
        except Exception:
            pass
    return kw


# ════════════════════════════════════════════════════════════════
class StockWidget:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self._dx = self._dy = 0
        self._watchlist: list[dict] = []
        self._name_map:  dict = {}
        self._code_map:  dict = {}
        self._map_ready      = False
        self._dropdown       = None
        self._search_timer   = None
        self._news_cache:      dict = {}   # code6 → list[dict]
        self._sentiment_cache: dict = {}   # headline → 'pos'|'neg'|'neu'
        self._stock_sent:      dict = {}   # code6 → 'pos'|'neg'|'neu'
        self._selected_code:   str  = ""   # 뉴스 패널에 표시 중인 종목코드
        self._win_w          = TOTAL_W
        self._win_h          = 0
        self._rx = self._ry = self._rw0 = self._rh0 = 0
        self._resize_dir     = "both"

        # BERT 백그라운드 로드
        if not _bert_ready and not _bert_failed:
            threading.Thread(target=self._load_bert_bg, daemon=True).start()

        self._build_ui()
        self._place_window()
        self._load()
        self._rebuild_list()
        self._init_data()

        self.root.mainloop()

    def _load_bert_bg(self):
        _load_bert_model()
        if _bert_ready:
            self._sentiment_cache.clear()
            self._stock_sent.clear()
            if self._selected_code:
                code = self._selected_code
                name = self._reverse_name(code) or code
                self.root.after(0, lambda: self._show_news(code, name))
            self.root.after(0, self._rebuild_list)

    def _cached_sentiment(self, title: str) -> str:
        if title not in self._sentiment_cache:
            self._sentiment_cache[title] = _news_sentiment(title)
        return self._sentiment_cache[title]

    def _compute_stock_sent(self, code6: str):
        """(pos_count, neg_count, overall_sentiment) 튜플 반환"""
        if code6 in self._stock_sent:
            return self._stock_sent[code6]
        arts = self._news_cache.get(code6) or []
        if not arts:
            result = (0, 0, 'neu')
            self._stock_sent[code6] = result
            return result
        counts = {'pos': 0, 'neg': 0, 'neu': 0}
        for a in arts:
            counts[self._cached_sentiment(a["title"])] += 1
        overall = 'pos' if counts['pos'] > counts['neg'] else \
                  ('neg' if counts['neg'] > counts['pos'] else 'neu')
        result = (counts['pos'], counts['neg'], overall)
        self._stock_sent[code6] = result
        return result

    # ── 창 초기 위치 ─────────────────────────────────────────────
    def _place_window(self):
        sw = self.root.winfo_screenwidth()
        self.root.geometry(
            f"{TOTAL_W}x{HANDLE_H + SEARCH_H + STOCK_H + 22}+{sw - TOTAL_W - 12}+40"
        )

    # ════════════════════════════════════════════════════════════
    #  UI 구성
    # ════════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── 드래그 핸들 (상단) ────────────────────────────────────
        hdl = tk.Frame(self.root, bg=BG_HDL, height=HANDLE_H)
        hdl.pack(fill="x", side="top")
        hdl.pack_propagate(False)
        hdl.bind("<ButtonPress-1>", self._drag_start)
        hdl.bind("<B1-Motion>",     self._drag_move)
        tk.Label(hdl, text="  주식 시세", bg=BG_HDL, fg=FG_MED, font=FONTS).pack(side="left", pady=3)
        tk.Button(hdl, text="✕", bg=BG_HDL, fg=FG_MED, font=FONTS, bd=0, padx=5,
                  activebackground="#222", cursor="hand2",
                  command=self.root.destroy).pack(side="right", pady=2, padx=2)

        # ── 리사이즈 그립 — 하단 (main보다 먼저 pack해야 표시됨) ──
        grip_b = tk.Frame(self.root, bg="#1c2430", height=10,
                          cursor="sb_v_double_arrow")
        grip_b.pack(fill="x", side="bottom")
        tk.Label(grip_b, text="⠿", bg="#1c2430", fg="#334455",
                 font=("Malgun Gothic", 6)).pack()
        grip_b.bind("<ButtonPress-1>",   lambda e: self._resize_start(e, "h"))
        grip_b.bind("<B1-Motion>",       self._resize_move)
        grip_b.bind("<ButtonRelease-1>", self._resize_end)

        # ── 메인 영역: 왼쪽(종목) + 구분선 + 우측그립 + 오른쪽(뉴스) ─
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True)

        # 왼쪽 패널 (고정 너비)
        left = tk.Frame(main, bg=BG, width=WIDTH_L)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        # 세로 구분선
        tk.Frame(main, bg="#1e1e1e", width=1).pack(side="left", fill="y")

        # ── 리사이즈 그립 — 우측 (뉴스 패널보다 먼저 side=right 로 pack) ─
        grip_r = tk.Frame(main, bg="#1c2430", width=10,
                          cursor="sb_h_double_arrow")
        grip_r.pack(side="right", fill="y")
        grip_r.bind("<ButtonPress-1>",   lambda e: self._resize_start(e, "w"))
        grip_r.bind("<B1-Motion>",       self._resize_move)
        grip_r.bind("<ButtonRelease-1>", self._resize_end)

        # 오른쪽 뉴스 패널
        self._rp = tk.Frame(main, bg=BG_NEWS)
        self._rp.pack(side="left", fill="both", expand=True)

        # ── 검색 바 (왼쪽) ───────────────────────────────────────
        sf = tk.Frame(left, bg=BG, padx=6, pady=5)
        sf.pack(fill="x")
        self._sv = tk.StringVar()
        self._entry = tk.Entry(sf, textvariable=self._sv, bg="#1e1e1e", fg=FG,
                               insertbackground=FG, relief="flat", font=FONT, bd=2)
        self._entry.pack(side="left", fill="x", expand=True, ipady=4)
        self._sv.trace_add("write", self._on_text_change)
        self._entry.bind("<Return>",   self._on_enter_key)
        self._entry.bind("<Down>",     self._drop_focus)
        self._entry.bind("<Escape>",   lambda _: self._hide_dropdown())
        self._entry.bind("<FocusIn>",  self._ph_clear)
        self._entry.bind("<FocusOut>", self._ph_restore)
        self._ph = True
        self._ph_restore()
        tk.Button(sf, text="+", bg="#252525", fg=FG, font=FONTB, relief="flat",
                  padx=7, activebackground="#383838", cursor="hand2",
                  command=self._add_from_entry).pack(side="left", padx=(4, 0))

        # ── 종목 리스트 (왼쪽) ───────────────────────────────────
        self._list_frame = tk.Frame(left, bg=BG)
        self._list_frame.pack(fill="both", expand=True)

        # ── 상태바 (왼쪽) ────────────────────────────────────────
        self._sv_status = tk.StringVar(value="초기화 중...")
        tk.Label(left, textvariable=self._sv_status,
                 bg=BG, fg=FG_DIM, font=FONTS, anchor="w").pack(fill="x", padx=6, pady=2)

        # ── 뉴스 패널 구성 ────────────────────────────────────────
        self._build_news_panel()

    def _build_news_panel(self):
        """오른쪽 뉴스 패널 초기 구성"""
        # 헤더 (검색바와 높이 맞춤)
        hdr = tk.Frame(self._rp, bg=BG_NEWS, height=SEARCH_H)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        self._news_title_var = tk.StringVar(value="  뉴스")
        tk.Label(hdr, textvariable=self._news_title_var,
                 bg=BG_NEWS, fg=FG_MED, font=FONTS, anchor="w").pack(fill="x", padx=6, pady=3)

        # 구분선
        tk.Frame(self._rp, bg="#1e2830", height=1).pack(fill="x")

        # 스크롤 가능한 뉴스 목록
        container = tk.Frame(self._rp, bg=BG_NEWS)
        container.pack(fill="both", expand=True)

        sb = tk.Scrollbar(container, orient="vertical", width=6,
                          bg="#1e1e1e", troughcolor=BG_NEWS, bd=0, relief="flat")
        self._news_canvas = tk.Canvas(container, bg=BG_NEWS,
                                      yscrollcommand=sb.set,
                                      highlightthickness=0, bd=0)
        sb.config(command=self._news_canvas.yview)
        sb.pack(side="right", fill="y")
        self._news_canvas.pack(side="left", fill="both", expand=True)

        self._news_inner = tk.Frame(self._news_canvas, bg=BG_NEWS)
        self._canvas_win = self._news_canvas.create_window(
            (0, 0), window=self._news_inner, anchor="nw")

        self._news_inner.bind("<Configure>",
            lambda e: self._news_canvas.configure(
                scrollregion=self._news_canvas.bbox("all")))
        self._news_canvas.bind("<Configure>",
            lambda e: self._news_canvas.itemconfig(
                self._canvas_win, width=e.width))

        def _wheel(e):
            self._news_canvas.yview_scroll(-1 if e.delta > 0 else 1, "units")
        self._news_canvas.bind("<MouseWheel>", _wheel)
        self._news_inner.bind("<MouseWheel>", _wheel)
        self._news_wheel = _wheel  # 뉴스 아이템 생성 시 재사용

        # 초기 안내 문구
        tk.Label(self._news_inner, text="종목에 마우스를 올리면\n뉴스가 표시됩니다",
                 bg=BG_NEWS, fg=FG_DIM, font=FONTSS, justify="center").pack(pady=20)

    def _show_news(self, code6: str, name: str):
        """오른쪽 뉴스 패널 갱신"""
        self._news_title_var.set(f"  {name}  뉴스")
        for w in self._news_inner.winfo_children():
            w.destroy()

        headlines = self._news_cache.get(code6)  # None=미조회, []=없음, [..]=있음

        if headlines is None:
            tk.Label(self._news_inner, text="불러오는 중...",
                     bg=BG_NEWS, fg=FG_DIM, font=FONTSS).pack(pady=12, padx=8)
            return

        if not headlines:
            tk.Label(self._news_inner, text="뉴스를 가져오지 못했습니다",
                     bg=BG_NEWS, fg=FG_DIM, font=FONTSS).pack(pady=12, padx=8)
            return

        import webbrowser
        for i, art in enumerate(headlines):
            title = art["title"] if isinstance(art, dict) else art
            link  = art.get("link", "")  if isinstance(art, dict) else ""
            hhmm  = art.get("date", "")  if isinstance(art, dict) else ""

            sentiment = self._cached_sentiment(title)
            fg_color  = UP if sentiment == 'pos' else (DN if sentiment == 'neg' else NEWS_FG)

            # 한 줄 포맷: "HH:MM : 기사제목"
            prefix = f"{hhmm} : " if hhmm else ""
            line   = prefix + title

            row_bg = BG_NEWS if i % 2 == 0 else "#111820"
            frm = tk.Frame(self._news_inner, bg=row_bg, cursor="hand2" if link else "")
            frm.pack(fill="x")

            lbl = tk.Label(frm, text=line,
                           bg=row_bg, fg=fg_color,
                           font=FONTSS,
                           wraplength=WIDTH_R - 16,
                           justify="left", anchor="nw",
                           padx=6, pady=4)
            lbl.pack(fill="x")

            if link:
                def _open(e, url=link):
                    webbrowser.open(url)
                lbl.bind("<Button-1>", _open)
                frm.bind("<Button-1>", _open)

            lbl.bind("<MouseWheel>", self._news_wheel)
            frm.bind("<MouseWheel>", self._news_wheel)
            if i < len(headlines) - 1:
                tk.Frame(self._news_inner, bg="#1a2030", height=1).pack(fill="x")

        self._news_canvas.yview_moveto(0)

    # ── 플레이스홀더 ─────────────────────────────────────────────
    def _ph_clear(self, _=None):
        if self._ph:
            self._ph = False
            self._entry.delete(0, "end")
            self._entry.config(fg=FG)
        else:
            # 기존 텍스트가 있을 때 클릭 → 전체 삭제 후 바로 입력 가능
            self._entry.delete(0, "end")
            self._hide_dropdown()

    def _ph_restore(self, _=None):
        self._hide_dropdown()
        if not self._sv.get():
            self._ph = True
            self._entry.insert(0, "종목명 또는 코드 검색")
            self._entry.config(fg=FG_DIM)

    # ── 드래그 ───────────────────────────────────────────────────
    def _drag_start(self, e):
        self._dx = e.x_root - self.root.winfo_x()
        self._dy = e.y_root - self.root.winfo_y()

    def _drag_move(self, e):
        self.root.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")
        self._reposition_dropdown()

    # ── 리사이즈 ─────────────────────────────────────────────────
    def _resize_start(self, e, direction="both"):
        self._resize_dir = direction
        self._rx   = e.x_root
        self._ry   = e.y_root
        self._rw0  = self.root.winfo_width()
        self._rh0  = self.root.winfo_height()

    def _resize_move(self, e):
        dw = e.x_root - self._rx
        dh = e.y_root - self._ry
        d  = getattr(self, "_resize_dir", "both")
        nw = max(WIDTH_L + 120, self._rw0 + dw) if d in ("w", "both") else self._rw0
        nh = max(100,            self._rh0 + dh) if d in ("h", "both") else self._rh0
        x, y = self.root.winfo_x(), self.root.winfo_y()
        self.root.geometry(f"{nw}x{nh}+{x}+{y}")
        self._win_w = nw
        self._win_h = nh

    def _resize_end(self, e):
        """드래그 끝 → wraplength 업데이트 위해 뉴스 패널 새로고침"""
        global WIDTH_R
        WIDTH_R = max(80, self._win_w - WIDTH_L - 12)  # 12 = 구분선1 + 우측그립10 + 여유1
        if self._selected_code:
            name = self._reverse_name(self._selected_code) or self._selected_code
            self._show_news(self._selected_code, name)

    # ════════════════════════════════════════════════════════════
    #  자동완성 드롭다운
    # ════════════════════════════════════════════════════════════
    def _on_text_change(self, *_):
        if self._ph:
            return
        q = self._sv.get().strip()
        if not q:
            self._hide_dropdown()
            return
        if self._name_map:
            results = self._search(q)
            if results:
                self._show_dropdown(results)
            else:
                self._hide_dropdown()
        else:
            if self._search_timer:
                self.root.after_cancel(self._search_timer)
            self._search_timer = self.root.after(
                400, lambda sq=q: threading.Thread(
                    target=self._yf_search_update, args=(sq,), daemon=True
                ).start()
            )

    def _on_enter_key(self, _=None):
        if self._ph:
            return
        q = self._sv.get().strip()
        if not q:
            return
        if self._dropdown and self._dropdown.winfo_exists():
            self._add_selected()
            return
        results = self._search(q)
        if not results:
            return
        if len(results) == 1:
            name, code, suf = results[0]
            self._add_item(code, code + suf, name)
            self._sv.set(""); self._ph_restore()
        else:
            self._show_dropdown(results)

    def _drop_focus(self, _=None):
        if self._dropdown and self._dropdown.winfo_exists():
            self._lb.focus_set()
            if self._lb.size() > 0:
                self._lb.selection_set(0); self._lb.activate(0)

    def _search(self, q: str) -> list:
        if self._name_map:
            if q.isdigit() and len(q) <= 6:
                res = [(n, c, s) for n, (c, s) in self._name_map.items() if c.startswith(q)]
                return res[:40]
            starts   = [(n, c, s) for n, (c, s) in self._name_map.items() if n.startswith(q)]
            contains = [(n, c, s) for n, (c, s) in self._name_map.items()
                        if q in n and not n.startswith(q)]
            return (starts + contains)[:40]
        return self._search_yf(q)

    def _search_yf(self, q: str) -> list:
        if not yf:
            return []
        try:
            search = yf.Search(query=q, max_results=20)
            quotes = getattr(search, "quotes", None) or []
            results = []
            seen = set()
            for item in quotes:
                symbol = item.get("symbol", "")
                if not symbol or symbol in seen:
                    continue
                if not (symbol.endswith(".KS") or symbol.endswith(".KQ")):
                    continue
                if str(item.get("quoteType", "")).lower() not in ("equity", ""):
                    continue
                code = symbol[:-3]
                suf  = symbol[-3:]
                name = self._reverse_name(code) or item.get("shortname") or item.get("longname") or code
                seen.add(symbol)
                results.append((name, code, suf))
            _log.info(f"yf.Search '{q}': {len(results)}건")
            return results[:20]
        except Exception as e:
            _log.warning(f"yf.Search '{q}' 실패: {type(e).__name__}: {e}")
            return []

    def _yf_search_update(self, q: str):
        results = self._search_yf(q)
        def _apply():
            if self._sv.get().strip() != q:
                return
            if results:
                self._show_dropdown(results)
            else:
                self._hide_dropdown()
        self.root.after(0, _apply)

    def _show_dropdown(self, results: list):
        ex = self._entry.winfo_rootx()
        ey = self._entry.winfo_rooty() + self._entry.winfo_height() + 2
        ew = self._entry.winfo_width() + 35
        dh = min(len(results), DROP_MAX) * DROP_ROW + 30

        if not self._dropdown or not self._dropdown.winfo_exists():
            self._dropdown = tk.Toplevel(self.root)
            self._dropdown.overrideredirect(True)
            self._dropdown.attributes("-topmost", True)
            self._dropdown.configure(bg=BG_DROP)
            frm = tk.Frame(self._dropdown, bg=BG_DROP)
            frm.pack(fill="both", expand=True)
            sb = tk.Scrollbar(frm, orient="vertical", width=8,
                              bg="#333", troughcolor=BG_DROP, bd=0)
            self._lb = tk.Listbox(frm, selectmode=tk.EXTENDED, bg=BG_DROP, fg=FG,
                                  selectbackground=BG_SEL, selectforeground=FG,
                                  relief="flat", font=FONTS, activestyle="none",
                                  highlightthickness=0, bd=0, yscrollcommand=sb.set)
            sb.config(command=self._lb.yview)
            sb.pack(side="right", fill="y")
            self._lb.pack(side="left", fill="both", expand=True)
            self._lb.bind("<Double-Button-1>", lambda _: self._add_selected())
            self._lb.bind("<Return>",          lambda _: self._add_selected())
            self._lb.bind("<Escape>",          lambda _: self._hide_dropdown())
            self._lb.bind("<<ListboxSelect>>", self._on_lb_sel)
            bf = tk.Frame(self._dropdown, bg="#111", pady=3)
            bf.pack(fill="x")
            tk.Button(bf, text="선택 추가", bg="#2a2a2a", fg=FG, font=FONTS,
                      relief="flat", padx=8, activebackground="#3a3a3a",
                      cursor="hand2", command=self._add_selected).pack(side="left", padx=4)
            self._drop_cnt = tk.StringVar(value="")
            tk.Label(bf, textvariable=self._drop_cnt, bg="#111",
                     fg=FG_MED, font=FONTS).pack(side="left")

        self._lb.delete(0, tk.END)
        self._drop_items = results
        for name, code, _ in results:
            self._lb.insert(tk.END, f"  {name}  ({code})")
        self._dropdown.geometry(f"{ew}x{dh}+{ex}+{ey}")
        self._drop_cnt.set("")

    def _reposition_dropdown(self):
        if not self._dropdown or not self._dropdown.winfo_exists(): return
        ex = self._entry.winfo_rootx()
        ey = self._entry.winfo_rooty() + self._entry.winfo_height() + 2
        self._dropdown.geometry(f"+{ex}+{ey}")

    def _hide_dropdown(self, _=None):
        if self._dropdown and self._dropdown.winfo_exists():
            self._dropdown.destroy()
        self._dropdown = None

    def _on_lb_sel(self, _=None):
        n = len(self._lb.curselection())
        self._drop_cnt.set(f"선택 {n}개" if n else "")

    def _add_selected(self):
        sel = self._lb.curselection() or ((0,) if self._lb.size() > 0 else ())
        for i in sel:
            if i < len(self._drop_items):
                name, code, suf = self._drop_items[i]
                self._add_item(code, code + suf, name)
        if sel:
            self._hide_dropdown()
            self._sv.set(""); self._ph_restore()

    def _add_from_entry(self, _=None):
        if self._ph: return
        q = self._sv.get().strip()
        if not q: return
        code6, ycode, name = self._resolve(q)
        if not ycode:
            self._sv_status.set(f"'{q}' 검색 실패"); return
        self._add_item(code6, ycode, name)
        self._sv.set(""); self._ph_restore()

    def _add_item(self, code6: str, ycode: str, name: str):
        if any(s["ycode"] == ycode for s in self._watchlist):
            self._sv_status.set(f"{name} 이미 추가됨"); return
        item = dict(code=code6, ycode=ycode, name=name,
                    price=None, chg=None, avg_price=None, qty=None)
        self._watchlist.append(item)
        self._rebuild_list()
        self._save()
        threading.Thread(target=self._fetch_one, args=(item,), daemon=True).start()

    # ════════════════════════════════════════════════════════════
    #  데이터
    # ════════════════════════════════════════════════════════════
    def _init_data(self):
        if krx is None:
            self._map_ready = True
            self._sv_status.set("코드 직접 입력 모드")
            self._start_refresh()
        else:
            threading.Thread(target=self._build_map, daemon=True).start()

    def _build_map(self):
        import datetime

        def _ui(msg):
            self.root.after(0, lambda m=msg: self._sv_status.set(m))

        try:
            _ui("종목 DB 로딩 중...")
            _log.info("_build_map 시작")
            loaded = False

            if not loaded and krx:
                today = datetime.date.today()
                for delta in range(7):
                    ds = (today - datetime.timedelta(days=delta)).strftime("%Y%m%d")
                    try:
                        tickers = krx.get_market_ticker_list(ds, market="KOSPI")
                        if not tickers:
                            _log.warning(f"pykrx delta={delta} ({ds}): 빈 리스트")
                            continue
                        _log.info(f"pykrx 영업일={ds}, KOSPI={len(tickers)}")
                        for mkt, suf in [("KOSPI", ".KS"), ("KOSDAQ", ".KQ")]:
                            for t in krx.get_market_ticker_list(ds, market=mkt):
                                code = re.sub(r'^[A-Za-z]+', '', t)
                                if len(code) != 6 or not code.isdigit(): continue
                                name = krx.get_market_ticker_name(t)
                                if not name: continue
                                self._name_map[name] = (code, suf)
                                self._code_map[code] = suf
                        loaded = bool(self._code_map)
                        break
                    except Exception as e:
                        _log.warning(f"pykrx delta={delta} ({ds}) 실패: {type(e).__name__}: {e}")
                        break

            if not loaded:
                _log.info("KRX KIND URL 시도 (kind.krx.co.kr)")
                try:
                    import io, requests as _req, pandas as _pd
                    url = ("https://kind.krx.co.kr/corpgeneral/corpList.do"
                           "?method=download&searchType=13")
                    resp = _req.get(url, timeout=20,
                                    headers={"User-Agent": "Mozilla/5.0"})
                    resp.raise_for_status()
                    html = resp.content.decode("cp949", errors="ignore")
                    df = _pd.read_html(io.StringIO(html), flavor="lxml")[0]
                    _log.info(f"KIND 컬럼: {list(df.columns)}, 행수={len(df)}")
                    mkt_map = {"유가": ".KS", "코스닥": ".KQ", "KOSPI": ".KS", "KOSDAQ": ".KQ"}
                    for _, row in df.iterrows():
                        code = str(row.get("종목코드", "")).strip().zfill(6)
                        name = str(row.get("회사명", "")).strip()
                        mkt  = str(row.get("시장구분", "")).strip()
                        suf  = mkt_map.get(mkt)
                        if not suf or not code.isdigit() or len(code) != 6 or not name:
                            continue
                        self._name_map[name] = (code, suf)
                        self._code_map[code] = suf
                    loaded = bool(self._code_map)
                    _log.info(f"KIND 완료: {len(self._code_map)}종목")
                except ImportError:
                    _log.warning("requests/pandas/lxml 미설치")
                except Exception as e:
                    _log.error(f"KIND 실패: {type(e).__name__}: {e}\n{traceback.format_exc()}")

            if not loaded:
                raise RuntimeError("종목 DB 로드 실패 — pykrx/KIND 모두 불가")

            self._map_ready = True
            sam = next((k for k in self._name_map if "삼성전자" in k), None)
            _log.info(f"완료: {len(self._code_map)}종목, 삼성전자={self._name_map.get(sam)}")
            _ui(f"DB 준비 ({len(self._code_map):,}종목)")
        except Exception as ex:
            _log.error(f"_build_map 예외: {type(ex).__name__}: {ex}\n{traceback.format_exc()}")
            self._map_ready = True
            _ui("KRX 접속 불가 — Naver 검색 모드")
        finally:
            self.root.after(0, self._start_refresh)

    def _resolve(self, q: str):
        q = q.strip()
        if q.isdigit() and len(q) == 6:
            suf = self._code_map.get(q, ".KS")
            return q, q + suf, self._reverse_name(q) or q
        if q in self._name_map:
            c, suf = self._name_map[q]; return c, c + suf, q
        for name, (c, suf) in self._name_map.items():
            if q in name: return c, c + suf, name
        return q, q + ".KS", q

    def _reverse_name(self, code6: str):
        for name, (c, _) in self._name_map.items():
            if c == code6: return name
        return None

    # ── 가격 조회 ────────────────────────────────────────────────
    def _fetch_one(self, item: dict):
        price, chg = self._fetch_pykrx(item.get("code", ""))

        if price is None and yf:
            price, chg = self._fetch_yf(item["ycode"])
            if price is None and item["ycode"].endswith(".KS"):
                alt = item["ycode"].replace(".KS", ".KQ")
                price, chg = self._fetch_yf(alt)
                if price is not None:
                    item["ycode"] = alt; self._save()

        item["price"] = price; item["chg"] = chg

        if krx and item.get("code"):
            has_kor = any('\uAC00' <= c <= '\uD7A3' for c in item.get("name", ""))
            if not has_kor:
                try:
                    kor = krx.get_market_ticker_name(item["code"])
                    if kor:
                        item["name"] = kor; self._save()
                except Exception:
                    pass

        threading.Thread(target=self._fetch_news, args=(item,), daemon=True).start()
        self.root.after(0, self._rebuild_list)

    def _fetch_pykrx(self, code6: str):
        if not krx or not code6 or not code6.isdigit():
            return None, None
        try:
            import datetime
            today  = datetime.date.today()
            from_d = (today - datetime.timedelta(days=7)).strftime("%Y%m%d")
            to_d   = today.strftime("%Y%m%d")
            df = krx.get_market_ohlcv_by_date(from_d, to_d, code6)
            if df is None or df.empty:
                _log.warning(f"fetch_pykrx({code6}): 빈 데이터")
                return None, None
            last  = df.iloc[-1]
            price = float(last["종가"])
            chg   = float(last["등락률"]) if "등락률" in df.columns else None
            if chg is None and len(df) >= 2:
                prev = float(df.iloc[-2]["종가"])
                chg  = (price - prev) / prev * 100 if prev > 0 else 0.0
            _log.info(f"fetch_pykrx({code6}): price={price}, chg={chg}")
            return price, chg
        except Exception as e:
            _log.error(f"fetch_pykrx({code6}) 예외: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            return None, None

    def _fetch_yf(self, symbol: str):
        try:
            hist = yf.Ticker(symbol).history(period="5d", interval="1d")
            if hist.empty: return None, None
            price = float(hist["Close"].iloc[-1])
            if len(hist) >= 2:
                prev = float(hist["Close"].iloc[-2])
                chg  = (price - prev) / prev * 100 if prev > 0 else 0.0
            else:
                chg = 0.0
            return price, chg
        except Exception:
            return None, None

    def _fetch_news(self, item: dict):
        """Google News RSS로 뉴스 헤드라인 조회"""
        import urllib.request, xml.etree.ElementTree as ET
        from urllib.parse import quote_plus
        code6 = item.get("code", "")
        name  = item.get("name", "") or code6
        if not code6:
            return
        try:
            url = (f"https://news.google.com/rss/search"
                   f"?q={quote_plus(name)}&hl=ko&gl=KR&ceid=KR:ko")
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read()

            root = ET.fromstring(raw)
            articles = []
            seen = set()
            for it in root.findall(".//item"):
                title   = (it.findtext("title")   or "").strip()
                link    = (it.findtext("link")    or "").strip()
                pubdate = (it.findtext("pubDate") or "").strip()
                if not title or len(title) < 6 or title in seen:
                    continue
                seen.add(title)
                articles.append({
                    "title": title,
                    "link":  link,
                    "date":  _parse_pubdate(pubdate),
                })
                if len(articles) >= 10:
                    break

            self._news_cache[code6] = articles
            self._stock_sent.pop(code6, None)  # 캐시 무효화

            def _ui_update(c=code6, n=name):
                if self._selected_code == c:
                    self._show_news(c, n)
                self._rebuild_list()

            self.root.after(0, _ui_update)

            _log.debug(f"뉴스({code6}): {len(articles)}건"
                       + (f" — {articles[0]['title'][:35]}" if articles else ""))
        except Exception as e:
            self._news_cache[code6] = []
            _log.debug(f"뉴스 조회 실패 ({code6}): {type(e).__name__}: {e}")

    def _fetch_all(self):
        for item in self._watchlist:
            threading.Thread(target=self._fetch_one, args=(item,), daemon=True).start()

    def _start_refresh(self): self._fetch_all(); self._schedule()
    def _schedule(self):      self.root.after(REFRESH_MS, self._auto_refresh)
    def _auto_refresh(self):
        self._sv_status.set(f"갱신 중... {time.strftime('%H:%M:%S')}")
        self._fetch_all(); self._schedule()

    # ════════════════════════════════════════════════════════════
    #  리스트 UI
    # ════════════════════════════════════════════════════════════
    def _rebuild_list(self):
        for w in self._list_frame.winfo_children():
            w.destroy()
        if not self._watchlist:
            tk.Label(self._list_frame, text="종목을 추가하세요",
                     bg=BG, fg=FG_DIM, font=FONTS).pack(pady=10)
        else:
            for i, s in enumerate(self._watchlist):
                self._make_row(i, s)
            # 선택된 종목 없으면 첫 번째 자동 선택
            if not self._selected_code:
                first = self._watchlist[0]
                self._selected_code = first.get("code", "")
                self._show_news(self._selected_code, first.get("name", ""))
        self._fit_window()

    def _toggle_expand(self, idx: int):
        s = self._watchlist[idx]
        s["expanded"] = not s.get("expanded", False)
        self._rebuild_list()

    def _make_row(self, idx: int, s: dict):
        bg       = BG_A if idx % 2 == 0 else BG_B
        expanded = s.get("expanded", False)
        code6    = s.get("code", "")
        name     = s.get("name", "")

        # ── 행1: 종목명 | 현재가 | 등락률 | ▸ | × ────────────────
        row1 = tk.Frame(self._list_frame, bg=bg, height=ROW_H)
        row1.pack(fill="x"); row1.pack_propagate(False)

        chg = s.get("chg")
        if chg is None:       p_col, c_txt, c_col = FG_MED, "---", FG_DIM
        elif chg > 0.005:     p_col, c_txt, c_col = UP, f"+{chg:.2f}%", UP
        elif chg < -0.005:    p_col, c_txt, c_col = DN, f"{chg:.2f}%",  DN
        else:                 p_col, c_txt, c_col = NC, f"{chg:.2f}%",  NC

        tk.Button(row1, text="×", bg=bg, fg=FG_DIM, font=FONTSS,
                  bd=0, padx=4, activebackground="#333", activeforeground=UP,
                  cursor="hand2", command=lambda i=idx: self._remove(i),
                  ).pack(side="right", padx=(0, 3))
        tk.Button(row1, text="▾" if expanded else "▸",
                  bg=bg, fg=FG_MED, font=FONTSS,
                  bd=0, padx=2, activebackground="#333",
                  cursor="hand2", command=lambda i=idx: self._toggle_expand(i),
                  ).pack(side="right")

        # 종목명 색상: 뉴스 호재/악재 반영 + ♥호재수 종목명 ☁악재수
        pos_cnt, neg_cnt, nsent = self._compute_stock_sent(code6)
        name_fg = UP if nsent == 'pos' else (DN if nsent == 'neg' else FG)
        tk.Label(row1, text=f"♥{pos_cnt}", bg=bg,
                 fg=UP if pos_cnt > 0 else FG_DIM,
                 font=FONTSS).pack(side="left", padx=(5, 1))
        tk.Label(row1, text=name, bg=bg, fg=name_fg,
                 font=FONT, anchor="w", width=7).pack(side="left")
        tk.Label(row1, text=f"☁{neg_cnt}", bg=bg,
                 fg=DN if neg_cnt > 0 else FG_DIM,
                 font=FONTSS).pack(side="left", padx=(1, 2))
        p_txt = f"{s['price']:,.0f}" if isinstance(s["price"], (int, float)) else "---"
        tk.Label(row1, text=p_txt, bg=bg, fg=p_col,
                 font=FONTB, anchor="e", width=8).pack(side="left")
        tk.Label(row1, text=c_txt, bg=bg, fg=c_col,
                 font=FONTB, anchor="e", width=8).pack(side="left")

        # 우클릭 메뉴 + 호버 → 뉴스 패널 갱신
        def ctx(e, i=idx):
            m = tk.Menu(self.root, tearoff=0, bg="#252525", fg=FG,
                        activebackground="#383838", font=FONTS)
            m.add_command(label=f"  {self._watchlist[i]['name']} 삭제  ",
                          command=lambda: self._remove(i))
            m.tk_popup(e.x_root, e.y_root)

        def on_enter(e, c=code6, n=name):
            self._selected_code = c
            self._show_news(c, n)

        for w in [row1] + list(row1.winfo_children()):
            w.bind("<Button-3>", ctx)
            w.bind("<Enter>", on_enter)

        if not expanded:
            return

        # ── 행2: 평단 / 수량 / 평가금액 (펼쳤을 때만) ────────────
        row2 = tk.Frame(self._list_frame, bg=bg, height=SUB_H)
        row2.pack(fill="x"); row2.pack_propagate(False)

        tk.Label(row2, text=" 평단", bg=bg, fg=FG_MED, font=FONTSS).pack(side="left")

        avg_var = tk.StringVar(value=f"{int(s['avg_price']):,}" if s.get("avg_price") else "")
        avg_e   = tk.Entry(row2, textvariable=avg_var, width=8,
                           bg=BG_ENT, fg=FG, insertbackground=FG,
                           relief="flat", font=FONTSS, justify="right",
                           highlightthickness=1, highlightbackground="#333")
        avg_e.pack(side="left", padx=(2, 1), ipady=2)

        tk.Label(row2, text="×", bg=bg, fg=FG_MED, font=FONTSS).pack(side="left")

        qty_var = tk.StringVar(value=str(s["qty"]) if s.get("qty") else "")
        qty_e   = tk.Entry(row2, textvariable=qty_var, width=5,
                           bg=BG_ENT, fg=FG, insertbackground=FG,
                           relief="flat", font=FONTSS, justify="right",
                           highlightthickness=1, highlightbackground="#333")
        qty_e.pack(side="left", padx=(1, 4), ipady=2)

        eval_lbl = tk.Label(row2, text="", bg=bg, font=FONTSS, anchor="w")
        eval_lbl.pack(side="left", fill="x", expand=True)

        def _calc_and_show():
            try:
                avg = float(avg_var.get().replace(",", "").strip())
                qty = int(qty_var.get().replace(",", "").strip())
                if avg <= 0 or qty <= 0: raise ValueError
            except (ValueError, AttributeError):
                eval_lbl.config(text="", fg=FG_MED); return
            price = s.get("price")
            if not isinstance(price, (int, float)):
                eval_lbl.config(text=f"수량 {qty:,}주", fg=FG_MED); return
            eval_amt   = price * qty
            cost       = avg   * qty
            profit     = eval_amt - cost
            profit_pct = profit / cost * 100 if cost else 0
            col  = UP if profit > 0.5 else (DN if profit < -0.5 else NC)
            sign = "+" if profit >= 0 else ""
            eval_lbl.config(fg=col,
                text=f"{eval_amt:,.0f}원  {sign}{profit_pct:.1f}%  ({sign}{profit:,.0f})")

        def _on_change(_=None):
            try:
                raw = avg_var.get().replace(",", "").strip()
                s["avg_price"] = float(raw) if raw else None
            except ValueError:
                s["avg_price"] = None
            try:
                raw = qty_var.get().replace(",", "").strip()
                s["qty"] = int(raw) if raw else None
            except ValueError:
                s["qty"] = None
            self._save(); _calc_and_show()

        avg_e.bind("<FocusOut>", _on_change)
        avg_e.bind("<Return>",   _on_change)
        qty_e.bind("<FocusOut>", _on_change)
        qty_e.bind("<Return>",   _on_change)
        _calc_and_show()

        for w in [row2] + list(row2.winfo_children()):
            if w not in (avg_e, qty_e):
                w.bind("<Button-3>", ctx)

    # ── 창 높이 자동 조정 (수동 리사이즈 후에는 호출 안 됨) ───────
    def _fit_window(self):
        if self._win_h > 0:  # 사용자가 수동 리사이즈한 경우 → 자동 높이 조정 안 함
            return
        if not self._watchlist:
            content_h = 30
        else:
            content_h = sum(
                STOCK_H if s.get("expanded") else ROW_H
                for s in self._watchlist
            )
        h = HANDLE_H + SEARCH_H + content_h + 22 + 6  # +6 for grip
        x, y = self.root.winfo_x(), self.root.winfo_y()
        self.root.geometry(f"{self._win_w}x{h}+{x}+{y}")

    def _remove(self, idx: int):
        code = self._watchlist[idx].get("code", "")
        del self._watchlist[idx]
        self._news_cache.pop(code, None)
        if self._selected_code == code:
            self._selected_code = ""
        self._rebuild_list()
        self._save()

    # ── 저장 / 불러오기 ──────────────────────────────────────────
    def _save(self):
        try:
            with open(SAVE_FILE, "w", encoding="utf-8") as f:
                json.dump([{
                    "code":      s["code"],
                    "ycode":     s["ycode"],
                    "name":      s["name"],
                    "avg_price": s.get("avg_price"),
                    "qty":       s.get("qty"),
                } for s in self._watchlist], f, ensure_ascii=False, indent=2)
        except Exception: pass

    def _load(self):
        if not os.path.exists(SAVE_FILE): return
        try:
            with open(SAVE_FILE, encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                ycode = d.get("ycode", "")
                suf   = ".KS" if ycode.endswith(".KS") else (".KQ" if ycode.endswith(".KQ") else "")
                if not suf or not ycode[:-3].isdigit() or len(ycode[:-3]) != 6: continue
                self._watchlist.append(dict(
                    code=d["code"], ycode=ycode, name=d["name"],
                    price=None, chg=None,
                    avg_price=d.get("avg_price"),
                    qty=d.get("qty"),
                ))
        except Exception: pass


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if yf  is None: print("경고: pip install yfinance")
    if krx is None: print("경고: pip install pykrx")
    StockWidget()
