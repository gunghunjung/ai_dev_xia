#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
야유회 기획 도우미 (tkinter 버전)
구미 근교 | 23명 | 4월 말 | 1인 10만원
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.font import Font

# ─────────────────────────────────────────────────────────────────────────────
# 장소 후보 데이터
# ─────────────────────────────────────────────────────────────────────────────
VENUES = [
    ("v01","구미","금오산 도립공원",       "울창한 숲+금오저수지 산책로, 피크닉 광장 완비",           False,["자연","산책","피크닉"]),
    ("v02","구미","낙동강 생태공원",        "강변 잔디광장+자전거도로, 봄꽃 절정 명소",               False,["자연","피크닉"]),
    ("v03","구미","선산 전통시장",          "경북 5대 시장, 먹거리 골목+지역 특산품 탐방",            False,["시장","맛집"]),
    ("v04","구미","박정희 생가 + 인근 공원","역사 문화 투어 + 주변 공원 피크닉",                       False,["역사","산책"]),
    ("v05","구미","구미 봉곡동 카페거리",   "감성 카페 밀집 지역, 자유 타임 & 브런치 코스 최적",      False,["카페","감성"]),
    ("v06","김천","직지사",                 "천년 고찰+봄 벚꽃 터널 유명, 사찰 산책로",               False,["사찰","봄꽃"]),
    ("v07","김천","직지문화공원",           "넓은 잔디광장+야외무대, 바베큐 가능 구역",               False,["잔디","바베큐"]),
    ("v08","김천","김천 레인보우힐링파크",  "숲속 힐링 테마파크, 짚라인·산책로 패키지",              False,["힐링","체험"]),
    ("v09","김천","황악산 자연휴양림",      "조용한 숲속 힐링, 통나무집 캠프 분위기",                 False,["숲","힐링"]),
    ("v10","김천","중앙시장 + 요리체험관",  "재래시장 탐방+인근 요리 체험 시설 연계",                 True, ["시장","요리"]),
    ("v11","상주","경천섬",                 "낙동강 벚꽃 명소 1위, 피크닉 잔디+자전거 코스",          False,["봄꽃","피크닉"]),
    ("v12","상주","도남서원",               "조선시대 서원, 한옥 분위기+고택 체험",                   False,["역사","한옥"]),
    ("v13","상주","갑장산 자연휴양림",      "숲속 산책+계곡, 조용한 힐링 캠프 가능",                  False,["숲","계곡"]),
    ("v14","상주","상주 전통시장",          "쌀·곶감 특산품+먹거리, 요리 재료 구매 최적",             False,["시장","특산품"]),
    ("v15","대구북부","팔공산 케이블카",    "케이블카 탑승+능선 산책, 대구 대표 봄 나들이 코스",      False,["산","봄꽃"]),
    ("v16","대구북부","동화사",             "팔공산 천년 고찰, 봄꽃 명소+계곡 산책",                  False,["사찰","봄꽃"]),
    ("v17","대구북부","대구 북구 공방거리", "도자기·캔들·리스 공방 밀집, 단체 체험 패키지 가능",      True, ["공방","실내"]),
    ("v18","안동","하회마을 한옥 체험관",   "유네스코 세계유산, 한옥 체험+전통 놀이 단체 가능",       True, ["한옥","전통"]),
    ("v19","안동","월영교 + 안동 카페거리", "야경 명소+감성 카페, 저녁 코스로 최적",                  False,["감성","야경"]),
    ("v20","경주","교촌마을 + 경주 맛집",   "첨성대·교촌 한옥 마을, 맛집 투어+역사 산책",            False,["역사","맛집"]),
    ("v21","기타","인근 펜션 전용 대관",    "전용 마당+바베큐+수영장 옵션, 프라이빗 운영",            True, ["프라이빗","바베큐"]),
    ("v22","기타","청도 와인터널",          "140년 된 철도 터널 와인 숙성 공간, 이색 투어+시음",      True, ["이색","와인"]),
    ("v23","기타","성주 참외밭 농촌 체험",  "봄 참외 수확 체험+전원 피크닉, 이색 협업 미션 가능",    False,["농촌","체험"]),
]

# ─────────────────────────────────────────────────────────────────────────────
# 활동 후보 데이터
# ─────────────────────────────────────────────────────────────────────────────
ACTIVITIES = [
    ("a01","미션/게임형","감성 포토 미션",     "지정 스팟 5곳 팀별 창의 사진 → 투표로 우승팀 선발",          90,  0,     "스마트폰 충전 상태 확인"),
    ("a02","미션/게임형","미스터리 미션 랠리", "단서 카드 추적 → 지역 명소 탐방하며 보물 찾기",              120, 5000, "QR코드 활용 시 자동 채점 가능"),
    ("a03","미션/게임형","회사 퀴즈 배틀",     "우리 회사 상식+일반 퀴즈, 카훗(Kahoot) 앱 활용",            60,  0,    "사전 문제 30문항 준비"),
    ("a04","미션/게임형","맛집 스탬프 투어",   "팀별 맛집 3~4곳 스탬프 수집 → 품평회 발표",                 120, 20000,"사전 식당 동선 답사 필수"),
    ("a05","미션/게임형","보물 지도 빙고",      "장소 이름 빙고판+인증샷 조건, 이동 중 버스 안 진행 가능",   60,  0,    "버스 이동 구간 활용"),
    ("a06","미션/게임형","팀 탐험대 OX퀴즈",  "지역 역사/상식 OX, 틀리면 미션(노래 한 소절) 수행",         45,  0,    "부담 없는 벌칙 설계 중요"),
    ("a07","감성/힐링형","자유 피크닉 타임",   "돗자리+도시락+음악, 그냥 쉬는 시간",                         90,  8000, "블루투스 스피커 필수"),
    ("a08","감성/힐링형","보드게임 자유 타임", "루미큐브·할리갈리·UNO 등, 원하는 사람끼리 자연스럽게",       90,  3000, "강요 없는 분위기 조성이 핵심"),
    ("a09","감성/힐링형","감성 카페 투어",     "뷰 맛집 카페 1~2곳 방문, 디저트+자유 담소",                  60,  8000, "사전 예약 필수 (단체 23명)"),
    ("a10","감성/힐링형","한옥/숲 자유 산책",  "자연 속 자유 산책, 사진 찍으며 힐링",                        60,  0,    "담당자 동선 파악만 해두면 OK"),
    ("a11","감성/힐링형","소원 카드 쓰기",     "봄에 쓰는 올해 소원 엽서 → 팀끼리 공유 or 비공개 보관",     30,  2000, "엽서+펜 사전 준비"),
    ("a12","경쟁형","전통 놀이 배틀",          "투호·윷놀이·제기차기 팀 대항, 점수 합산 시상",               90,  0,    "한옥/문화관 시설 활용 시 도구 제공"),
    ("a13","경쟁형","작품 감상 투표",          "공방 체험 후 최고 작품 익명 투표+시상",                      30,  0,    "모두가 상받는 구조 추천"),
    ("a14","경쟁형","팀 사진전 투표",          "포토 미션 결과물 실시간 공유 → 좋아요 수 집계",              30,  0,    "단체 카톡방 활용"),
    ("a15","협업형","팀 바베큐",               "역할 분담(굽기·반찬·음료)으로 함께 만드는 저녁 식사",        90,  22000,"그릴·숯·착화제 사전 준비"),
    ("a16","협업형","캔들 공방 체험",          "향기·색상 선택 → 강사 지도로 캔들 직접 제작, 완성품 가져감", 120, 35000,"완성품 집에 가져감 → 기억 지속"),
    ("a17","협업형","도자기 공방 체험",        "물레 or 핸드빌딩으로 나만의 그릇 만들기",                    120, 40000,"2~4주 후 택배 수령 가능 시설 선택"),
    ("a18","협업형","봄꽃 리스 만들기",        "생화·조화로 봄꽃 리스 제작, 강사 진행으로 부담 없음",        90,  30000,"4월 봄 시즌과 완벽히 맞음"),
    ("a19","협업형","팀 상징 만들기",          "한지·소품으로 팀 이름+슬로건+마크 제작 후 발표",             60,  5000, "팀 정체성 강화 효과"),
    ("a20","협업형","팀 요리 대결",            "시장 재료 구매 → 지정 요리 1가지 팀별 완성+시식",            120, 15000,"비빔밥·전 부치기 등 쉬운 메뉴 추천"),
    ("a21","협업형","팀원 소통 카드게임",      "우리팀 몰래카메라 - 사전 설문 → 누구일까요 퀴즈",            60,  3000, "사전 설문지 배포 필수 (1주 전)"),
    ("a22","협업형","팀 영상 제작",            "오늘 하루를 15초 릴스로 편집 → 저녁 상영회",                 60,  0,    "편집 잘하는 팀원 미리 파악"),
]

# ─────────────────────────────────────────────────────────────────────────────
# 기획안 생성 로직
# ─────────────────────────────────────────────────────────────────────────────
def generate_plan(venue_id, activity_ids):
    venue = next((v for v in VENUES if v[0] == venue_id), None)
    activities = [a for a in ACTIVITIES if a[0] in activity_ids]
    if not venue or not activities:
        return None
    act_cost = sum(a[5] for a in activities)
    total = min(400000 + (act_cost + 30000) * 23 + 700000, 2300000)

    schedule = [("09:00", "출발 (회사 집합, 버스 이동)"),
                ("10:00", f"도착 — {venue[2]} 체크인 및 오리엔테이션"),
                ("10:30", "팀 구성 및 오늘 일정 소개 (아이스브레이킹)")]
    cur_h, cur_m = 11, 0
    lunch_added = False
    for act in activities:
        if cur_h >= 12 and not lunch_added:
            schedule.append(("12:30", "점심 식사 (현지 맛집 or 도시락)"))
            cur_h, cur_m = 14, 0
            lunch_added = True
        schedule.append((f"{cur_h:02d}:{cur_m:02d}", f"▶ {act[2]} : {act[3][:35]}"))
        cur_m += act[4]
        cur_h += cur_m // 60; cur_m %= 60
    if not lunch_added:
        schedule.append(("12:30", "점심 식사 (현지 맛집 or 도시락)"))
    dh = max(cur_h, 17)
    schedule.append((f"{dh:02d}:30", "저녁 식사 + 자유 담소"))
    schedule.append((f"{max(dh+1,19):02d}:30", "귀사 이동"))
    schedule.append((f"{max(dh+2,21):02d}:00", "해산"))

    checklist = ["버스 예약 확인","명찰","구급약품","블루투스 스피커"]
    for a in activities:
        extras = {
            "a01":["충전된 스마트폰","카메라"], "a02":["미션 키트(단서 카드)","QR코드 카드"],
            "a03":["카훗 문제 준비 (30문항)"], "a04":["스탬프 투어 코스지","평가지"],
            "a07":["돗자리 5개","파라솔 3개","도시락"], "a08":["보드게임 세트"],
            "a12":["투호 세트","윷놀이 세트"], "a15":["바베큐 그릴","숯+착화제","고기+채소"],
            "a16":["공방 사전 예약(필수)","앞치마"], "a17":["공방 사전 예약(필수)"],
            "a18":["공방 사전 예약(필수)"], "a21":["팀원 사전 설문지"],
        }.get(a[0], [])
        checklist += extras
    checklist = list(dict.fromkeys(checklist))
    return dict(venue=venue, activities=activities, schedule=schedule,
                checklist=checklist, total=total, act_cost=act_cost)

# ─────────────────────────────────────────────────────────────────────────────
# 색상 테마
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":      "#F0F2F5",
    "panel":   "#FFFFFF",
    "header":  "#1E2235",
    "accent":  "#3D7EF5",
    "accent2": "#F4A261",
    "green":   "#3DBE7A",
    "red":     "#E25A5A",
    "text":    "#1A1D2E",
    "sub":     "#6B7280",
    "border":  "#D1D5DB",
    "sel_bg":  "#EBF2FF",
}

CAT_CLR = {
    "미션/게임형": "#3D7EF5",
    "감성/힐링형": "#F4A261",
    "경쟁형":      "#E25A5A",
    "협업형":      "#3DBE7A",
}

REG_CLR = {
    "구미":    "#3D7EF5",
    "김천":    "#3DBE7A",
    "상주":    "#F4A261",
    "대구북부":"#C9674C",
    "안동":    "#9B59B6",
    "경주":    "#E67E22",
    "기타":    "#7F8C8D",
}

# ─────────────────────────────────────────────────────────────────────────────
# 메인 앱
# ─────────────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🌸 야유회 기획 도우미 — 구미 근교 23인 4월 말")
        self.geometry("1200x780")
        self.minsize(960, 640)
        self.configure(bg=C["bg"])

        # 폰트
        self.f_title  = Font(family="맑은 고딕", size=16, weight="bold")
        self.f_head   = Font(family="맑은 고딕", size=13, weight="bold")
        self.f_body   = Font(family="맑은 고딕", size=11)
        self.f_small  = Font(family="맑은 고딕", size=10)
        self.f_badge  = Font(family="맑은 고딕", size=9, weight="bold")

        self.selected_venue = tk.StringVar()
        self.activity_vars  = {}

        self._build_header()
        self._build_body()

    # ── 헤더 ──────────────────────────────────────────────────────────────────
    def _build_header(self):
        hf = tk.Frame(self, bg=C["header"], height=54)
        hf.pack(fill="x"); hf.pack_propagate(False)
        tk.Label(hf, text="🌸  야유회 기획 도우미",
                 font=self.f_title, bg=C["header"], fg="white").pack(side="left", padx=20, pady=10)
        tk.Label(hf, text="구미 근교 | 23명 | 4월 말 | 1인 10만원",
                 font=self.f_small, bg=C["header"], fg="#A0AABB").pack(side="right", padx=20)

    # ── 본문 (노트북 탭) ──────────────────────────────────────────────────────
    def _build_body(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook", background=C["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", font=("맑은 고딕", 11),
                        padding=[16, 6], background=C["bg"], foreground=C["sub"])
        style.map("TNotebook.Tab",
                  background=[("selected", C["panel"])],
                  foreground=[("selected", C["accent"])])

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=12, pady=(8,12))

        self.tab_select = tk.Frame(self.nb, bg=C["bg"])
        self.tab_result = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(self.tab_select, text="  1️⃣  장소 & 활동 선택  ")
        self.nb.add(self.tab_result, text="  2️⃣  기획안 결과  ")

        self._build_select_tab()
        self._build_result_tab()

    # ── 선택 탭 ───────────────────────────────────────────────────────────────
    def _build_select_tab(self):
        tab = self.tab_select
        # 좌우 분할
        paned = tk.PanedWindow(tab, orient="horizontal", bg=C["bg"],
                               sashwidth=6, sashpad=2, sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=4, pady=4)

        # ── 왼쪽: 장소 ────────────────────────────────────────────────────────
        left = tk.Frame(paned, bg=C["bg"])
        paned.add(left, minsize=350)

        lh = tk.Frame(left, bg=C["bg"])
        lh.pack(fill="x", pady=(4,6))
        tk.Label(lh, text="📍 장소 선택  (1개)", font=self.f_head,
                 bg=C["bg"], fg=C["text"]).pack(side="left")

        # 지역 필터
        rf = tk.Frame(left, bg=C["bg"])
        rf.pack(fill="x", pady=(0,4))
        self._region_var = tk.StringVar(value="전체")
        for reg in ["전체","구미","김천","상주","대구북부","안동","경주","기타"]:
            rb = tk.Radiobutton(rf, text=reg, value=reg, variable=self._region_var,
                                font=self.f_small, bg=C["bg"], fg=C["sub"],
                                activebackground=C["bg"], selectcolor=C["bg"],
                                command=self._filter_venues, indicatoron=False,
                                relief="flat", bd=1, padx=6, pady=2, cursor="hand2")
            rb.pack(side="left", padx=1)

        # 장소 목록
        vf = tk.Frame(left, bg=C["bg"])
        vf.pack(fill="both", expand=True)
        vc = tk.Canvas(vf, bg=C["bg"], highlightthickness=0)
        vsb = ttk.Scrollbar(vf, orient="vertical", command=vc.yview)
        self.venue_inner = tk.Frame(vc, bg=C["bg"])
        self.venue_inner.bind("<Configure>",
            lambda e: vc.configure(scrollregion=vc.bbox("all")))
        vc.create_window((0,0), window=self.venue_inner, anchor="nw")
        vc.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        vc.pack(side="left", fill="both", expand=True)
        vc.bind("<MouseWheel>", lambda e: vc.yview_scroll(-1*(e.delta//120),"units"))
        self._venue_canvas = vc
        self._venue_frames = {}
        self._build_venue_list()

        # ── 오른쪽: 활동 ──────────────────────────────────────────────────────
        right = tk.Frame(paned, bg=C["bg"])
        paned.add(right, minsize=400)

        rh = tk.Frame(right, bg=C["bg"])
        rh.pack(fill="x", pady=(4,6))
        tk.Label(rh, text="🎯 활동 선택  (여러 개)", font=self.f_head,
                 bg=C["bg"], fg=C["text"]).pack(side="left")
        self._act_count_lbl = tk.Label(rh, text="0개 선택됨",
                                       font=self.f_small, bg=C["bg"], fg=C["accent"])
        self._act_count_lbl.pack(side="right")

        # 카테고리 필터
        cf = tk.Frame(right, bg=C["bg"])
        cf.pack(fill="x", pady=(0,4))
        self._cat_var = tk.StringVar(value="전체")
        for cat in ["전체","미션/게임형","감성/힐링형","경쟁형","협업형"]:
            rb = tk.Radiobutton(cf, text=cat, value=cat, variable=self._cat_var,
                                font=self.f_small, bg=C["bg"], fg=C["sub"],
                                activebackground=C["bg"], selectcolor=C["bg"],
                                command=self._filter_activities, indicatoron=False,
                                relief="flat", bd=1, padx=6, pady=2, cursor="hand2")
            rb.pack(side="left", padx=1)

        # 활동 목록
        af = tk.Frame(right, bg=C["bg"])
        af.pack(fill="both", expand=True)
        ac = tk.Canvas(af, bg=C["bg"], highlightthickness=0)
        asb = ttk.Scrollbar(af, orient="vertical", command=ac.yview)
        self.act_inner = tk.Frame(ac, bg=C["bg"])
        self.act_inner.bind("<Configure>",
            lambda e: ac.configure(scrollregion=ac.bbox("all")))
        ac.create_window((0,0), window=self.act_inner, anchor="nw")
        ac.configure(yscrollcommand=asb.set)
        asb.pack(side="right", fill="y")
        ac.pack(side="left", fill="both", expand=True)
        ac.bind("<MouseWheel>", lambda e: ac.yview_scroll(-1*(e.delta//120),"units"))
        self._act_canvas = ac
        self._act_frames = {}
        self._build_activity_list()

        # 하단 버튼
        bot = tk.Frame(right, bg=C["bg"])
        bot.pack(fill="x", pady=(8,4))
        self._budget_lbl = tk.Label(bot, text="예상 비용: 계산 중...",
                                    font=self.f_small, bg=C["bg"], fg=C["sub"])
        self._budget_lbl.pack(side="left")
        gen_btn = tk.Button(bot, text="  기획안 생성  ▶",
                            font=Font(family="맑은 고딕", size=12, weight="bold"),
                            bg=C["accent"], fg="white", relief="flat",
                            padx=16, pady=8, cursor="hand2", bd=0,
                            activebackground="#2A6AE0", activeforeground="white",
                            command=self._generate)
        gen_btn.pack(side="right")

    def _make_venue_row(self, v, parent):
        vid, region, name, desc, indoor, tags = v
        row = tk.Frame(parent, bg=C["panel"], relief="flat", bd=0,
                       highlightthickness=1, highlightbackground=C["border"],
                       cursor="hand2")
        row.pack(fill="x", padx=4, pady=2)
        row.columnconfigure(3, weight=1)

        # 클릭 이벤트
        def click(_evt=None, _vid=vid, _row=row):
            self.selected_venue.set(_vid)
            self._refresh_venue_sel()
        row.bind("<Button-1>", click)
        for w in row.winfo_children():
            w.bind("<Button-1>", click)

        # 지역 배지
        rc = REG_CLR.get(region, "#888")
        rbl = tk.Label(row, text=region, font=self.f_badge,
                       bg=rc, fg="white", width=5, anchor="center", padx=2, pady=2)
        rbl.grid(row=0, column=0, rowspan=2, padx=(8,6), pady=6, sticky="ns")
        rbl.bind("<Button-1>", click)

        # 이름
        nl = tk.Label(row, text=name, font=Font(family="맑은 고딕", size=11, weight="bold"),
                      bg=C["panel"], fg=C["text"], anchor="w")
        nl.grid(row=0, column=1, sticky="w", padx=2, pady=(6,0))
        nl.bind("<Button-1>", click)

        # 설명
        dl = tk.Label(row, text=desc[:44]+("…" if len(desc)>44 else ""),
                      font=self.f_small, bg=C["panel"], fg=C["sub"], anchor="w")
        dl.grid(row=1, column=1, sticky="w", padx=2, pady=(0,4))
        dl.bind("<Button-1>", click)

        # 실내/야외
        ic = "#3D7EF5" if indoor else "#3DBE7A"
        itl = tk.Label(row, text="실내" if indoor else "야외",
                       font=self.f_badge, bg=C["panel"], fg=ic)
        itl.grid(row=0, column=2, rowspan=2, padx=8, pady=6)
        itl.bind("<Button-1>", click)

        self._venue_frames[vid] = row
        return row

    def _build_venue_list(self):
        for w in self.venue_inner.winfo_children():
            w.destroy()
        self._venue_frames = {}
        for v in VENUES:
            self._make_venue_row(v, self.venue_inner)

    def _filter_venues(self):
        reg = self._region_var.get()
        for v in VENUES:
            frame = self._venue_frames.get(v[0])
            if not frame:
                continue
            show = reg == "전체" or v[1] == reg
            if show:
                frame.pack(fill="x", padx=4, pady=2)
            else:
                frame.pack_forget()

    def _refresh_venue_sel(self):
        sel = self.selected_venue.get()
        for vid, frame in self._venue_frames.items():
            if vid == sel:
                frame.configure(bg=C["sel_bg"], highlightbackground=C["accent"],
                                highlightthickness=2)
                for w in frame.winfo_children():
                    try:
                        w.configure(bg=C["sel_bg"])
                    except:
                        pass
            else:
                frame.configure(bg=C["panel"], highlightbackground=C["border"],
                                highlightthickness=1)
                for w in frame.winfo_children():
                    try:
                        w.configure(bg=C["panel"])
                    except:
                        pass
        self._update_budget()

    def _make_activity_row(self, a, parent):
        aid, cat, name, desc, dur, cost, tip = a
        var = tk.BooleanVar()
        self.activity_vars[aid] = var

        cc = CAT_CLR.get(cat, "#888")
        row = tk.Frame(parent, bg=C["panel"], relief="flat", bd=0,
                       highlightthickness=1, highlightbackground=C["border"],
                       cursor="hand2")
        row.pack(fill="x", padx=4, pady=2)
        row.columnconfigure(2, weight=1)

        def toggle(_evt=None, _aid=aid, _var=var, _row=row, _cc=cc):
            _var.set(not _var.get())
            self._refresh_act_row(_aid, _row, _cc)
            self._update_budget()
        row.bind("<Button-1>", toggle)

        # 체크박스 흉내 (라벨)
        chk = tk.Label(row, text="☐", font=Font(family="맑은 고딕", size=13),
                       bg=C["panel"], fg=C["sub"], width=2)
        chk.grid(row=0, column=0, rowspan=2, padx=(8,4), pady=6)
        chk.bind("<Button-1>", toggle)

        # 카테고리 배지
        bl = tk.Label(row, text=cat, font=self.f_badge,
                      bg=cc, fg="white", padx=4, pady=2)
        bl.grid(row=0, column=1, rowspan=2, padx=4, pady=6, sticky="ns")
        bl.bind("<Button-1>", toggle)

        # 이름 + 설명
        nl = tk.Label(row, text=name, font=Font(family="맑은 고딕", size=11, weight="bold"),
                      bg=C["panel"], fg=C["text"], anchor="w")
        nl.grid(row=0, column=2, sticky="w", padx=4, pady=(6,0))
        nl.bind("<Button-1>", toggle)
        dl = tk.Label(row, text=desc[:44]+("…" if len(desc)>44 else ""),
                      font=self.f_small, bg=C["panel"], fg=C["sub"], anchor="w")
        dl.grid(row=1, column=2, sticky="w", padx=4, pady=(0,4))
        dl.bind("<Button-1>", toggle)

        # 메타
        meta_txt = f"⏱{dur}분  " + ("무료" if cost == 0 else f"+{cost//10000}만원")
        ml = tk.Label(row, text=meta_txt, font=self.f_small,
                      bg=C["panel"], fg=C["green"] if cost==0 else C["red"])
        ml.grid(row=0, column=3, rowspan=2, padx=8)
        ml.bind("<Button-1>", toggle)

        self._act_frames[aid] = (row, chk)
        return row

    def _refresh_act_row(self, aid, row, cc):
        var = self.activity_vars[aid]
        chk = self._act_frames[aid][1]
        if var.get():
            row.configure(bg="#F0FFF6", highlightbackground=cc, highlightthickness=2)
            chk.configure(text="☑", fg=cc, bg="#F0FFF6")
            for w in row.winfo_children():
                try:
                    if w != chk:
                        w.configure(bg="#F0FFF6")
                except:
                    pass
        else:
            row.configure(bg=C["panel"], highlightbackground=C["border"], highlightthickness=1)
            chk.configure(text="☐", fg=C["sub"], bg=C["panel"])
            for w in row.winfo_children():
                try:
                    if w != chk:
                        w.configure(bg=C["panel"])
                except:
                    pass
        cnt = sum(1 for v in self.activity_vars.values() if v.get())
        self._act_count_lbl.configure(text=f"{cnt}개 선택됨")

    def _build_activity_list(self):
        for w in self.act_inner.winfo_children():
            w.destroy()
        self._act_frames = {}
        self.activity_vars = {}
        for a in ACTIVITIES:
            self._make_activity_row(a, self.act_inner)

    def _filter_activities(self):
        cat = self._cat_var.get()
        for a in ACTIVITIES:
            frame = self._act_frames.get(a[0])
            if not frame:
                continue
            show = cat == "전체" or a[1] == cat
            if show:
                frame[0].pack(fill="x", padx=4, pady=2)
            else:
                frame[0].pack_forget()

    def _update_budget(self):
        sel_acts = [aid for aid, v in self.activity_vars.items() if v.get()]
        act_cost = sum(a[5] for a in ACTIVITIES if a[0] in sel_acts)
        total = min(400000 + (act_cost + 30000) * 23 + 700000, 2300000)
        self._budget_lbl.configure(text=f"예상 비용 약 {total//10000}만원 / 총 230만원")

    def _generate(self):
        vid = self.selected_venue.get()
        if not vid:
            messagebox.showwarning("장소 미선택", "장소를 1곳 선택해 주세요.")
            return
        sel_acts = [aid for aid, v in self.activity_vars.items() if v.get()]
        if not sel_acts:
            messagebox.showwarning("활동 미선택", "활동을 1개 이상 선택해 주세요.")
            return
        plan = generate_plan(vid, sel_acts)
        if plan:
            self._show_result(plan)
            self.nb.select(1)

    # ── 결과 탭 ───────────────────────────────────────────────────────────────
    def _build_result_tab(self):
        tab = self.tab_result
        # 버튼 바
        top = tk.Frame(tab, bg=C["bg"])
        top.pack(fill="x", padx=8, pady=(6,4))
        tk.Button(top, text="← 다시 선택", font=self.f_small,
                  bg=C["header"], fg="white", relief="flat", padx=10, pady=4,
                  cursor="hand2", activebackground="#2A3560", activeforeground="white",
                  command=lambda: self.nb.select(0)).pack(side="left")
        tk.Button(top, text="💾 텍스트 저장 (.txt)", font=self.f_small,
                  bg=C["green"], fg="white", relief="flat", padx=10, pady=4,
                  cursor="hand2", activebackground="#2BAE68", activeforeground="white",
                  command=self._save_txt).pack(side="right")

        # 스크롤 영역
        outer = tk.Frame(tab, bg=C["bg"])
        outer.pack(fill="both", expand=True, padx=8, pady=4)
        rc = tk.Canvas(outer, bg=C["bg"], highlightthickness=0)
        rsb = ttk.Scrollbar(outer, orient="vertical", command=rc.yview)
        self.result_inner = tk.Frame(rc, bg=C["bg"])
        self.result_inner.bind("<Configure>",
            lambda e: rc.configure(scrollregion=rc.bbox("all")))
        rc.create_window((0,0), window=self.result_inner, anchor="nw")
        rc.configure(yscrollcommand=rsb.set)
        rsb.pack(side="right", fill="y")
        rc.pack(side="left", fill="both", expand=True)
        rc.bind("<MouseWheel>", lambda e: rc.yview_scroll(-1*(e.delta//120),"units"))
        self._result_canvas = rc
        self._current_plan = None

    def _show_result(self, plan):
        self._current_plan = plan
        for w in self.result_inner.winfo_children():
            w.destroy()

        venue = plan["venue"]
        activities = plan["activities"]

        # 헤더
        hf = tk.Frame(self.result_inner, bg=C["header"])
        hf.pack(fill="x", padx=4, pady=(4,8))
        tk.Label(hf, text=f"🗓  야유회 기획안 — {venue[2]}",
                 font=self.f_title, bg=C["header"], fg="white").pack(anchor="w", padx=16, pady=(12,2))
        tk.Label(hf, text=f"📍 {venue[1]} | 👥 23명 | 💰 1인 10만원 | 🌸 4월 말",
                 font=self.f_small, bg=C["header"], fg="#A0AABB").pack(anchor="w", padx=16, pady=(0,12))

        # 섹션 헬퍼
        def add_section(title, color, lines):
            sf = tk.Frame(self.result_inner, bg=C["panel"],
                          highlightthickness=0, bd=0)
            sf.pack(fill="x", padx=8, pady=4)
            # 왼쪽 컬러 바
            bar = tk.Frame(sf, bg=color, width=5)
            bar.pack(side="left", fill="y")
            content = tk.Frame(sf, bg=C["panel"])
            content.pack(side="left", fill="both", expand=True, padx=10, pady=8)
            tk.Label(content, text=title, font=self.f_head,
                     bg=C["panel"], fg=color).pack(anchor="w")
            sep = tk.Frame(content, bg=color, height=1)
            sep.pack(fill="x", pady=(2,6))
            for line in lines:
                tk.Label(content, text=line, font=self.f_body,
                         bg=C["panel"], fg=C["text"], anchor="w",
                         wraplength=900, justify="left").pack(anchor="w", pady=1)

        # 장소
        add_section("📍 선택된 장소", C["accent"], [
            f"  {venue[2]} ({venue[1]})  ─  {'실내' if venue[4] else '야외'}",
            f"  {venue[3]}",
            f"  태그: {', '.join(venue[5])}",
        ])

        # 활동
        act_lines = []
        for a in activities:
            cost_txt = "무료" if a[5]==0 else f"+{a[5]//10000}만원/인"
            act_lines.append(f"  [{a[1]}]  {a[2]}  ({a[4]}분 / {cost_txt})")
            act_lines.append(f"       → {a[3]}")
        add_section("🎯 선택된 활동", C["accent2"], act_lines)

        # 일정
        sched_lines = [f"  {t}   {d}" for t, d in plan["schedule"]]
        add_section("⏰ 상세 일정", "#3DBE7A", sched_lines)

        # 비용
        act_t = plan["act_cost"] * 23
        add_section("💰 예상 비용 구조", "#C9674C", [
            f"  버스 대절 (왕복)                  40만원",
            f"  활동/체험비 (23인 합계)            {act_t//10000}만원",
            f"  점심 식사 (인당 1.5만원)           34만원",
            f"  저녁 식사 (인당 2만원)             46만원",
            f"  장소 대관/입장료                   약 30만원",
            f"  상품/경품                          20만원",
            f"  음료/간식/예비비                   잔액 분배",
            f"  ─────────────────────────────────────────",
            f"  예상 합계:  약 {plan['total']//10000}만원  /  총 230만원",
        ])

        # 준비물
        add_section("📋 준비물 리스트", "#9B59B6",
                    [f"  ☐  {item}" for item in plan["checklist"]])

        # 팁
        add_section("💡 운영 팁", C["accent"], [
            "  모든 프로그램은 자유 참여 원칙 (강요 금지)",
            "  팀 구성은 사전에 확정, 당일 어색함 최소화",
            "  날씨 대비: 실내 대체 공간 사전 예약 권장",
            "  담당자 1명 + 자원 진행자 2명으로 원활하게 운영 가능",
            "  이동 버스 안에서 간단한 아이스브레이킹 게임 추천",
        ])

        tk.Frame(self.result_inner, bg=C["bg"], height=20).pack()
        self._result_canvas.yview_moveto(0)

    def _save_txt(self):
        if not self._current_plan:
            messagebox.showinfo("알림", "먼저 기획안을 생성해 주세요.")
            return
        plan = self._current_plan
        path, _ = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile="야유회_기획안.txt",
            filetypes=[("Text Files","*.txt"),("All Files","*.*")]), None
        if isinstance(path, tuple):
            path = path[0]
        if not path:
            return
        venue = plan["venue"]
        with open(path, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write(f"  야유회 기획안 — {venue[2]}\n")
            f.write(f"  구미 근교 | 23명 | 4월 말 | 1인 10만원\n")
            f.write("="*60 + "\n\n")
            f.write(f"【장소】 {venue[2]} ({venue[1]})\n  {venue[3]}\n\n")
            f.write("【선택된 활동】\n")
            for a in plan["activities"]:
                f.write(f"  - [{a[1]}] {a[2]} : {a[3]}\n")
            f.write("\n【상세 일정】\n")
            for t, d in plan["schedule"]:
                f.write(f"  {t}  {d}\n")
            f.write(f"\n【예상 비용】 약 {plan['total']//10000}만원 / 230만원\n\n")
            f.write("【준비물】\n")
            for item in plan["checklist"]:
                f.write(f"  ☐ {item}\n")
        messagebox.showinfo("저장 완료", f"저장되었습니다:\n{path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
