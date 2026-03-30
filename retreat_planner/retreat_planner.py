#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기업 야유회 기획 도우미
구미 근교 | 23명 | 4월 말 | 1인 10만원
장소 + 활동 선택 → 기획안 자동 생성
"""

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# ─────────────────────────────────────────────────────────────────────────────
# 장소 후보 데이터
# (id, 지역, 장소명, 한줄 설명, 실내여부, 주요특징tags)
# ─────────────────────────────────────────────────────────────────────────────
VENUES = [
    # ── 구미 ──────────────────────────────────────────────────────────────────
    ("v01", "구미", "금오산 도립공원",       "울창한 숲 + 금오저수지 산책로, 피크닉 광장 완비",           False, ["자연","산책","피크닉"]),
    ("v02", "구미", "낙동강 생태공원",        "강변 잔디광장 + 자전거도로, 봄꽃 절정 명소",                False, ["자연","산책","피크닉"]),
    ("v03", "구미", "선산 전통시장",          "경북 5대 시장, 먹거리 골목 + 지역 특산품 탐방",            False, ["시장","맛집","체험"]),
    ("v04", "구미", "박정희 생가 + 인근 공원", "역사 문화 투어 + 주변 공원 피크닉",                        False, ["역사","문화","산책"]),
    ("v05", "구미", "구미 봉곡동 카페거리",   "감성 카페 밀집 지역, 자유 타임 & 브런치 코스 최적",        False, ["카페","감성","자유"]),
    # ── 김천 ──────────────────────────────────────────────────────────────────
    ("v06", "김천", "직지사",                 "천년 고찰 + 사찰 산책로, 봄 벚꽃 터널 유명",               False, ["사찰","자연","문화","봄꽃"]),
    ("v07", "김천", "직지문화공원",           "넓은 잔디광장 + 야외무대, 바베큐 가능 구역 있음",           False, ["잔디","피크닉","바베큐"]),
    ("v08", "김천", "김천 레인보우힐링파크",  "숲속 힐링 테마파크, 짚라인·산책로 패키지",                 False, ["힐링","자연","체험"]),
    ("v09", "김천", "황악산 자연휴양림",      "조용한 숲 속 힐링, 통나무집 캠프 분위기",                  False, ["숲","힐링","조용"]),
    ("v10", "김천", "중앙시장 + 요리체험관",  "재래시장 탐방 + 인근 요리 체험 시설 연계",                 True,  ["시장","요리","체험"]),
    # ── 상주 ──────────────────────────────────────────────────────────────────
    ("v11", "상주", "경천섬",                 "낙동강 벚꽃 명소 1위, 피크닉 잔디 + 자전거 코스",           False, ["봄꽃","피크닉","자전거","잔디"]),
    ("v12", "상주", "도남서원",               "조선 시대 서원 투어, 한옥 분위기 + 고택 체험",             False, ["역사","한옥","문화"]),
    ("v13", "상주", "갑장산 자연휴양림",      "숲 속 산책 + 계곡, 조용한 힐링 캠프 가능",                 False, ["숲","힐링","계곡"]),
    ("v14", "상주", "상주 전통시장",          "쌀·곶감 특산품 + 먹거리, 요리 대결 재료 구매 최적",        False, ["시장","맛집","특산품"]),
    ("v15", "상주", "고령가야 테마파크 (근교)","상주~고령 루트, 역사 테마 야외 관람",                      False, ["역사","테마","야외"]),
    # ── 대구 북부 ─────────────────────────────────────────────────────────────
    ("v16", "대구북부", "팔공산 케이블카 + 산책로", "케이블카 탑승 + 능선 산책, 대구 대표 봄 나들이 코스",  False, ["산","케이블카","산책","봄꽃"]),
    ("v17", "대구북부", "동화사",             "팔공산 천년 고찰, 봄꽃 명소 + 계곡 산책",                  False, ["사찰","봄꽃","계곡"]),
    ("v18", "대구북부", "대구 염색공예관",    "천연 염색 체험 + 전통 공예, 단체 예약 가능",                True,  ["공방","체험","실내"]),
    ("v19", "대구북부", "대구 북구 공방거리", "도자기·캔들·리스 공방 밀집, 단체 체험 패키지 가능",         True,  ["공방","캔들","도자기","실내"]),
    ("v20", "대구북부", "파군재 캠핑&피크닉", "대구 근교 한적한 자연 피크닉 + 불멍 가능",                  False, ["자연","피크닉","캠핑"]),
    # ── 안동/경주 (1시간 이내) ────────────────────────────────────────────────
    ("v21", "안동", "하회마을 한옥 체험관",   "유네스코 세계유산, 한옥 체험 + 전통 놀이 단체 가능",        True,  ["한옥","전통","역사","문화"]),
    ("v22", "안동", "월영교 + 안동 카페거리", "야경 명소 + 감성 카페, 저녁 코스로 최적",                  False, ["감성","카페","야경"]),
    ("v23", "경주", "교촌마을 + 경주 맛집",   "첨성대·교촌 한옥 마을, 맛집 투어 + 역사 산책",             False, ["역사","맛집","한옥","산책"]),
    # ── 기타 체험 시설 ────────────────────────────────────────────────────────
    ("v24", "기타", "인근 펜션 (전용 대관)",  "전용 마당 + 바베큐 + 수영장 옵션, 프라이빗 운영",           True,  ["프라이빗","바베큐","힐링"]),
    ("v25", "기타", "청도 와인터널",          "140년 된 철도 터널 와인 숙성 공간, 이색 투어 + 시음",       True,  ["이색","와인","터널","실내"]),
    ("v26", "기타", "성주 참외밭 농촌 체험",  "봄 참외 수확 체험 + 전원 피크닉, 이색 협업 미션 가능",     False, ["농촌","체험","이색"]),
]

# ─────────────────────────────────────────────────────────────────────────────
# 활동 후보 데이터
# (id, 카테고리, 활동명, 한줄 설명, 소요시간(분), 인당비용(원), 팁)
# ─────────────────────────────────────────────────────────────────────────────
ACTIVITIES = [
    # ── 미션/게임형 ───────────────────────────────────────────────────────────
    ("a01","미션/게임형","감성 포토 미션",     "지정 스팟 5곳 팀별 창의 사진 → 투표로 우승팀 선발",         90, 0,     "스마트폰 충전 상태 확인"),
    ("a02","미션/게임형","미스터리 미션 랠리", "단서 카드 추적 → 지역 명소 탐방하며 보물 찾기",             120, 5000, "QR코드 활용 시 자동 채점 가능"),
    ("a03","미션/게임형","회사 퀴즈 배틀",     "우리 회사 상식 + 일반 퀴즈, 카훗(Kahoot) 앱 활용",          60, 0,    "사전 문제 준비 필수 (30문항)"),
    ("a04","미션/게임형","맛집 스탬프 투어",   "팀별 맛집 3~4곳 스탬프 수집 → 품평회 발표",                 120, 20000,"사전 식당 동선 답사 필수"),
    ("a05","미션/게임형","보물 지도 빙고",     "장소 이름 빙고판 + 인증샷 조건, 이동 중 진행 가능",          60, 0,    "이동 버스 안에서도 진행 가능"),
    ("a06","미션/게임형","팀 탐험대 OX퀴즈",  "지역 역사/상식 OX, 틀리면 미션(노래 한 소절 등) 수행",       45, 0,    "부담 없는 벌칙 설계 중요"),
    ("a07","미션/게임형","런닝맨 명함 뜯기",   "팀별 명함 지키기 + 빼앗기 게임, 저강도 실외 미션",          60, 0,    "자유 참여 보장, 구경도 OK"),
    # ── 감성/힐링형 ───────────────────────────────────────────────────────────
    ("a08","감성/힐링형","자유 피크닉 타임",   "돗자리 + 도시락 + 음악, 그냥 쉬는 시간",                    90, 8000, "블루투스 스피커 필수"),
    ("a09","감성/힐링형","보드게임 자유 타임", "루미큐브·할리갈리·UNO 등, 원하는 사람끼리 자연스럽게",       90, 3000, "강요 없는 분위기 조성이 핵심"),
    ("a10","감성/힐링형","감성 카페 투어",     "뷰 맛집 카페 1~2곳 방문, 디저트 + 자유 담소",               60, 8000, "사전 예약 필수 (단체 23명)"),
    ("a11","감성/힐링형","한옥/숲 산책",       "자연 속 자유 산책, 사진 찍으며 힐링",                        60, 0,    "담당자 동선 파악만 해두면 OK"),
    ("a12","감성/힐링형","야외 팀 독서",       "1인 1책 준비 → 잔디에서 20분 침묵 독서 후 한마디 공유",      60, 0,    "독특하고 기억에 남는 경험"),
    ("a13","감성/힐링형","소원 카드 쓰기",     "봄에 쓰는 올해 소원 엽서 → 팀끼리 공유 or 비공개 보관",      30, 2000, "소품(엽서+펜) 사전 준비"),
    # ── 가벼운 경쟁형 ─────────────────────────────────────────────────────────
    ("a14","경쟁형","전통 놀이 배틀",          "투호·윷놀이·제기차기 팀 대항, 점수 합산 시상",              90, 0,    "한옥/문화관 시설 활용 시 도구 제공"),
    ("a15","경쟁형","줄다리기/이어달리기",     "초저강도 팀 대항, 응원전 포함 → 분위기 최고조",              60, 0,    "운동화 착용 권장"),
    ("a16","경쟁형","작품 감상 투표",          "공방 체험 후 '최고 작품' 익명 투표 + 시상",                  30, 0,    "모두가 상받는 구조 추천"),
    ("a17","경쟁형","요리 완성 심사",          "팀 요리 대결 후 전원 투표 → 5가지 부문 시상",               60, 0,    "꼴찌 팀도 위로상 필수"),
    ("a18","경쟁형","팀 사진전 투표",          "포토 미션 결과물 실시간 공유 → 좋아요 수 집계",              30, 0,    "단체 카톡방 활용"),
    # ── 협업형 ───────────────────────────────────────────────────────────────
    ("a19","협업형","팀 바베큐",               "역할 분담(굽기·반찬·음료)으로 함께 만드는 저녁 식사",        90, 22000,"그릴·숯·착화제 사전 준비"),
    ("a20","협업형","캔들 공방 체험",          "향기·색상 선택 → 강사 지도 하에 캔들 직접 제작",            120, 35000,"완성품 집에 가져감 → 기억 지속"),
    ("a21","협업형","도자기 공방 체험",        "물레 or 핸드빌딩으로 나만의 그릇 만들기",                   120, 40000,"2~4주 후 택배 수령 가능 시설 선택"),
    ("a22","협업형","봄꽃 리스 만들기",        "생화·조화로 봄꽃 리스 제작, 강사 진행으로 부담 없음",        90, 30000,"4월 봄 시즌과 완벽히 맞음"),
    ("a23","협업형","팀 상징 만들기",          "한지·소품으로 팀 이름+슬로건+마크 제작 후 발표",             60, 5000, "팀 정체성 강화 효과"),
    ("a24","협업형","팀 요리 대결",            "시장 재료 구매 → 지정 요리 1가지 팀별 완성 + 시식",          120, 15000,"비빔밥·전 부치기 등 쉬운 메뉴 추천"),
    ("a25","협업형","팀원 소통 카드게임",      "'우리팀 몰래카메라' - 사전 설문 → 누구일까요 퀴즈",          60, 3000, "사전 설문지 배포 필수 (1주 전)"),
    ("a26","협업형","팀 영상 제작",            "오늘 하루를 15초 릴스로 편집 → 저녁 상영회",                 60, 0,    "편집 잘하는 팀원 미리 파악"),
]

# ─────────────────────────────────────────────────────────────────────────────
# 기획안 자동 생성 로직
# ─────────────────────────────────────────────────────────────────────────────
def generate_plan(venue_id, activity_ids):
    venue = next((v for v in VENUES if v[0] == venue_id), None)
    activities = [a for a in ACTIVITIES if a[0] in activity_ids]
    if not venue or not activities:
        return None

    # 비용 계산
    activity_cost = sum(a[5] for a in activities)
    bus_cost = 400000
    meal_cost = 150000  # 점심+저녁 per person equivalent (총액 기준)
    venue_cost = 300000
    prize_cost = 200000
    misc_cost = 200000
    estimated_total = bus_cost + (activity_cost * 23) + meal_cost * 23 // 10000 * 10000 + venue_cost + prize_cost + misc_cost

    # 일정 자동 생성
    schedule = [("09:00", "출발 (회사 집합, 버스 이동)")]
    t = 10  # 10시 도착
    schedule.append((f"{t:02d}:00", f"도착 — {venue[2]} 체크인 및 오리엔테이션"))
    t_min = 30
    schedule.append((f"{t:02d}:{t_min:02d}", "팀 구성 및 오늘 일정 소개 (아이스브레이킹)"))
    current_hour = 11
    current_min = 0

    for act in activities:
        dur = act[4]
        h = current_hour
        m = current_min
        end_m = m + dur
        end_h = h + end_m // 60
        end_m = end_m % 60
        schedule.append((f"{h:02d}:{m:02d}", f"▶ {act[2]} | {act[3][:30]}"))
        current_hour = end_h
        current_min = end_m
        if current_hour == 12 and current_min >= 0 and "점심" not in [s[1] for s in schedule]:
            if current_hour >= 12:
                schedule.append((f"13:00", "점심 식사 (현지 맛집 or 도시락)"))
                current_hour = 14
                current_min = 0

    if "13:00" not in [s[0] for s in schedule]:
        schedule.append(("12:30", "점심 식사 (현지 맛집 or 도시락)"))
    schedule.append((f"{max(current_hour,17):02d}:30", "저녁 식사 + 자유 담소"))
    schedule.append((f"{max(current_hour,19):02d}:30", "귀사 이동"))
    schedule.append((f"{max(current_hour,21):02d}:00", "해산"))

    # 준비물 자동 생성
    checklist = ["버스 예약 확인", "명찰", "구급약품", "블루투스 스피커"]
    for act in activities:
        if act[0] == "a01": checklist += ["충전된 스마트폰", "카메라"]
        if act[0] == "a02": checklist += ["미션 키트 (단서 카드)", "QR코드 카드"]
        if act[0] == "a03": checklist += ["카훗 문제 준비", "인터넷 가능 기기"]
        if act[0] == "a04": checklist += ["스탬프 투어 코스지", "평가지"]
        if act[0] == "a08": checklist += ["돗자리 5개", "파라솔 3개", "도시락"]
        if act[0] == "a09": checklist += ["보드게임 세트 (루미큐브/UNO)"]
        if act[0] == "a14": checklist += ["투호 세트", "윷놀이 세트"]
        if act[0] == "a19": checklist += ["바베큐 그릴", "숯+착화제", "고기+채소"]
        if act[0] in ("a20","a21","a22","a23","a24"): checklist += ["공방 사전 예약 (필수)", "앞치마"]
        if act[0] == "a25": checklist += ["팀원 사전 설문지"]

    checklist = list(dict.fromkeys(checklist))  # 중복 제거

    return {
        "venue": venue,
        "activities": activities,
        "schedule": schedule,
        "checklist": checklist,
        "estimated_total": min(estimated_total, 2300000),  # 예산 캡
        "activity_cost_per_person": activity_cost,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 스타일 상수
# ─────────────────────────────────────────────────────────────────────────────
COLOR_BG       = "#F7F8FA"
COLOR_SIDEBAR  = "#1E2235"
COLOR_ACCENT   = "#5B8DEF"
COLOR_ACCENT2  = "#F4A261"
COLOR_SUCCESS  = "#52C97E"
COLOR_CARD     = "#FFFFFF"
COLOR_BORDER   = "#E2E6EA"
COLOR_TEXT     = "#2D3142"
COLOR_SUB      = "#7A8099"
COLOR_HEADER   = "#1E2235"

CAT_COLORS = {
    "미션/게임형": "#5B8DEF",
    "감성/힐링형": "#F4A261",
    "경쟁형":      "#E25A5A",
    "협업형":      "#52C97E",
}

REGION_COLORS = {
    "구미":    "#5B8DEF",
    "김천":    "#52C97E",
    "상주":    "#F4A261",
    "대구북부":"#C9674C",
    "안동":    "#9B59B6",
    "경주":    "#E67E22",
    "기타":    "#7F8C8D",
}

# ─────────────────────────────────────────────────────────────────────────────
# 위젯: 장소 선택
# ─────────────────────────────────────────────────────────────────────────────
class VenueCard(QFrame):
    selected = pyqtSignal(str, bool)

    def __init__(self, venue_data, parent=None):
        super().__init__(parent)
        self.vid = venue_data[0]
        self.region = venue_data[1]
        self.name = venue_data[2]
        self.desc = venue_data[3]
        self.indoor = venue_data[4]
        self.tags = venue_data[5]
        self._checked = False
        self._setup_ui()

    def _setup_ui(self):
        self.setFixedHeight(72)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLOR_CARD};
                border: 1.5px solid {COLOR_BORDER};
                border-radius: 10px;
                margin: 2px 0px;
            }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        # 체크박스 (사실상 라디오지만 카드 클릭으로 동작)
        self.radio = QRadioButton()
        self.radio.setAttribute(Qt.WA_TransparentForMouseEvents)
        layout.addWidget(self.radio)

        # 지역 배지
        badge = QLabel(self.region)
        rc = REGION_COLORS.get(self.region, "#888")
        badge.setStyleSheet(f"""
            background: {rc}22; color: {rc}; border: 1px solid {rc}66;
            border-radius: 4px; padding: 1px 6px; font-size: 11px; font-weight: bold;
        """)
        badge.setFixedWidth(58)
        badge.setAlignment(Qt.AlignCenter)
        layout.addWidget(badge)

        # 텍스트
        text_w = QWidget()
        text_l = QVBoxLayout(text_w)
        text_l.setContentsMargins(0,0,0,0)
        text_l.setSpacing(2)
        name_l = QLabel(self.name)
        name_l.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {COLOR_TEXT};")
        desc_l = QLabel(self.desc)
        desc_l.setStyleSheet(f"font-size: 11px; color: {COLOR_SUB};")
        text_l.addWidget(name_l)
        text_l.addWidget(desc_l)
        layout.addWidget(text_w, 1)

        # 실내 여부
        indoor_l = QLabel("실내" if self.indoor else "야외")
        ic = "#5B8DEF" if self.indoor else "#52C97E"
        indoor_l.setStyleSheet(f"font-size: 10px; color: {ic}; border: 1px solid {ic}; border-radius: 3px; padding: 1px 5px;")
        layout.addWidget(indoor_l)

    def setChecked(self, v):
        self._checked = v
        self.radio.setChecked(v)
        bc = COLOR_ACCENT if v else COLOR_BORDER
        bw = "2px" if v else "1.5px"
        self.setStyleSheet(f"""
            QFrame {{
                background: {"#EEF3FF" if v else COLOR_CARD};
                border: {bw} solid {bc};
                border-radius: 10px;
                margin: 2px 0px;
            }}
        """)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.selected.emit(self.vid, True)

# ─────────────────────────────────────────────────────────────────────────────
# 위젯: 활동 선택
# ─────────────────────────────────────────────────────────────────────────────
class ActivityCard(QFrame):
    toggled = pyqtSignal(str, bool)

    def __init__(self, activity_data, parent=None):
        super().__init__(parent)
        self.aid = activity_data[0]
        self.cat = activity_data[1]
        self.name = activity_data[2]
        self.desc = activity_data[3]
        self.dur = activity_data[4]
        self.cost = activity_data[5]
        self.tip = activity_data[6]
        self._checked = False
        self._setup_ui()

    def _setup_ui(self):
        self.setFixedHeight(72)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLOR_CARD};
                border: 1.5px solid {COLOR_BORDER};
                border-radius: 10px;
                margin: 2px 0px;
            }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        self.chk = QCheckBox()
        self.chk.setAttribute(Qt.WA_TransparentForMouseEvents)
        layout.addWidget(self.chk)

        # 카테고리 배지
        cc = CAT_COLORS.get(self.cat, "#888")
        badge = QLabel(self.cat)
        badge.setStyleSheet(f"""
            background: {cc}22; color: {cc}; border: 1px solid {cc}66;
            border-radius: 4px; padding: 1px 5px; font-size: 10px; font-weight: bold;
        """)
        badge.setFixedWidth(72)
        badge.setAlignment(Qt.AlignCenter)
        layout.addWidget(badge)

        text_w = QWidget()
        text_l = QVBoxLayout(text_w)
        text_l.setContentsMargins(0,0,0,0)
        text_l.setSpacing(2)
        name_l = QLabel(self.name)
        name_l.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {COLOR_TEXT};")
        desc_l = QLabel(self.desc[:45] + ("..." if len(self.desc) > 45 else ""))
        desc_l.setStyleSheet(f"font-size: 11px; color: {COLOR_SUB};")
        text_l.addWidget(name_l)
        text_l.addWidget(desc_l)
        layout.addWidget(text_w, 1)

        meta_w = QWidget()
        meta_l = QVBoxLayout(meta_w)
        meta_l.setContentsMargins(0,0,0,0)
        meta_l.setSpacing(2)
        dur_l = QLabel(f"⏱ {self.dur}분")
        dur_l.setStyleSheet(f"font-size: 10px; color: {COLOR_SUB};")
        cost_l = QLabel("무료" if self.cost == 0 else f"+{self.cost//10000}만원/인")
        cost_l.setStyleSheet(f"font-size: 10px; color: {'#52C97E' if self.cost == 0 else '#E25A5A'};")
        meta_l.addWidget(dur_l)
        meta_l.addWidget(cost_l)
        layout.addWidget(meta_w)

    def setChecked(self, v):
        self._checked = v
        self.chk.setChecked(v)
        bc = CAT_COLORS.get(self.cat, COLOR_ACCENT) if v else COLOR_BORDER
        bw = "2px" if v else "1.5px"
        bg = f"{CAT_COLORS.get(self.cat,'#5B8DEF')}11" if v else COLOR_CARD
        self.setStyleSheet(f"""
            QFrame {{
                background: {bg};
                border: {bw} solid {bc};
                border-radius: 10px;
                margin: 2px 0px;
            }}
        """)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._checked = not self._checked
            self.setChecked(self._checked)
            self.toggled.emit(self.aid, self._checked)

# ─────────────────────────────────────────────────────────────────────────────
# 선택 패널
# ─────────────────────────────────────────────────────────────────────────────
class SelectionPanel(QWidget):
    plan_generated = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_venue = None
        self.selected_activities = set()
        self._venue_cards = {}
        self._activity_cards = {}
        self._setup_ui()

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        # ── 왼쪽: 장소 선택 ────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(420)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0,0,0,0)
        ll.setSpacing(8)

        lh = QLabel("📍 장소 선택  (1개 선택)")
        lh.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLOR_TEXT}; padding: 4px 0;")
        ll.addWidget(lh)

        # 지역 필터
        rf = QHBoxLayout()
        self.region_btns = {}
        for region in ["전체", "구미", "김천", "상주", "대구북부", "안동/경주", "기타"]:
            btn = QPushButton(region)
            btn.setCheckable(True)
            btn.setChecked(region == "전체")
            btn.setFixedHeight(28)
            btn.setStyleSheet(self._filter_btn_style(region == "전체"))
            btn.clicked.connect(lambda c, r=region: self._filter_venues(r))
            rf.addWidget(btn)
            self.region_btns[region] = btn
        ll.addLayout(rf)

        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        vlist = QWidget()
        self.venue_layout = QVBoxLayout(vlist)
        self.venue_layout.setContentsMargins(4, 4, 4, 4)
        self.venue_layout.setSpacing(4)

        for v in VENUES:
            card = VenueCard(v)
            card.selected.connect(self._on_venue_selected)
            self.venue_layout.addWidget(card)
            self._venue_cards[v[0]] = card
        self.venue_layout.addStretch()

        sa.setWidget(vlist)
        ll.addWidget(sa)
        root.addWidget(left)

        # 구분선
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color: {COLOR_BORDER};")
        root.addWidget(sep)

        # ── 오른쪽: 활동 선택 ──────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0,0,0,0)
        rl.setSpacing(8)

        rh_row = QHBoxLayout()
        rh = QLabel("🎯 활동 선택  (여러 개 선택 가능)")
        rh.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLOR_TEXT}; padding: 4px 0;")
        rh_row.addWidget(rh, 1)
        self.act_count_l = QLabel("0개 선택됨")
        self.act_count_l.setStyleSheet(f"font-size: 12px; color: {COLOR_ACCENT}; font-weight: bold;")
        rh_row.addWidget(self.act_count_l)
        rl.addLayout(rh_row)

        # 카테고리 필터
        cf = QHBoxLayout()
        self.cat_btns = {}
        for cat in ["전체", "미션/게임형", "감성/힐링형", "경쟁형", "협업형"]:
            btn = QPushButton(cat)
            btn.setCheckable(True)
            btn.setChecked(cat == "전체")
            btn.setFixedHeight(28)
            btn.setStyleSheet(self._filter_btn_style(cat == "전체"))
            btn.clicked.connect(lambda c, ct=cat: self._filter_activities(ct))
            cf.addWidget(btn)
            self.cat_btns[cat] = btn
        rl.addLayout(cf)

        sa2 = QScrollArea()
        sa2.setWidgetResizable(True)
        sa2.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        sa2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        alist = QWidget()
        self.activity_layout = QVBoxLayout(alist)
        self.activity_layout.setContentsMargins(4, 4, 4, 4)
        self.activity_layout.setSpacing(4)

        for a in ACTIVITIES:
            card = ActivityCard(a)
            card.toggled.connect(self._on_activity_toggled)
            self.activity_layout.addWidget(card)
            self._activity_cards[a[0]] = card
        self.activity_layout.addStretch()
        sa2.setWidget(alist)
        rl.addWidget(sa2)

        # 생성 버튼 영역
        btn_row = QHBoxLayout()
        self.budget_l = QLabel("예상 비용: 계산 중...")
        self.budget_l.setStyleSheet(f"font-size: 12px; color: {COLOR_SUB};")
        btn_row.addWidget(self.budget_l, 1)
        gen_btn = QPushButton("  기획안 생성  ▶")
        gen_btn.setFixedHeight(44)
        gen_btn.setFixedWidth(160)
        gen_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLOR_ACCENT}; color: white;
                border-radius: 10px; font-size: 14px; font-weight: bold;
            }}
            QPushButton:hover {{ background: #4A7ADE; }}
            QPushButton:pressed {{ background: #3A6ACE; }}
        """)
        gen_btn.clicked.connect(self._generate)
        btn_row.addWidget(gen_btn)
        rl.addLayout(btn_row)
        root.addWidget(right)

    def _filter_btn_style(self, active):
        if active:
            return f"""
                QPushButton {{ background: {COLOR_ACCENT}; color: white;
                    border-radius: 6px; font-size: 11px; font-weight: bold; padding: 0 8px; }}
            """
        return f"""
            QPushButton {{ background: {COLOR_CARD}; color: {COLOR_TEXT};
                border: 1px solid {COLOR_BORDER}; border-radius: 6px; font-size: 11px; padding: 0 8px; }}
            QPushButton:hover {{ background: #EEF3FF; }}
        """

    def _filter_venues(self, region):
        for r, b in self.region_btns.items():
            b.setChecked(r == region)
            b.setStyleSheet(self._filter_btn_style(r == region))
        for v in VENUES:
            card = self._venue_cards[v[0]]
            if region == "전체":
                card.setVisible(True)
            elif region == "안동/경주":
                card.setVisible(v[1] in ("안동", "경주"))
            else:
                card.setVisible(v[1] == region)

    def _filter_activities(self, cat):
        for c, b in self.cat_btns.items():
            b.setChecked(c == cat)
            b.setStyleSheet(self._filter_btn_style(c == cat))
        for a in ACTIVITIES:
            card = self._activity_cards[a[0]]
            card.setVisible(cat == "전체" or a[1] == cat)

    def _on_venue_selected(self, vid, _):
        self.selected_venue = vid
        for v_id, card in self._venue_cards.items():
            card.setChecked(v_id == vid)
        self._update_budget()

    def _on_activity_toggled(self, aid, checked):
        if checked:
            self.selected_activities.add(aid)
        else:
            self.selected_activities.discard(aid)
        n = len(self.selected_activities)
        self.act_count_l.setText(f"{n}개 선택됨")
        self._update_budget()

    def _update_budget(self):
        act_cost = sum(a[5] for a in ACTIVITIES if a[0] in self.selected_activities)
        total_approx = 400000 + (act_cost + 30000) * 23 + 700000
        total_approx = min(total_approx, 2300000)
        self.budget_l.setText(f"예상 비용 약 {total_approx//10000}만원 / 총 230만원")

    def _generate(self):
        if not self.selected_venue:
            QMessageBox.warning(self, "장소 미선택", "장소를 1곳 선택해 주세요.")
            return
        if len(self.selected_activities) == 0:
            QMessageBox.warning(self, "활동 미선택", "활동을 1개 이상 선택해 주세요.")
            return
        plan = generate_plan(self.selected_venue, list(self.selected_activities))
        if plan:
            self.plan_generated.emit(plan)

# ─────────────────────────────────────────────────────────────────────────────
# 결과 패널
# ─────────────────────────────────────────────────────────────────────────────
class ResultPanel(QScrollArea):
    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet("QScrollArea { border: none; background: #F7F8FA; }")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._container = QWidget()
        self._container.setStyleSheet(f"background: {COLOR_BG};")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(32, 24, 32, 32)
        self._layout.setSpacing(16)
        self.setWidget(self._container)

    def load_plan(self, plan):
        # 기존 위젯 제거
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        venue = plan["venue"]
        activities = plan["activities"]

        # ── 헤더 ─────────────────────────────────────────────────────────────
        header = QFrame()
        header.setStyleSheet(f"""
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 {COLOR_HEADER}, stop:1 #2A3560);
            border-radius: 14px;
        """)
        hl = QVBoxLayout(header)
        hl.setContentsMargins(24, 20, 24, 20)
        title = QLabel(f"🗓  야유회 기획안 — {venue[2]}")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: white;")
        sub = QLabel(f"📍 {venue[1]} | 👥 23명 | 💰 1인 10만원 | 🌸 4월 말")
        sub.setStyleSheet("font-size: 13px; color: #B0BCDD; margin-top: 4px;")
        back_btn = QPushButton("← 다시 선택")
        back_btn.setFixedHeight(32)
        back_btn.setFixedWidth(100)
        back_btn.setStyleSheet("""
            QPushButton { background: rgba(255,255,255,0.15); color: white;
                border: 1px solid rgba(255,255,255,0.3); border-radius: 7px; font-size: 12px; }
            QPushButton:hover { background: rgba(255,255,255,0.25); }
        """)
        back_btn.clicked.connect(self.back_requested.emit)
        top_row = QHBoxLayout()
        top_row.addWidget(title, 1)
        top_row.addWidget(back_btn)
        hl.addLayout(top_row)
        hl.addWidget(sub)
        self._layout.addWidget(header)

        # ── 장소 정보 ─────────────────────────────────────────────────────────
        self._add_section("📍 선택된 장소", COLOR_ACCENT, [
            f"<b>{venue[2]}</b> ({venue[1]})",
            f"{venue[3]}",
            f"{'실내' if venue[4] else '야외'} | 태그: {', '.join(venue[5])}",
        ])

        # ── 선택된 활동 ───────────────────────────────────────────────────────
        act_lines = []
        for a in activities:
            cc = CAT_COLORS.get(a[1], "#888")
            act_lines.append(
                f"<span style='background:{cc}22;color:{cc};border:1px solid {cc}66;"
                f"border-radius:3px;padding:1px 5px;font-size:11px;'>{a[1]}</span>"
                f"&nbsp; <b>{a[2]}</b> — {a[3][:40]}{'...' if len(a[3])>40 else ''}"
                f"&nbsp; <span style='color:#7A8099;font-size:11px;'>({a[4]}분"
                + (f" / +{a[5]//10000}만원" if a[5] > 0 else " / 무료") + ")</span>"
            )
        self._add_section("🎯 선택된 활동", COLOR_ACCENT2, act_lines, html=True)

        # ── 일정표 ────────────────────────────────────────────────────────────
        sched_lines = [f"<b>{t}</b>&nbsp;&nbsp;{desc}" for t, desc in plan["schedule"]]
        self._add_section("⏰ 상세 일정", "#6A9E6A", sched_lines, html=True)

        # ── 예상 비용 ─────────────────────────────────────────────────────────
        total = plan["estimated_total"]
        act_total = plan["activity_cost_per_person"] * 23
        budget_lines = [
            f"버스 대절 (왕복): <b>40만원</b>",
            f"활동/체험비 (23인): <b>{act_total//10000}만원</b>",
            f"점심 식사 (인당 1.5만원): <b>34만원</b>",
            f"저녁 식사 (인당 2만원): <b>46만원</b>",
            f"장소 대관/입장료: <b>약 30만원</b>",
            f"상품/경품: <b>20만원</b>",
            f"음료/간식/예비비: <b>잔액 분배</b>",
            f"<span style='font-size:14px;color:#E25A5A;'><b>예상 합계: 약 {total//10000}만원 / 230만원</b></span>",
        ]
        self._add_section("💰 예상 비용 구조", "#C9674C", budget_lines, html=True)

        # ── 준비물 ────────────────────────────────────────────────────────────
        self._add_section("📋 준비물 리스트", "#9B59B6", [f"☐  {item}" for item in plan["checklist"]])

        # ── 운영 팁 ───────────────────────────────────────────────────────────
        tip_lines = ["모든 프로그램은 <b>자유 참여</b> 원칙 (강요 금지)",
                     "팀 구성은 <b>사전에 확정</b>, 당일 어색함 최소화",
                     "<b>날씨 대비:</b> 실내 대체 공간 사전 예약 권장",
                     "담당자 1명 + 자원 진행자 2명으로 원활하게 운영 가능",
                     "이동 버스 안에서 간단한 <b>아이스브레이킹 게임</b> 추천"]
        self._add_section("💡 운영 팁", "#5B8DEF", tip_lines, html=True)

        # ── 인쇄 버튼 ─────────────────────────────────────────────────────────
        print_btn = QPushButton("🖨  기획안 텍스트 저장 (.txt)")
        print_btn.setFixedHeight(44)
        print_btn.setStyleSheet(f"""
            QPushButton {{ background: {COLOR_SUCCESS}; color: white;
                border-radius: 10px; font-size: 13px; font-weight: bold; }}
            QPushButton:hover {{ background: #40B86C; }}
        """)
        print_btn.clicked.connect(lambda: self._save_txt(plan))
        self._layout.addWidget(print_btn)
        self._layout.addStretch()

    def _add_section(self, title, color, lines, html=False):
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{ background: {COLOR_CARD}; border-radius: 12px;
                border-left: 4px solid {color}; }}
        """)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 14, 16, 14)
        cl.setSpacing(8)

        title_l = QLabel(title)
        title_l.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {color};")
        cl.addWidget(title_l)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {color}44;"); cl.addWidget(sep)

        for line in lines:
            lbl = QLabel(line)
            lbl.setStyleSheet(f"font-size: 12px; color: {COLOR_TEXT}; padding: 1px 0;")
            lbl.setWordWrap(True)
            if html:
                lbl.setTextFormat(Qt.RichText)
            cl.addWidget(lbl)

        self._layout.addWidget(card)

    def _save_txt(self, plan):
        venue = plan["venue"]
        activities = plan["activities"]
        path, _ = QFileDialog.getSaveFileName(self, "저장", "야유회_기획안.txt", "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"  야유회 기획안 — {venue[2]}\n")
            f.write(f"  구미 근교 | 23명 | 4월 말 | 1인 10만원\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"【장소】 {venue[2]} ({venue[1]})\n")
            f.write(f"  {venue[3]}\n\n")
            f.write("【선택된 활동】\n")
            for a in activities:
                f.write(f"  - [{a[1]}] {a[2]} : {a[3]}\n")
            f.write("\n【상세 일정】\n")
            for t, d in plan["schedule"]:
                f.write(f"  {t}  {d}\n")
            f.write(f"\n【예상 비용】 약 {plan['estimated_total']//10000}만원 / 230만원\n\n")
            f.write("【준비물】\n")
            for item in plan["checklist"]:
                f.write(f"  ☐ {item}\n")
        QMessageBox.information(self, "저장 완료", f"저장되었습니다:\n{path}")

# ─────────────────────────────────────────────────────────────────────────────
# 메인 윈도우
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌸 야유회 기획 도우미 — 구미 근교 23인 4월 말")
        self.setMinimumSize(1100, 720)
        self.resize(1300, 800)
        self._setup_ui()
        self._apply_global_style()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 상단 타이틀 바
        title_bar = QFrame()
        title_bar.setFixedHeight(56)
        title_bar.setStyleSheet(f"""
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 {COLOR_HEADER}, stop:1 #2A3560);
        """)
        tb_l = QHBoxLayout(title_bar)
        tb_l.setContentsMargins(24, 0, 24, 0)
        t1 = QLabel("🌸  야유회 기획 도우미")
        t1.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        t2 = QLabel("구미 근교 | 23명 | 4월 말 | 1인 10만원")
        t2.setStyleSheet("font-size: 12px; color: #B0BCDD;")
        tb_l.addWidget(t1)
        tb_l.addStretch()
        tb_l.addWidget(t2)
        layout.addWidget(title_bar)

        # 스택 위젯 (선택 화면 ↔ 결과 화면)
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        self.selection_panel = SelectionPanel()
        self.result_panel = ResultPanel()

        self.stack.addWidget(self.selection_panel)  # index 0
        self.stack.addWidget(self.result_panel)       # index 1

        self.selection_panel.plan_generated.connect(self._show_result)
        self.result_panel.back_requested.connect(lambda: self.stack.setCurrentIndex(0))

    def _show_result(self, plan):
        self.result_panel.load_plan(plan)
        self.stack.setCurrentIndex(1)

    def _apply_global_style(self):
        self.setStyleSheet(f"""
            QWidget {{ font-family: "맑은 고딕", "Malgun Gothic", "Noto Sans KR", sans-serif;
                background: {COLOR_BG}; }}
            QScrollBar:vertical {{ background: {COLOR_BG}; width: 8px; border-radius: 4px; }}
            QScrollBar::handle:vertical {{ background: #C5CDE0; border-radius: 4px; min-height: 30px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        """)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("야유회 기획 도우미")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
