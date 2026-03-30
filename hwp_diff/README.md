# 문서 변경점 비교기 (HWP Diff Tool)

HWP/HWPX/DOCX/TXT 문서를 비교하여 변경사항을 분석하고 Excel로 내보내는 도구입니다.

## 기능

- **다양한 형식 지원**: HWPX, HWP(변환필요), DOCX, TXT
- **변경 유형 탐지**: 추가, 삭제, 수정, 이동, 서식변경
- **표 비교**: 셀 단위 비교, 행/열 추가/삭제 탐지
- **중요도 분류**: 수치값 변경은 높음, 의미적 변경은 중간, 공백 변경은 낮음
- **Excel 내보내기**: 전체 변경사항, 표 변경, 제목 변경, 요약 통계 시트
- **나란히 보기**: HTML 하이라이팅으로 변경된 부분 시각화
- **동기화 스크롤**: 양쪽 문서가 동시에 스크롤됨

## 설치

```bash
cd hwp_diff
pip install -r requirements.txt
```

## 실행

```bash
python main.py
```

## 사용법

1. "기준문서 찾기" 버튼으로 원본 파일 선택 (또는 드래그&드롭)
2. "비교문서 찾기" 버튼으로 수정본 파일 선택
3. "비교 실행" 버튼 클릭
4. 우측 변경사항 목록에서 항목 클릭 시 해당 위치로 이동
5. "엑셀 저장" 버튼으로 결과 내보내기

## 테스트 실행

```bash
cd hwp_diff
python -m pytest tests/ -v
```

또는 개별 테스트:

```bash
python -m unittest tests.test_parsers -v
python -m unittest tests.test_diff_engine -v
python -m unittest tests.test_exporters -v
```

## 샘플 문서 테스트

```bash
python -c "
from app.core.controller import CompareController
ctrl = CompareController()
result = ctrl.run_compare_from_paths(
    'sample_docs/sample_original.txt',
    'sample_docs/sample_modified.txt'
)
print(result.get_summary_stats())
ctrl.export_excel(result, 'output/test_result.xlsx')
"
```

## 디렉터리 구조

```
hwp_diff/
├── main.py                    # 진입점
├── requirements.txt
├── app/
│   ├── models/                # 데이터 모델
│   │   ├── document.py        # DocumentStructure, DocumentBlock
│   │   └── change_record.py   # ChangeRecord, CompareResult
│   ├── utils/
│   │   ├── logger.py          # 로깅 설정
│   │   └── text_utils.py      # 유사도, 정규화, 중요도 분석
│   ├── parsers/               # 형식별 파서
│   │   ├── hwpx_parser.py     # HWPX (zip+XML)
│   │   ├── docx_parser.py     # DOCX (python-docx)
│   │   ├── txt_parser.py      # TXT
│   │   └── hwp_converter.py   # HWP 바이너리 변환
│   ├── normalizers/
│   │   └── document_normalizer.py
│   ├── diff_engine/
│   │   ├── text_differ.py     # 단어/글자 수준 diff
│   │   ├── paragraph_matcher.py # TF-IDF 기반 문단 매칭
│   │   ├── table_matcher.py   # 표 셀 비교
│   │   └── diff_engine.py     # 메인 오케스트레이터
│   ├── exporters/
│   │   └── excel_exporter.py  # openpyxl 기반 Excel 출력
│   ├── core/
│   │   └── controller.py      # 비즈니스 로직 컨트롤러
│   └── ui/
│       ├── main_window.py     # 메인 윈도우
│       ├── file_panel.py      # 파일 선택 패널
│       ├── compare_view.py    # 나란히 보기
│       ├── change_list.py     # 변경사항 목록
│       └── options_dialog.py  # 옵션 대화상자
├── tests/
│   ├── test_parsers.py
│   ├── test_diff_engine.py
│   └── test_exporters.py
└── sample_docs/
    ├── sample_original.txt    # 원본 시험절차서
    └── sample_modified.txt    # 수정된 시험절차서
```

## HWP 파일 지원

HWP(바이너리) 형식은 다음 순서로 처리됩니다:
1. **win32com** (Windows + 한글 프로그램 설치 시): HWPX로 변환 후 파싱
2. **pyhwp**: pyhwp 라이브러리 사용
3. **바이너리 추출**: zlib 압축 해제 및 UTF-16LE 텍스트 추출 (품질 저하)

최상의 결과를 위해 HWPX 형식 사용을 권장합니다.

## 요구사항

- Python 3.9+
- PySide6 >= 6.4.0
- python-docx >= 0.8.11
- openpyxl >= 3.1.0
- scikit-learn >= 1.0.0
- lxml >= 4.9.0
