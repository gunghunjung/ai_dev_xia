"""
HWP (binary format) converter.

전략 우선순위:
1. win32com HWP COM 자동화  → HWPX 저장 → 실패시 TXT 저장
2. 내장 CFB 파서 (표준 라이브러리, olefile 불필요)
3. olefile 기반 CFB 파싱 (설치된 경우)
4. pyhwp CLI
5. zlib 스캔 최후 수단
"""
import os
import re
import struct
import tempfile
import zlib
from pathlib import Path
from typing import List, Optional, Tuple

from app.models.document import DocumentStructure
from app.parsers.base_parser import BaseParser
from app.utils.logger import get_logger

logger = get_logger("parsers.hwp")

# ── HWP5 레코드 태그 상수 ──────────────────────────────────────────
HWPTAG_PARA_TEXT = 67   # 0x43  문단 텍스트 (가장 중요)
HWPTAG_PARA_HEADER = 66 # 0x42  문단 헤더
# CFB 상수
CFB_SIGNATURE   = b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1'
FREESECT        = 0xFFFFFFFF
ENDOFCHAIN      = 0xFFFFFFFE
FATSECT         = 0xFFFFFFFD
DIFSECT         = 0xFFFFFFFC
_SPECIAL_SECTOR = 0xFFFFFFF8   # ≥ 이 값이면 FAT 특수 값


class HWPConverter(BaseParser):
    """HWP 바이너리 파일을 파싱한다."""

    def parse(self, file_path: str) -> DocumentStructure:
        p = self._check_file(file_path)
        doc_id = self._new_doc_id(file_path)
        logger.info("Parsing HWP: %s", file_path)

        # ── 전략 1: 내장 CFB 파서 ──────────────────────────────────────
        # 한글 앱 실행 없이 동작 → 팝업 없음, 빠름, 가장 먼저 시도
        doc = self._try_builtin_cfb(str(p), doc_id)
        if doc and len(doc.blocks) > 0:
            return doc

        # ── 전략 2: olefile 라이브러리 ──────────────────────────────
        doc = self._try_olefile(str(p), doc_id)
        if doc and len(doc.blocks) > 0:
            return doc

        # ── 전략 3: win32com (HWP → TXT 변환) ─────────────────────
        # 한글 앱 필요, 팝업 억제 시도하지만 완전하지 않을 수 있음
        doc = self._try_win32com(str(p), doc_id)
        if doc and len(doc.blocks) > 0:
            return doc

        # ── 전략 4: pyhwp CLI ────────────────────────────────────────
        doc = self._try_pyhwp(str(p), doc_id)
        if doc and len(doc.blocks) > 0:
            return doc

        # ── 전략 5: zlib 스캔 (최후 수단) ───────────────────────────
        return self._fallback_zlib_scan(str(p), doc_id)

    # ══════════════════════════════════════════════════════════════════
    # 전략 1: win32com
    # ══════════════════════════════════════════════════════════════════

    def _try_win32com(self, file_path: str, doc_id: str) -> Optional[DocumentStructure]:
        """
        한글 COM 자동화.
        HWPX → TXT → GetTextFile 순서로 시도한다.
        팝업/권한 대화상자를 자동으로 억제한다.
        """
        try:
            import win32com.client
            import pythoncom
        except ImportError:
            logger.debug("win32com 없음, 건너뜀")
            return None

        hwp = None
        try:
            pythoncom.CoInitialize()
            # DispatchEx: 항상 새 인스턴스 생성 (기존 열린 한글과 충돌 방지)
            hwp = win32com.client.DispatchEx("HWPFrame.HwpObject")

            # ── ① 모든 팝업/대화상자 자동 확인 ──────────────────────────
            # SetMessageBoxMode: 비트 플래그로 자동 응답 지정
            #   0x0001 = 확인(OK)   0x0002 = 예(Yes)   0x0004 = 아니오(No)
            #   0x00F0 = 모든 버튼에 대해 가장 긍정적 응답 자동 선택
            try:
                hwp.SetMessageBoxMode(0x00F0)
            except Exception as e:
                logger.debug("SetMessageBoxMode 미지원 버전: %s", e)

            # ── ② 창 숨기기 (사용자에게 한글 창 노출 방지) ───────────────
            try:
                hwp.XHwpWindows.Item(0).Visible = False
            except Exception:
                pass

            # ── ③ 파일 접근 권한 팝업 자동 허용 모듈 등록 ───────────────
            # RegisterModule 실패 → 파일 열 때마다 권한 팝업 발생
            registered = False
            for module_name in [
                "FilePathCheckerModule",        # 표준
                "FilePathCheckerModule_x64",    # 64비트
                "FilePathCheckerModuleA",       # 일부 구버전
            ]:
                try:
                    hwp.RegisterModule("FilePathCheckDLL", module_name)
                    registered = True
                    logger.debug("보안 모듈 등록 성공: %s", module_name)
                    break
                except Exception:
                    continue
            if not registered:
                logger.warning(
                    "HWP 보안 모듈 등록 실패 → 파일 접근 팝업이 뜰 수 있습니다.\n"
                    "  해결: 한글 프로그램을 관리자 권한으로 한 번 실행하거나,\n"
                    "  SetMessageBoxMode(0x00F0)가 적용되어 자동 확인 시도합니다."
                )

            abs_path = str(Path(file_path).resolve())
            hwp.Open(abs_path, "HWP", "forceopen:true")

            # ── HWPX 저장 시도 ──────────────────────────────────────
            doc = self._win32com_save_as_hwpx(hwp, doc_id, file_path)
            if doc and len(doc.blocks) > 0:
                logger.info("win32com HWPX 변환 성공: %d blocks", len(doc.blocks))
                return doc

            # ── TXT 저장 시도 (구형 한글도 지원) ────────────────────
            doc = self._win32com_save_as_txt(hwp, doc_id, file_path)
            if doc and len(doc.blocks) > 0:
                logger.info("win32com TXT 변환 성공: %d blocks", len(doc.blocks))
                return doc

            # ── GetTextFile 직접 추출 ────────────────────────────────
            doc = self._win32com_get_text(hwp, doc_id, file_path)
            if doc and len(doc.blocks) > 0:
                logger.info("win32com GetTextFile 성공: %d blocks", len(doc.blocks))
                return doc

            logger.warning("win32com: 모든 추출 방법 실패")
            return None

        except Exception as e:
            logger.warning("win32com HWP 실패: %s", e)
            return None
        finally:
            if hwp is not None:
                try:
                    hwp.Quit()
                except Exception:
                    pass
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

    def _win32com_save_as_hwpx(self, hwp, doc_id: str, file_path: str) -> Optional[DocumentStructure]:
        """HWPX로 저장 후 파싱."""
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".hwpx")
        try:
            os.close(tmp_fd)
            hwp.SaveAs(tmp_path, "HWPX", "")   # 3번째 인자 필수
            size = os.path.getsize(tmp_path)
            if size < 200:
                return None
            from app.parsers.hwpx_parser import HWPXParser
            doc = HWPXParser().parse(tmp_path)
            if len(doc.blocks) == 0:
                return None
            doc.doc_id = doc_id
            doc.file_path = file_path
            doc.file_type = "hwp"
            doc.parse_confidence = 0.95
            return doc
        except Exception as e:
            logger.debug("HWPX 저장 실패: %s", e)
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _win32com_save_as_txt(self, hwp, doc_id: str, file_path: str) -> Optional[DocumentStructure]:
        """TXT(UTF-8)로 저장 후 파싱. 구버전 한글에서도 동작."""
        for encoding_arg in ["code:utf-8", "code:utf8", ""]:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".txt")
            try:
                os.close(tmp_fd)
                hwp.SaveAs(tmp_path, "TEXT", encoding_arg)
                size = os.path.getsize(tmp_path)
                if size < 10:
                    continue

                # 인코딩 자동 판별
                raw = open(tmp_path, "rb").read()
                text = self._decode_bytes(raw)
                if len(text.strip()) < 10:
                    continue

                doc = self._text_to_document(text, doc_id, file_path, "hwp")
                if len(doc.blocks) == 0:
                    continue
                doc.parse_confidence = 0.85
                return doc
            except Exception as e:
                logger.debug("TXT SaveAs(%s) 실패: %s", encoding_arg, e)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        return None

    def _win32com_get_text(self, hwp, doc_id: str, file_path: str) -> Optional[DocumentStructure]:
        """GetTextFile API로 직접 텍스트 추출."""
        try:
            text_raw = hwp.GetTextFile("TEXT", "")
            if isinstance(text_raw, (bytes, bytearray)):
                text = self._decode_bytes(bytes(text_raw))
            else:
                text = str(text_raw) if text_raw else ""

            if len(text.strip()) < 10:
                return None
            doc = self._text_to_document(text, doc_id, file_path, "hwp")
            doc.parse_confidence = 0.80
            return doc
        except Exception as e:
            logger.debug("GetTextFile 실패: %s", e)
            return None

    # ══════════════════════════════════════════════════════════════════
    # 전략 2: 내장 CFB 파서 (표준 라이브러리 전용, olefile 불필요)
    # ══════════════════════════════════════════════════════════════════

    def _try_builtin_cfb(self, file_path: str, doc_id: str) -> Optional[DocumentStructure]:
        """
        표준 라이브러리만으로 HWP5 CFB 구조를 직접 파싱.
        CFB (Compound File Binary) = Microsoft OLE2 포맷.
        """
        try:
            data = Path(file_path).read_bytes()
        except Exception as e:
            logger.debug("파일 읽기 실패: %s", e)
            return None

        if data[:8] != CFB_SIGNATURE:
            logger.debug("CFB 시그니처 불일치 (HWP5 아님?)")
            return None

        try:
            reader = _BuiltinCFBReader(data)
            sections = reader.find_sections()
            if not sections:
                logger.debug("내장 CFB: BodyText 섹션 없음")
                return None

            is_compressed = reader.is_bodytext_compressed()
            logger.debug("내장 CFB: 압축=%s, 섹션=%d개", is_compressed, len(sections))

            all_paragraphs: List[str] = []
            for name, start, size in sections:
                try:
                    raw = reader.read_stream(start, size)
                    decompressed = _decompress_hwp_section(raw, is_compressed)
                    paras = _parse_hwp_records(decompressed)
                    all_paragraphs.extend(paras)
                except Exception as sec_e:
                    logger.debug("섹션 %s 파싱 오류: %s", name, sec_e)

            if not all_paragraphs:
                return None

            full_text = "\n".join(p for p in all_paragraphs if p.strip())
            doc = self._text_to_document(full_text, doc_id, file_path, "hwp")
            doc.parse_confidence = 0.75
            doc.page_confidence = 0.2
            doc.table_confidence = 0.3
            logger.info("내장 CFB 파서 성공: %d 문단 → %d blocks",
                        len(all_paragraphs), len(doc.blocks))
            return doc

        except Exception as e:
            logger.warning("내장 CFB 파서 실패: %s", e)
            return None

    # ══════════════════════════════════════════════════════════════════
    # 전략 3: olefile 라이브러리
    # ══════════════════════════════════════════════════════════════════

    def _try_olefile(self, file_path: str, doc_id: str) -> Optional[DocumentStructure]:
        try:
            import olefile
        except ImportError:
            logger.debug("olefile 미설치 (pip install olefile 권장)")
            return None

        try:
            with olefile.OleFileIO(file_path) as ole:
                is_compressed = _olefile_check_compressed(ole)
                section_streams = _olefile_collect_sections(ole)
                if not section_streams:
                    return None

                all_paragraphs: List[str] = []
                for stream_path in section_streams:
                    try:
                        raw = ole.openstream(stream_path).read()
                        decompressed = _decompress_hwp_section(raw, is_compressed)
                        paras = _parse_hwp_records(decompressed)
                        all_paragraphs.extend(paras)
                    except Exception as e:
                        logger.debug("섹션 %s 오류: %s", stream_path, e)

            if not all_paragraphs:
                return None

            full_text = "\n".join(p for p in all_paragraphs if p.strip())
            doc = self._text_to_document(full_text, doc_id, file_path, "hwp")
            doc.parse_confidence = 0.78
            logger.info("olefile HWP 파싱 성공: %d 문단", len(all_paragraphs))
            return doc

        except Exception as e:
            logger.warning("olefile HWP 파싱 실패: %s", e)
            return None

    # ══════════════════════════════════════════════════════════════════
    # 전략 4: pyhwp CLI
    # ══════════════════════════════════════════════════════════════════

    def _try_pyhwp(self, file_path: str, doc_id: str) -> Optional[DocumentStructure]:
        try:
            import subprocess
            result = subprocess.run(
                ["hwp5txt", file_path],
                capture_output=True,
                timeout=60,
            )
            if result.returncode == 0 and result.stdout:
                text = self._decode_bytes(result.stdout)
                if len(text.strip()) > 10:
                    doc = self._text_to_document(text, doc_id, file_path, "hwp")
                    doc.parse_confidence = 0.70
                    logger.info("pyhwp 변환 성공")
                    return doc
        except FileNotFoundError:
            logger.debug("hwp5txt 명령어 없음")
        except Exception as e:
            logger.warning("pyhwp 실패: %s", e)
        return None

    # ══════════════════════════════════════════════════════════════════
    # 전략 5: zlib 스캔 (최후 수단)
    # ══════════════════════════════════════════════════════════════════

    def _fallback_zlib_scan(self, file_path: str, doc_id: str) -> DocumentStructure:
        logger.warning("zlib 스캔 최후 수단 사용 (신뢰도 낮음): %s", file_path)
        doc = DocumentStructure(
            doc_id=doc_id,
            file_path=file_path,
            file_type="hwp",
            parse_confidence=0.15,
            page_confidence=0.05,
            table_confidence=0.05,
        )
        try:
            data = Path(file_path).read_bytes()
            paras = _zlib_scan_for_text(data)
            if paras:
                full_text = "\n".join(paras)
                doc = self._text_to_document(full_text, doc_id, file_path, "hwp")
                doc.parse_confidence = 0.15
        except Exception as e:
            logger.error("zlib 스캔 실패: %s", e)
        return doc

    # ══════════════════════════════════════════════════════════════════
    # 공통 유틸
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _decode_bytes(raw: bytes) -> str:
        """바이트를 한글 인코딩 자동 감지하여 문자열로 변환."""
        # BOM 확인
        if raw[:3] == b'\xef\xbb\xbf':
            return raw[3:].decode("utf-8", errors="replace")
        if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
            return raw.decode("utf-16", errors="replace")
        # UTF-8 시도
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            pass
        # EUC-KR / CP949
        try:
            return raw.decode("cp949")
        except UnicodeDecodeError:
            pass
        return raw.decode("utf-8", errors="replace")

    def _text_to_document(
        self, text: str, doc_id: str, file_path: str, file_type: str
    ) -> DocumentStructure:
        from app.parsers.txt_parser import TXTParser
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".txt")
        try:
            os.close(tmp_fd)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(text)
            doc = TXTParser().parse(tmp_path)
            doc.doc_id = doc_id
            doc.file_path = file_path
            doc.file_type = file_type
            return doc
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════
# 내장 CFB 파서 (표준 라이브러리만 사용)
# ══════════════════════════════════════════════════════════════════════

class _BuiltinCFBReader:
    """
    Microsoft CFB(Compound File Binary) / OLE2 파서.
    HWP5가 이 포맷을 컨테이너로 사용한다.
    표준 라이브러리(struct)만으로 구현.
    """

    def __init__(self, data: bytes):
        self.data = data
        self.sector_size = 1 << struct.unpack_from("<H", data, 30)[0]  # 보통 512
        self.fat: List[int] = []
        self._build_fat()
        self.dir_data: bytes = b""
        self._load_directory()
        self._entries: List[dict] = []
        self._parse_directory()

    def _sector_offset(self, sector: int) -> int:
        return (sector + 1) * self.sector_size

    def _read_sector(self, sector: int) -> bytes:
        offset = self._sector_offset(sector)
        return self.data[offset: offset + self.sector_size]

    def _build_fat(self):
        """DIFAT → FAT 체인 구축."""
        fat_sector_ids: List[int] = []

        # 헤더 내장 DIFAT (109개)
        for i in range(109):
            val = struct.unpack_from("<I", self.data, 76 + i * 4)[0]
            if val < _SPECIAL_SECTOR:
                fat_sector_ids.append(val)

        # 추가 DIFAT 섹터가 있으면 순회
        difat_next = struct.unpack_from("<I", self.data, 68)[0]
        visited_difat: set = set()
        while difat_next < _SPECIAL_SECTOR and difat_next not in visited_difat:
            visited_difat.add(difat_next)
            difat_sector = self._read_sector(difat_next)
            entries_per = (self.sector_size // 4) - 1
            for i in range(entries_per):
                val = struct.unpack_from("<I", difat_sector, i * 4)[0]
                if val < _SPECIAL_SECTOR:
                    fat_sector_ids.append(val)
            difat_next = struct.unpack_from("<I", difat_sector, self.sector_size - 4)[0]

        # FAT 조립
        self.fat = []
        for sid in fat_sector_ids:
            sector = self._read_sector(sid)
            for i in range(0, len(sector), 4):
                if i + 4 <= len(sector):
                    self.fat.append(struct.unpack_from("<I", sector, i)[0])

    def read_stream(self, start_sector: int, size: int = -1) -> bytes:
        """FAT 체인을 따라 스트림 데이터를 읽는다."""
        parts: List[bytes] = []
        sector = start_sector
        visited: set = set()
        while sector < _SPECIAL_SECTOR and sector not in visited:
            visited.add(sector)
            parts.append(self._read_sector(sector))
            if sector >= len(self.fat):
                break
            sector = self.fat[sector]
        raw = b"".join(parts)
        return raw[:size] if size >= 0 else raw

    def _load_directory(self):
        first_dir = struct.unpack_from("<I", self.data, 48)[0]
        self.dir_data = self.read_stream(first_dir)

    def _parse_directory(self):
        """128바이트씩 디렉토리 엔트리를 파싱한다."""
        self._entries = []
        entry_size = 128
        for i in range(len(self.dir_data) // entry_size):
            entry_bytes = self.dir_data[i * entry_size: (i + 1) * entry_size]
            if len(entry_bytes) < 128:
                break
            name_len = struct.unpack_from("<H", entry_bytes, 64)[0]
            if name_len < 2 or name_len > 64:
                self._entries.append(None)
                continue
            name = entry_bytes[:name_len - 2].decode("utf-16-le", errors="ignore")
            obj_type = entry_bytes[66]      # 0=invalid, 1=storage, 2=stream, 5=root
            child    = struct.unpack_from("<I", entry_bytes, 76)[0]
            left     = struct.unpack_from("<I", entry_bytes, 68)[0]
            right    = struct.unpack_from("<I", entry_bytes, 72)[0]
            start    = struct.unpack_from("<I", entry_bytes, 116)[0]
            size     = struct.unpack_from("<I", entry_bytes, 120)[0]
            self._entries.append({
                "name": name, "type": obj_type,
                "child": child, "left": left, "right": right,
                "start": start, "size": size,
            })

    def _find_entry_by_name(self, name_lower: str) -> Optional[dict]:
        for e in self._entries:
            if e and e["name"].lower() == name_lower:
                return e
        return None

    def _children_of(self, storage_entry: dict) -> List[dict]:
        """storage 엔트리의 자식 목록을 반환 (red-black tree 순회)."""
        results: List[dict] = []
        child_idx = storage_entry.get("child", FREESECT)
        visited: set = set()
        queue = [child_idx]
        while queue:
            idx = queue.pop(0)
            if idx == FREESECT or idx in visited or idx >= len(self._entries):
                continue
            visited.add(idx)
            entry = self._entries[idx]
            if entry is None:
                continue
            results.append(entry)
            if entry["left"] != FREESECT:
                queue.append(entry["left"])
            if entry["right"] != FREESECT:
                queue.append(entry["right"])
        return results

    def is_bodytext_compressed(self) -> bool:
        """FileHeader 스트림에서 압축 플래그를 읽는다."""
        fh = self._find_entry_by_name("fileheader")
        if fh and fh["start"] < _SPECIAL_SECTOR:
            try:
                header_data = self.read_stream(fh["start"], fh["size"])
                if len(header_data) >= 40:
                    flags = struct.unpack_from("<I", header_data, 36)[0]
                    return bool(flags & 0x1)
            except Exception:
                pass
        return True  # 기본값: 압축됨

    def find_sections(self) -> List[Tuple[str, int, int]]:
        """
        BodyText 아래의 Section* 스트림 목록 반환.
        반환: [(이름, start_sector, size), ...]
        """
        bt = self._find_entry_by_name("bodytext")
        if bt is None:
            # 일부 파일은 스토리지 이름이 다를 수 있음 - 전체 스캔
            return self._scan_all_sections()

        sections: List[Tuple[str, int, int]] = []
        for child in self._children_of(bt):
            if re.match(r"section\d*$", child["name"].lower()) and child["start"] < _SPECIAL_SECTOR:
                sections.append((child["name"], child["start"], child["size"]))

        def _sec_num(t: tuple) -> int:
            m = re.search(r"\d+", t[0])
            return int(m.group()) if m else 0
        sections.sort(key=_sec_num)
        return sections

    def _scan_all_sections(self) -> List[Tuple[str, int, int]]:
        """전체 엔트리에서 Section* 스트림을 이름으로 검색 (폴백)."""
        results = []
        for e in self._entries:
            if e and e["type"] == 2 and re.match(r"section\d*$", e["name"].lower()):
                if e["start"] < _SPECIAL_SECTOR:
                    results.append((e["name"], e["start"], e["size"]))

        def _sec_num(t: tuple) -> int:
            m = re.search(r"\d+", t[0])
            return int(m.group()) if m else 0
        results.sort(key=_sec_num)
        return results


# ══════════════════════════════════════════════════════════════════════
# HWP5 레코드 파싱 (공통 함수)
# ══════════════════════════════════════════════════════════════════════

def _decompress_hwp_section(data: bytes, is_compressed: bool) -> bytes:
    """HWP5 섹션 스트림 압축 해제. raw deflate(wbits=-15) 우선 시도."""
    if not is_compressed:
        return data
    for wbits, offset in [(-15, 0), (15, 0), (-15, 4)]:
        try:
            return zlib.decompress(data[offset:], wbits)
        except zlib.error:
            continue
    raise ValueError("HWP 섹션 압축 해제 실패")


def _parse_hwp_records(data: bytes) -> List[str]:
    """
    HWP5 레코드 스트림 파싱.
    레코드 헤더 (4바이트 little-endian):
      bits  0-9  : tag_id
      bits 10-11 : level
      bits 12-31 : size  (0xFFFFF이면 다음 4바이트가 실제 크기)
    HWPTAG_PARA_TEXT(67) 레코드에서 UTF-16LE 텍스트 추출.
    """
    paragraphs: List[str] = []
    pos = 0
    length = len(data)

    while pos + 4 <= length:
        header = struct.unpack_from("<I", data, pos)[0]
        tag_id = header & 0x3FF
        size   = (header >> 12) & 0xFFFFF
        pos += 4

        if size == 0xFFFFF:        # 확장 크기
            if pos + 4 > length:
                break
            size = struct.unpack_from("<I", data, pos)[0]
            pos += 4

        if size < 0 or pos + size > length:
            break

        if tag_id == HWPTAG_PARA_TEXT and size >= 2:
            text = _decode_para_text(data[pos: pos + size])
            if text:
                paragraphs.append(text)

        pos += size

    return paragraphs


def _decode_para_text(record_data: bytes) -> str:
    """PARA_TEXT 레코드 → 정제된 한글 문자열."""
    try:
        raw = record_data.decode("utf-16-le", errors="ignore")
    except Exception:
        return ""

    cleaned = []
    for ch in raw:
        cp = ord(ch)
        if cp == 0x0D:
            continue
        elif cp == 0x0A or cp == 0x09:
            cleaned.append(ch)
        elif cp < 0x20:          # 기타 제어 문자 → 공백
            cleaned.append(" ")
        else:
            cleaned.append(ch)

    result = "".join(cleaned).strip()
    # 의미 있는 출력 문자가 있어야 반환
    if sum(1 for c in result if c.isprintable() and not c.isspace()) < 1:
        return ""
    return result


# ══════════════════════════════════════════════════════════════════════
# olefile 헬퍼 함수
# ══════════════════════════════════════════════════════════════════════

def _olefile_check_compressed(ole) -> bool:
    try:
        hd = ole.openstream("FileHeader").read()
        if len(hd) >= 40:
            return bool(struct.unpack_from("<I", hd, 36)[0] & 0x1)
    except Exception:
        pass
    return True


def _olefile_collect_sections(ole) -> List[str]:
    result = []
    try:
        for entry in ole.listdir():
            if (len(entry) >= 2
                    and entry[0].lower() == "bodytext"
                    and re.match(r"section\d*$", entry[1].lower())):
                result.append("/".join(entry))
    except Exception:
        pass
    result.sort(key=lambda s: int(re.search(r"\d+", s).group() if re.search(r"\d+", s) else "0"))
    return result


# ══════════════════════════════════════════════════════════════════════
# zlib 스캔 (최후 수단)
# ══════════════════════════════════════════════════════════════════════

def _zlib_scan_for_text(raw: bytes) -> List[str]:
    """파일 전체를 스캔해 zlib 블록을 찾고 HWP 레코드 파싱 시도."""
    results: List[str] = []
    magics = [b"\x78\x9c", b"\x78\x01", b"\x78\xda", b"\x78\x5e"]
    found_at: set = set()
    pos = 0

    while pos < len(raw) - 2:
        for magic in magics:
            if raw[pos: pos + 2] == magic and pos not in found_at:
                for chunk_end in range(min(len(raw), pos + 1048576), pos + 64, -1024):
                    try:
                        decompressed = zlib.decompress(raw[pos: chunk_end])
                        paras = _parse_hwp_records(decompressed)
                        if paras:
                            results.extend(paras)
                        else:
                            # 레코드 파싱 실패 시 UTF-16LE 직접 디코딩
                            candidate = decompressed.decode("utf-16-le", errors="ignore")
                            parts = re.findall(
                                r"[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F\w\s\.,\-\(\)·:;/]{4,}",
                                candidate,
                            )
                            if parts:
                                results.extend(parts)
                        found_at.add(pos)
                        break
                    except (zlib.error, Exception):
                        continue
        pos += 1

    return results
