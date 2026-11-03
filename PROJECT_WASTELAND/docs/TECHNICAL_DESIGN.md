# PROJECT WASTELAND — 기술 설계서 (PHASE 0)

## 1. 기획 분석 요약

| 핵심 축 | 설계 결론 |
|---|---|
| 무지원 생존 | 시작 인벤토리 = 낡은 옷 1벌. 모든 자원은 지표 채집 · 수색 · 굴착 · 사냥에서만 획득. NPC 없음. |
| 발굴 (차별화 #1) | 셀 단위 지층(지표/얕은/중간/깊은) 데이터. 매장물은 **지역 × 지층 × 셀 해시**로 결정론적 배치 → 같은 셀은 항상 같은 결과, 랜덤 버튼 보상 아님. 매장 캐시는 지표에 녹슨 흔적(시각 단서)을 남김. |
| 희소성 (차별화 #2) | 탄약은 깊은 지층/군사지역/약탈자에게서만 극소량. 깨끗한 물 ≠ 오염된 물. 도구 내구도 소모 + 수리 비용. |
| 액션 | 이동/조준 분리, 근접 판정(사거리·호), 투사체, 히트스톱·넉백·카메라 셰이크·파티클·진동 — 강도는 설정에서 조절. |
| 소음 | 걷기/달리기/굴착/근접/총성 → `NoiseSystem` 이 반경 내 적에게 전파 → Investigate 상태. |

핵심 루프: 탐색 → 채집 → 발굴 → 제작 → 생존 → 전투 → 거점 → 위험지역.

## 2. 기술 스택 검토

- **pygame-ce 2.5** : SDL2 기반, `Joystick.rumble`, 핫플러그 이벤트(`JOYDEVICEADDED/REMOVED`) 지원. 
- **SQLite** : 단일 파일 세이브, 트랜잭션으로 원자적 저장(+ `.bak` 백업 회전), `meta.save_version` 으로 마이그레이션.
- **JSON** : 아이템·무기·적·지형·레시피·오브젝트·전리품 테이블 전부 `data/` 에 정의.
- **Pillow** : 에셋 파이프라인(아이콘 아틀라스 등) 용도로 확보. 런타임 필수 의존성 아님.
- **PyInstaller** : `build.spec` 으로 `data/`, `assets/` 포함 onedir 빌드.
- **Pytest** : 시스템 단위 테스트 + 헤드리스(SDL dummy) 통합 스모크 테스트.

## 3. 아키텍처

```
main.py → game.app.App (씬 스택)
            ├─ MenuScene / PauseScene / SettingsScene / DeathScene
            └─ GameplayScene
                  └─ GameSession  (게임 상태 + 시스템 오케스트레이션, 렌더 비의존)
                       ├─ WorldManager (청크/셀/오브젝트)   ← IWorldService
                       ├─ Player, Enemies, Wildlife, Projectiles(풀)
                       ├─ Systems: combat, digging, survival, inventory, crafting,
                       │           world(시간), weather, sound(소음), effects, spawn
                       └─ Services(Local*) ← interfaces/*  (향후 Network* 로 교체)
engine/: GameLoop(고정 60Hz 업데이트 + 가변 렌더), Camera(아이소 투영/셰이크),
         InputManager(논리 액션), AssetManager(캐시+플레이스홀더), AudioManager(합성 SFX), Renderer
```

### 게임 루프
고정 `dt = 1/60` 누산기 방식. 프레임 처리 순서: **입력 → 업데이트(n회) → 렌더**.
업데이트 내부 순서: 플레이어 입력 해석 → 이동/충돌 → AI(거리별 갱신 주기) → 전투 판정 → 생존 → 월드/날씨 → 이펙트.
게임 속도는 프레임레이트와 무관. 히트스톱은 업데이트 dt 를 0 으로 만드는 방식(이펙트/카메라는 계속 동작).

### 좌표계
월드 = 타일 단위 실수 좌표 `(x, y)`. 아이소 투영 `sx = (x-y)·TW/2`, `sy = (x+y)·TH/2 - z` (TW=64, TH=32).
모든 게임 로직(충돌, 판정, AI)은 월드 공간. 렌더만 투영. 마우스 → 월드는 역투영.

### 입력 (논리 액션)
`data/input_profiles.json` 에 패드 물리 레이아웃(XInput 기준 버튼/축 번호), 설정에 **액션 → 바인딩** 저장.
게임 코드는 `input.pressed("attack")` 처럼 액션만 사용. 마지막 입력 장치를 추적해 UI 버튼 안내 자동 전환.
스틱: 원형 데드존 + 재정규화 + 저역 필터. 오른쪽 스틱은 마지막 조준 방향 유지, 선택적 조준 보정(콘 내부 최근접 적 쪽으로 감도만큼 보간).

### 월드 / 굴착 데이터
- 160×160 셀, 16×16 청크, 청크는 접근 시 결정론적으로 생성(시드).
- 셀: `terrain(u8)`, `depth(u8 0..3)`, `work(f32 현재 지층 남은 작업량)`, `flags(캐시/수색됨)`.
- 저장은 **변경된 셀만** (`world_cells` 테이블) + 오브젝트 상태 + 구조물.
- 굴착 작업량 = `layer_work[depth] × terrain.hardness`, 진행 속도 = `tool.dig_power`, 도구별 최대 깊이 제한.

### 서버 권한 고려
클라이언트(입력/표현)와 상태 변경을 분리: 아이템 획득·굴착 결과·피해 확정은 `LocalWorldService`, `LocalInventoryService`, `LocalCombatService` 가 수행.
향후 동일 인터페이스의 네트워크 구현체로 교체 시 서버가 결과를 확정하고 클라이언트는 결과만 반영.

### 렌더링
- 화면 내 타일만 블릿(가시 범위 계산), 타일 텍스처는 지형별 변형 3종 캐시.
- 오브젝트/엔티티는 `(x+y)` 깊이 정렬, 화면 밖 컬링.
- 낮/밤: 광원맵(ambient + 가산 원형 그라데이션) 을 MULT 블렌딩.
- 모든 스프라이트는 `assets/` 에 파일이 있으면 그것을, 없으면 절차적 플레이스홀더를 사용 (교체 가능 구조).

### 저장
`saves/slot_N.db` : `meta, player, inventory, equipment, world_cells, world_objects, structures`.
`saves/settings.db` : 설정/키 바인딩. 저장은 임시 파일에 쓰고 원자적 교체, 직전 파일은 `.bak`.

## 4. 성능 정책
타일/스프라이트 캐시, 화면 밖 컬링, 공간 해시 충돌, 원거리 AI 5Hz 갱신, 투사체·파티클 풀링, 저장 시 단일 트랜잭션.
`F3` 디버그 오버레이: FPS, 업데이트 ms, 렌더 ms, 엔티티/오브젝트/파티클 수, 로드된 청크 수.

## 5. 단계별 결과
각 PHASE 의 실행/테스트/수정 결과는 `docs/DEVELOPMENT_LOG.md` 에 기록한다.
