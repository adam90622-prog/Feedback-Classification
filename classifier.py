"""
피드백 분류 엔진 v2
1차: 별점 ≤ 3 → 불만 강제 분류
2차: 리스크 유형별 기본 점수
3차: 2차 피해 추가 점수 + 별점 보정
"""

import json
import re
import requests


# ── 리스크 설정 ───────────────────────────────────────────────────

RISK_CONFIG = {
    "법적·안전":  {"score": 10, "label": "⚠️ 법적·안전",  "desc": "식품안전·알레르기·위생·신체 위험"},
    "대량발생":   {"score": 8,  "label": "📢 대량발생",    "desc": "시스템 오류 등 다수 동시 피해 가능"},
    "금전피해":   {"score": 7,  "label": "💳 금전피해",    "desc": "환불·결제·포인트 등 금전 손실"},
    "평판·법적":  {"score": 6,  "label": "📣 평판·법적",   "desc": "고소·SNS 확산·민원 등 외부 파급"},
    "반복운영":   {"score": 5,  "label": "🔁 반복운영",    "desc": "동일 문제 반복·운영 체계 부재"},
    "서비스품질": {"score": 3,  "label": "⏱ 서비스품질",  "desc": "응대·대기·시설 등 품질 불편"},
    "일반불편":   {"score": 1,  "label": "💬 일반불편",    "desc": "경미한 개인 불편"},
}

SECONDARY_CONFIG = {
    "신체피해": {"score": 5, "label": "🚑 신체피해"},
    "금전손실": {"score": 3, "label": "💸 금전손실"},
    "법적조치": {"score": 2, "label": "⚖️ 법적조치"},
}

STAR_BONUS = {1: 3, 2: 2, 3: 1}  # 별점 낮을수록 공개 노출 영향 크므로 가산점

def _compute_score(risk_type: str, secondary: list, star=None) -> int:
    base   = RISK_CONFIG.get(risk_type, RISK_CONFIG["일반불편"])["score"]
    bonus  = sum(SECONDARY_CONFIG[s]["score"] for s in secondary if s in SECONDARY_CONFIG)
    try:
        star_b = STAR_BONUS.get(int(float(star)), 0) if star not in (None, "") else 0
    except (ValueError, TypeError):
        star_b = 0
    return base + bonus + star_b


# ── LLM 분류 ─────────────────────────────────────────────────────

SYSTEM_PROMPT = "당신은 카페 고객 피드백을 분석하는 전문가입니다. 지시한 JSON 형식으로만 응답하세요."

def _build_user_prompt(feedbacks: list[dict]) -> str:
    lines = []
    for fb in feedbacks:
        line = f'[{fb["id"]}] {fb["내용"]}'
        if fb.get("별점") not in (None, ""):
            line += f' (별점: {fb["별점"]})'
        lines.append(line)

    risk_desc  = "\n".join(f"  - {k}: {v['desc']}" for k, v in RISK_CONFIG.items())
    sec_desc   = "\n".join(f"  - {k}: {v['label']}" for k, v in SECONDARY_CONFIG.items())

    return f"""다음 고객 피드백 {len(feedbacks)}건을 분석하세요.

{chr(10).join(lines)}

분류 규칙:
1. 내용이 부정적이거나 문제를 호소하면 별점 유무와 관계없이 "불만"으로 분류. 별점은 점수 보조용으로만 참고.
2. 불만이면 아래 리스크 유형 중 가장 심각한 하나를 선택.
3. 실제로 피해가 발생한 경우에만 2차피해유형에 포함 (발생 가능성이 아닌 실제 발생).

유형: 불만 | 칭찬 | 문의 | 요청

리스크유형 (불만일 때만):
{risk_desc}

2차피해유형 (실제 피해 발생 시, 복수 가능):
{sec_desc}

JSON 배열로만 응답 (다른 설명 없이):
[
  {{"id": 1, "유형": "불만", "리스크유형": "법적·안전", "2차피해유형": ["신체피해"], "긴급이유": "알레르기 실제 발생, 법적 책임 위험"}},
  {{"id": 2, "유형": "칭찬", "리스크유형": "", "2차피해유형": [], "긴급이유": ""}}
]"""


def classify_with_llm(feedbacks: list[dict], api_key: str) -> dict:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-haiku-4-5",
        "max_tokens": 4096,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": _build_user_prompt(feedbacks)}],
    }
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers, json=payload, timeout=30,
    )
    resp.raise_for_status()
    text  = resp.json()["content"][0]["text"].strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"LLM 응답 JSON 파싱 실패: {text[:200]}")
    return {str(r["id"]): r for r in json.loads(match.group())}


# ── 룰기반 분류 ───────────────────────────────────────────────────

RISK_PATTERNS = [
    ("법적·안전",  r"알레르기|식중독|이물질|벌레|머리카락|위생|상한|곰팡이|화상|화재|부상|다쳤|다칠"),
    ("대량발생",   r"시스템.*오류|앱.*오류|결제.*시스템|포인트.*사라|서버|모든 고객|다들|여러 명"),
    ("금전피해",   r"환불|결제.*안|포인트.*안|중복.*결제|이중.*결제|돈.*날|금전"),
    ("평판·법적",  r"고소|신고|언론|뉴스|sns|인스타|유튜브|법적|변호사|소비자원"),
    ("반복운영",   r"두 번|두번|반복|또|계속|매번|항상"),
    ("서비스품질", r"느려|늦|직원|응대|태도|불친절|오래.*기다|대기|오류|에러"),
]

SECONDARY_PATTERNS = [
    ("신체피해", r"알레르기.*났|부상.*당|다쳤|화상.*입|탈났|아팠|응급|병원"),
    ("금전손실", r"환불.*못|돈.*날|손해.*봤|중복.*결제.*됐"),
    ("법적조치", r"고소|신고|법적.*조치|변호사|소비자원|언론.*제보"),
]

NEGATIVE_PATTERNS = (
    r"오류|에러|실패|안 돼|안돼|안됩|안되|불편|불만|느려|늦|오래|좁|끊겨|식었"
    r"|잘못|못 받|틀렸|피해|환불|결제.*안|결제.*실패|포인트.*안|별로|최악|실망"
    r"|화가|짜증|문제|고장|먹통|작동.*안|되질|안 되|불량|항의|다시.*안|또.*안"
)

def _rule_risk_type(text: str) -> str:
    for risk_type, pattern in RISK_PATTERNS:
        if re.search(pattern, text):
            return risk_type
    return "일반불편"

def _rule_secondary(text: str) -> list:
    result = []
    for damage_type, pattern in SECONDARY_PATTERNS:
        if re.search(pattern, text):
            result.append(damage_type)
    return result

def _is_negative(text: str, star) -> bool:
    # 내용 기준 우선 판단
    if re.search(NEGATIVE_PATTERNS, text):
        return True
    # 내용으로 판단 안 되면 별점 보조 활용
    try:
        if star not in (None, "") and int(float(star)) <= 3:
            return True
    except (ValueError, TypeError):
        pass
    return False

def classify_with_rules(feedbacks: list[dict]) -> dict:
    results = {}
    for fb in feedbacks:
        text  = str(fb.get("내용", ""))
        star  = fb.get("별점")
        if _is_negative(text, star):
            label      = "불만"
            risk_type  = _rule_risk_type(text)
            secondary  = _rule_secondary(text)
            reason_map = {k: v["desc"] for k, v in RISK_CONFIG.items()}
            reason     = reason_map.get(risk_type, "")
            if secondary:
                reason += " / 실제 피해: " + ", ".join(SECONDARY_CONFIG[s]["label"] for s in secondary)
        else:
            for lbl, pats in [
                ("칭찬", r"좋아|맛있|감사|친절|깨끗|추천|최고|완벽|단골"),
                ("요청", r"있으면 좋|됐으면|해주세요|추가|옵션|제안"),
                ("문의", r"가능한가요|되나요|있나요|어떻게|언제|몇 시|영업|예약"),
            ]:
                if re.search(pats, text):
                    label = lbl
                    break
            else:
                label = "문의"
            risk_type, secondary, reason = "", [], ""

        score = _compute_score(risk_type, secondary, star) if label == "불만" else 0
        results[str(fb["id"])] = {
            "id": fb["id"],
            "유형": label,
            "리스크유형": risk_type,
            "2차피해유형": secondary,
            "긴급도": score,
            "긴급이유": reason,
        }
    return results


# ── 통합 진입점 ───────────────────────────────────────────────────

def classify(feedbacks: list[dict], api_key: str = "") -> tuple[list[dict], str]:
    if api_key:
        try:
            label_map = classify_with_llm(feedbacks, api_key)
            mode = "LLM (claude-haiku-4-5)"
        except Exception as e:
            print(f"[LLM 오류] {e} → 룰기반으로 폴백")
            label_map = classify_with_rules(feedbacks)
            mode = "룰기반 (폴백)"
    else:
        label_map = classify_with_rules(feedbacks)
        mode = "룰기반"

    enriched = []
    for fb in feedbacks:
        raw = label_map.get(str(fb["id"]), {
            "유형": "문의", "리스크유형": "", "2차피해유형": [], "긴급이유": "",
        })
        # LLM 결과면 점수를 코드에서 계산
        if "긴급도" not in raw:
            star = fb.get("별점")
            # 별점 ≤3이면 강제 불만
            try:
                if star not in (None, "") and int(float(star)) <= 3:
                    raw["유형"] = "불만"
            except (ValueError, TypeError):
                pass
            if raw["유형"] == "불만":
                raw["긴급도"] = _compute_score(
                    raw.get("리스크유형", "일반불편"),
                    raw.get("2차피해유형", []),
                    fb.get("별점"),
                )
            else:
                raw["긴급도"] = 0

        enriched.append({**fb, **raw})

    return enriched, mode


def get_type_counts(enriched: list[dict]) -> dict:
    counts = {"불만": 0, "칭찬": 0, "문의": 0, "요청": 0}
    for r in enriched:
        t = r.get("유형", "문의")
        counts[t] = counts.get(t, 0) + 1
    return counts
