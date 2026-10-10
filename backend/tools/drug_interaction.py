"""drug_interaction — MOCK interaction lookup.

This is a deliberately small, offline stand-in for a real interaction service —
enough to demonstrate that the agent *checks* for and *escalates* safety concerns
rather than ignoring them. It is NOT a clinical knowledge base.

Safety contract (matches the real-tool requirement): on failure we never assume
"no interaction" — the registry's retry/report wrapper turns a failure into an
explicit error the agent must observe. A clean "none found" result is always
qualified as a mock-DB limitation, not a clinical all-clear.
"""
import re

# Known pairwise interactions, keyed by a frozenset of normalized drug names.
_INTERACTIONS: dict[frozenset, tuple[str, str]] = {
    frozenset({"warfarin", "aspirin"}): ("major", "Increased bleeding risk."),
    frozenset({"warfarin", "ibuprofen"}): ("major", "Increased GI bleeding risk."),
    frozenset({"warfarin", "diclofenac"}): ("major", "NSAID + warfarin: increased GI bleeding risk."),
    frozenset({"levothyroxine", "pantoprazole"}): (
        "moderate", "PPIs can reduce levothyroxine absorption; separate dosing/monitor TSH."),
    frozenset({"levothyroxine", "calcium"}): (
        "moderate", "Calcium reduces levothyroxine absorption; separate by >=4h."),
    frozenset({"ciprofloxacin", "ondansetron"}): (
        "moderate", "Additive QT prolongation risk."),
    frozenset({"ondansetron", "metoclopramide"}): (
        "moderate", "Additive QT/extrapyramidal risk."),
    frozenset({"meropenem", "valproate"}): (
        "major", "Meropenem markedly lowers valproate levels; seizure risk."),
    frozenset({"metformin", "contrast"}): (
        "major", "Risk of lactate acidosis around iodinated contrast; hold metformin."),
    frozenset({"spironolactone", "potassium"}): (
        "major", "Hyperkalemia risk."),
    frozenset({"diclofenac", "insulin"}): (
        "minor", "NSAIDs may potentiate hypoglycemia; monitor glucose."),
}

# Map common brand/synonyms (incl. the Indian brand names in THIS corpus) ->
# the normalized generic used as keys above. Without this, brand-named chart
# entries like "INJ MEROMAC" never resolve and the check is meaningless.
_SYNONYMS = {
    # PPIs
    "pan": "pantoprazole", "pantodac": "pantoprazole", "pantop": "pantoprazole",
    # thyroid
    "thyronorm": "levothyroxine", "eltroxin": "levothyroxine", "thyroxine": "levothyroxine",
    # antiemetics
    "emeset": "ondansetron", "ondem": "ondansetron", "zofran": "ondansetron", "emset": "ondansetron",
    "perinorm": "metoclopramide", "reglan": "metoclopramide",
    # antibiotics
    "cifran": "ciprofloxacin", "cipro": "ciprofloxacin",
    "meromac": "meropenem", "meronem": "meropenem", "merocrit": "meropenem",
    # antiplatelet/anticoagulant
    "ecosprin": "aspirin", "asa": "aspirin",
    # analgesics / NSAIDs
    "dolo": "paracetamol", "sumol": "paracetamol", "sumo": "paracetamol",
    "calpol": "paracetamol", "pcm": "paracetamol",
    "painodac": "diclofenac", "voveran": "diclofenac", "dynapar": "diclofenac",
    # insulins
    "lantus": "insulin", "actrapid": "insulin", "hactrapid": "insulin",
    "mixtard": "insulin", "huminsulin": "insulin", "humalog": "insulin",
}

# Dosage-form / route / unit tokens to drop so we key on the drug, not the form.
_FORM_ROUTE_UNIT = {
    "inj", "injection", "tab", "tabs", "tablet", "cap", "caps", "capsule",
    "syp", "syr", "syrup", "ins", "insulin", "sol", "soln", "susp", "oint", "neb",
    "iv", "im", "sc", "po", "sl", "pr", "s", "h",
    "mg", "g", "gm", "kg", "ml", "l", "mcg", "ug", "u", "iu", "unit", "units",
    "amp", "ampoule", "cc", "od", "bd", "tds", "qid", "sos", "stat", "hs", "plus",
}


def _normalize(name: str) -> str:
    """Resolve a chart entry to a generic name: strip form/dose tokens, map synonyms.

    'INJ MEROMAC 1 g' -> 'meropenem';  'INS LANTUS 10 U' -> 'insulin';
    'TAB DOLO 650' -> 'paracetamol'.  'insulin' is special-cased above because
    the form word IS the generic class.
    """
    toks = re.findall(r"[a-z0-9]+", (name or "").lower())
    for t in toks:
        if any(c.isdigit() for c in t):
            continue
        if t in _SYNONYMS:            # brand resolves directly (even if also a form word)
            return _SYNONYMS[t]
        if t in _FORM_ROUTE_UNIT or len(t) <= 1:
            continue
        return _SYNONYMS.get(t, t)    # first real token, synonym-mapped if known
    return ""


def drug_interaction(meds) -> dict:
    """Check a list of drug names/strings for known pairwise interactions (mock)."""
    if isinstance(meds, str):
        meds = [m for m in meds.replace(";", ",").split(",") if m.strip()]
    names = sorted({_normalize(m) for m in (meds or []) if _normalize(m)})

    found = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sev_msg = _INTERACTIONS.get(frozenset({names[i], names[j]}))
            if sev_msg:
                found.append({
                    "pair": [names[i], names[j]],
                    "severity": sev_msg[0],
                    "description": sev_msg[1],
                })
    return {
        "ok": True,
        "checked": names,
        "interactions": found,
        "source": "MOCK interaction DB — not for clinical use",
        "note": ("No interaction in the mock DB does NOT mean none exist."
                 if not found else "Surface and escalate to the reviewer."),
    }
