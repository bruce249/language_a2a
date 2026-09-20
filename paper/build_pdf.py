"""Render the paper markdown to PDF with figures in place."""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
MD = ROOT / "language-as-an-oversight-surface.md"
PDF = ROOT / "language-as-an-oversight-surface.pdf"
FIG = ROOT / "figures"

MATH = [
    (r"\$\\mathrm\{LPI\}_L\$", "LPI<sub>L</sub>"),
    (r"\$\\mathrm\{CER\}_L\$", "CER<sub>L</sub>"),
    (r"\$\\mathrm\{CCA\}_L\$", "CCA<sub>L</sub>"),
    (r"\$\\mathrm\{OG\}_L\$", "OG<sub>L</sub>"),
    (r"\$\\rho_\{\\mathrm\{raw\}\}\$", "rho<sub>raw</sub>"),
    (r"\$\\rho_\{\\mathrm\{residual\}\}\$", "rho<sub>residual</sub>"),
    (r"\$\\rho_\{\\mathrm\{res\}\}\$", "rho<sub>res</sub>"),
    (r"\$\\rho\$", "rho"),
    (r"\$A_\{\\mathrm\{en\}\}\$", "A<sub>en</sub>"),
    (r"\$C_\{\\mathrm\{en\}\}\$", "C<sub>en</sub>"),
    (r"\$U_\{\\mathrm\{en\}\}\$", "U<sub>en</sub>"),
    (r"\$A_L\$", "A<sub>L</sub>"),
    (r"\$C_L\$", "C<sub>L</sub>"),
    (r"\$S_L\$", "S<sub>L</sub>"),
    (r"\$D_L\$", "D<sub>L</sub>"),
    (r"\$U_L\$", "U<sub>L</sub>"),
    (r"\$\\beta\$", "beta"),
    (r"\$\\times\$", "x"),
    (r"\$\\approx\$", "~"),
    (r"\$\\ge\$", "&gt;="),
    (r"\$\\notin\$", "not in"),
    (r"\$\\in\$", "in"),
    (r"\s*\\cdot\s*", " · "),
    (r"\s*\\mid\s*", " | "),
    (r"\$n=30\$", "<i>n</i>=30"),
    (r"\$n=50\$", "<i>n</i>=50"),
    (r"\$n=5\$", "<i>n</i>=5"),
    (r"\$n=100\$", "<i>n</i>=100"),
    (r"\$n=200\$", "<i>n</i>=200"),
    (r"\$L\$", "<i>L</i>"),
    (r"\$t\$", "<i>t</i>"),
    (r"\$P\(\\mathrm\{en\}\)=0\.685\$", "P(en)=0.685"),
]


def _math_to_rl(inner: str) -> str:
    wrapped = f"${inner}$"
    for pat, repl in MATH:
        wrapped = re.sub(pat, repl, wrapped)
    if wrapped.startswith("$") and wrapped.endswith("$"):
        inner = wrapped[1:-1]
        inner = inner.replace("\\mathrm{", "").replace("\\text{", "")
        inner = inner.replace("\\rho", "rho").replace("\\beta", "beta")
        inner = inner.replace("\\times", "x").replace("\\approx", "~")
        inner = inner.replace("\\ge", ">=").replace("\\in", " in ")
        inner = inner.replace("{", "").replace("}", "").replace("\\", "")
        inner = re.sub(r"_([A-Za-z0-9]+)", r"<sub>\1</sub>", inner)
        return f"<i>{inner}</i>"
    return wrapped


def rl(text: str) -> str:
    text = text.replace("—", " - ").replace("–", "-").replace("→", "->")
    text = text.replace("×", "x").replace("≈", "~").replace("∈", " in ")
    held: list[str] = []

    def stash_math(match: re.Match) -> str:
        held.append(_math_to_rl(match.group(1)))
        return f"@@MATH{len(held) - 1}@@"

    text = re.sub(r"\$([^$]+)\$", stash_math, text)
    text = re.sub(r"\$(\d+(?:\.\d+)?)", r"USD \1", text)
    codes: list[str] = []

    def stash_code(match: re.Match) -> str:
        codes.append(match.group(1))
        return f"@@CODE{len(codes) - 1}@@"

    text = re.sub(r"`([^`]+)`", stash_code, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = text.replace("&", "&amp;")
    protect = ("b", "i", "sub", "super")
    for tag in protect:
        text = text.replace(f"<{tag}>", f"\x00{tag}\x00").replace(f"</{tag}>", f"\x00/{tag}\x00")
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    for tag in protect:
        text = text.replace(f"\x00{tag}\x00", f"<{tag}>").replace(f"\x00/{tag}\x00", f"</{tag}>")
    for idx, code in enumerate(codes):
        safe = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"@@CODE{idx}@@", f"<font face='Courier' size='8'>{safe}</font>")
    for idx, math in enumerate(held):
        text = text.replace(f"@@MATH{idx}@@", math)
    return text


def styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PTitle",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=16,
            leading=20,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "PSub",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#333333"),
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "PH1",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=13,
            leading=16,
            spaceBefore=16,
            spaceAfter=8,
            textColor=colors.HexColor("#1a1a1a"),
        ),
        "h2": ParagraphStyle(
            "PH2",
            parent=base["Heading2"],
            fontName="Times-Bold",
            fontSize=11.5,
            leading=14,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "PBody",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10,
            leading=13,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        ),
        "bullet": ParagraphStyle(
            "PBullet",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10,
            leading=13,
            leftIndent=16,
            spaceAfter=3,
        ),
        "quote": ParagraphStyle(
            "PQuote",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=9.5,
            leading=12.5,
            leftIndent=22,
            rightIndent=12,
            textColor=colors.HexColor("#222222"),
            spaceBefore=4,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "PCap",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=12,
        ),
        "th": ParagraphStyle(
            "PTH",
            parent=base["Normal"],
            fontName="Times-Bold",
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
        ),
        "td": ParagraphStyle(
            "PTD",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=8,
            leading=10,
            alignment=TA_LEFT,
        ),
        "footer": ParagraphStyle(
            "PFoot",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
        ),
    }


CAPTIONS = {
    "fig1_detection.png": (
        "Figure 1. Share of seeded leaks flagged, by transcript language, with 95% Wilson "
        "intervals. The weak monitor is flat at 1.0; the mid monitor varies, and its English "
        "and Russian intervals overlap."
    ),
    "fig2_cca.png": (
        "Figure 2. Covert channel advantage against the mid monitor, relative to English, with "
        "95% bootstrap intervals. Values below 1 mean the language yields less undetected "
        "utility than English. The weak monitor admits no undetected utility in any language, "
        "so the ratio is undefined there and is not plotted."
    ),
    "fig3_selection.png": (
        "Figure 3. Study D language selection over 200 sequential trials, as a rolling share in "
        "the last 20 trials. The dotted line is the uniform rate."
    ),
    "fig4_shift.png": (
        "Figure 4. Probability that the picker keeps the same language at trial t+1, split by "
        "whether trial t was flagged, with 95% Wilson intervals."
    ),
    "fig5_script.png": (
        "Figure 5. Script control (Study E): share of seeded leaks flagged by the mid monitor in "
        "native and romanised script, with 95% Wilson intervals. Fifteen leaks per arm."
    ),
    "fig6_lpi.png": (
        "Figure 6. Language Preference Index for agent-card routing (Study A). Parity is 1.0."
    ),
}


def figure(name: str, sty: dict):
    path = FIG / name
    img = Image(str(path), width=6.15 * inch, height=3.55 * inch, kind="proportional")
    cap = Paragraph(CAPTIONS[name], sty["caption"])
    return KeepTogether([Spacer(1, 6), img, cap])


def table_flow(rows: list[list[str]], sty: dict):
    data = []
    for i, row in enumerate(rows):
        style = sty["th"] if i == 0 else sty["td"]
        data.append([Paragraph(rl(cell.strip()), style) for cell in row])
    ncols = len(data[0])
    width = 6.5 * inch
    col_w = [width / ncols] * ncols
    tbl = Table(data, colWidths=col_w, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f4f6f8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f4f6f8"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0b8c1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return KeepTogether([tbl, Spacer(1, 10)])


def parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c or "-") for c in cells):
            continue
        rows.append(cells)
    return rows


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#1f3a5f"))
    canvas.setLineWidth(0.6)
    canvas.line(0.75 * inch, letter[1] - 0.48 * inch, letter[0] - 0.75 * inch, letter[1] - 0.48 * inch)
    canvas.setFont("Times-Italic", 8)
    canvas.setFillColor(colors.HexColor("#444444"))
    canvas.drawString(0.75 * inch, letter[1] - 0.40 * inch, "Language as an Oversight Surface in A2A Systems")
    canvas.line(0.75 * inch, 0.55 * inch, letter[0] - 0.75 * inch, 0.55 * inch)
    canvas.drawCentredString(letter[0] / 2, 0.38 * inch, f"{doc.page}")
    canvas.restoreState()


def build() -> Path:
    sty = styles()
    story = []
    raw = MD.read_text(encoding="utf-8").splitlines()
    i = 0
    para: list[str] = []
    quote: list[str] = []
    bullets: list[str] = []
    numbered: list[str] = []
    table_lines: list[str] = []

    def flush_para():
        nonlocal para
        if para:
            story.append(Paragraph(rl(" ".join(para)), sty["body"]))
            para = []

    def flush_quote():
        nonlocal quote
        if quote:
            story.append(Paragraph(rl(" ".join(quote)), sty["quote"]))
            quote = []

    def flush_bullets():
        nonlocal bullets
        if bullets:
            for b in bullets:
                story.append(Paragraph(f"• {rl(b)}", sty["bullet"]))
            bullets = []

    def flush_numbered():
        nonlocal numbered
        if numbered:
            for n, item in enumerate(numbered, 1):
                story.append(Paragraph(f"{n}. {rl(item)}", sty["bullet"]))
            numbered = []

    def flush_table():
        nonlocal table_lines
        if table_lines:
            story.append(table_flow(parse_table(table_lines), sty))
            table_lines = []

    def flush_all():
        flush_para()
        flush_quote()
        flush_bullets()
        flush_numbered()
        flush_table()

    while i < len(raw):
        line = raw[i]
        stripped = line.strip()

        if stripped.startswith("![") and "](" in stripped:
            flush_all()
            path = stripped.split("](", 1)[1].rstrip(")")
            name = Path(path).name
            story.append(figure(name, sty))
            i += 1
            continue

        if stripped.startswith("|"):
            flush_para()
            flush_quote()
            flush_bullets()
            flush_numbered()
            table_lines.append(stripped)
            i += 1
            continue
        if table_lines:
            flush_table()

        if stripped == "---":
            flush_all()
            i += 1
            continue

        if stripped.startswith("# "):
            flush_all()
            story.append(Spacer(1, 8))
            story.append(Paragraph(rl(stripped[2:]), sty["title"]))
            i += 1
            continue
        if stripped.startswith("## "):
            flush_all()
            story.append(Paragraph(rl(stripped[3:]), sty["h1"]))
            i += 1
            continue
        if stripped.startswith("### "):
            flush_all()
            story.append(Paragraph(rl(stripped[4:]), sty["h2"]))
            i += 1
            continue

        if stripped.startswith("> "):
            flush_para()
            flush_bullets()
            quote.append(stripped[2:])
            i += 1
            continue
        if quote and not stripped.startswith(">"):
            flush_quote()

        if re.match(r"^\d+\.\s", stripped):
            flush_para()
            numbered.append(re.sub(r"^\d+\.\s+", "", stripped))
            i += 1
            continue
        if numbered and not re.match(r"^\d+\.\s", stripped):
            flush_numbered()

        if stripped.startswith("- "):
            flush_para()
            bullets.append(stripped[2:])
            i += 1
            continue
        if bullets and not stripped.startswith("- "):
            flush_bullets()

        if not stripped:
            flush_para()
            i += 1
            continue

        if stripped.startswith("**A benchmark"):
            flush_all()
            story.append(Paragraph(rl(stripped.strip("*")), sty["subtitle"]))
            i += 1
            continue

        para.append(stripped)
        i += 1

    flush_all()

    doc = SimpleDocTemplate(
        str(PDF),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Language as an Oversight Surface in Agent-to-Agent Systems",
        author="Language A2A benchmark",
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return PDF


if __name__ == "__main__":
    path = build()
    print(path)
