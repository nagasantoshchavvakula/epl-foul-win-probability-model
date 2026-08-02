#!/usr/bin/env python3
"""
Run:
    python reports/generate_report.py

This script reads `reports/EPL_Foul_Win_Probability_Model_Report.md` and
generates a real PDF at `reports/EPL_Foul_Win_Probability_Model_Report.pdf`
using ReportLab and platypus. It preserves headings, paragraphs, lists,
tables and code blocks and creates a Title page and Table of Contents.
"""
import os
import re
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    ListFlowable,
    ListItem,
    Preformatted,
)
from reportlab.platypus.tableofcontents import TableOfContents


INPUT_MD = os.path.join(os.path.dirname(__file__), "EPL_Foul_Win_Probability_Model_Report.md")
OUTPUT_PDF = os.path.join(os.path.dirname(__file__), "EPL_Foul_Win_Probability_Model_Report.pdf")


class MyDocTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        from reportlab.platypus.paragraph import Paragraph as _Paragraph

        if isinstance(flowable, _Paragraph):
            style_name = getattr(flowable.style, "name", "")
            m = re.match(r"Heading(\d+)", style_name)
            if m:
                level = int(m.group(1))
                text = flowable.getPlainText()
                # notify TOC
                self.notify('TOCEntry', (level, text, self.page))


def make_styles():
    styles = getSampleStyleSheet()
    if 'TitlePage' not in styles:
        styles.add(ParagraphStyle(name="TitlePage", parent=styles["Title"], alignment=1, spaceAfter=20, fontSize=28))
    else:
        styles['TitlePage'].alignment = 1
        styles['TitlePage'].spaceAfter = 20
        styles['TitlePage'].fontSize = 28
    if 'TOCTitle' not in styles:
        styles.add(ParagraphStyle(name="TOCTitle", parent=styles["Title"], alignment=0, fontSize=18))
    else:
        styles['TOCTitle'].alignment = 0
        styles['TOCTitle'].fontSize = 18

    # Update existing heading styles rather than re-adding
    if 'Heading1' in styles:
        styles['Heading1'].spaceBefore = 12
        styles['Heading1'].spaceAfter = 6
    else:
        styles.add(ParagraphStyle(name="Heading1", spaceBefore=12, spaceAfter=6))
    if 'Heading2' in styles:
        styles['Heading2'].spaceBefore = 10
        styles['Heading2'].spaceAfter = 4
    else:
        styles.add(ParagraphStyle(name="Heading2", spaceBefore=10, spaceAfter=4))
    if 'Heading3' in styles:
        styles['Heading3'].spaceBefore = 8
        styles['Heading3'].spaceAfter = 4
    else:
        styles.add(ParagraphStyle(name="Heading3", spaceBefore=8, spaceAfter=4))

    if 'BodyText' in styles:
        styles['BodyText'].spaceAfter = 6
    else:
        styles.add(ParagraphStyle(name="BodyText", spaceAfter=6))
    if 'Code' in styles:
        styles['Code'].fontName = 'Courier'
        styles['Code'].fontSize = 8
        styles['Code'].leading = 10
    else:
        styles.add(ParagraphStyle(name="Code", fontName='Courier', fontSize=8, leading=10))
    return styles


def parse_markdown(md_text):
    lines = md_text.splitlines()
    i = 0
    items = []
    while i < len(lines):
        line = lines[i]
        if line.strip() == "":
            i += 1
            continue

        # fenced code block
        if line.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            items.append({"type": "code", "text": "\n".join(code_lines)})
            continue

        # heading
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            items.append({"type": "heading", "level": level, "text": text})
            i += 1
            continue

        # bullet list
        if re.match(r"^[\-\*\+]\s+", line):
            bullets = []
            while i < len(lines) and re.match(r"^[\-\*\+]\s+", lines[i]):
                bullets.append(re.sub(r"^[\-\*\+]\s+", "", lines[i]).strip())
                i += 1
            items.append({"type": "bullets", "items": bullets})
            continue

        # table (markdown pipe table)
        if '|' in line:
            # check for header separator on next line
            if i + 1 < len(lines) and re.match(r"^\s*\|?\s*[:\-]+", lines[i + 1]):
                header = [c.strip() for c in line.strip().strip('|').split('|')]
                i += 2
                rows = []
                while i < len(lines) and '|' in lines[i]:
                    rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
                    i += 1
                items.append({"type": "table", "header": header, "rows": rows})
                continue

        # paragraph
        para_lines = [line.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() != "" and not re.match(r"^(#{1,6})\s+", lines[i]) and not lines[i].startswith("```") and not re.match(r"^[\-\*\+]\s+", lines[i]) and '|' not in lines[i]:
            para_lines.append(lines[i].strip())
            i += 1
        items.append({"type": "para", "text": " ".join(para_lines)})

    return items


def build_pdf(md_path, out_pdf):
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    items = parse_markdown(text)

    styles = make_styles()

    doc = MyDocTemplate(out_pdf, pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=1 * inch, bottomMargin=1 * inch,)

    # use Canvas maker to ensure PDF version 1.4
    def canvas_maker(filename, pagesize=letter):
        return Canvas(filename, pagesize=pagesize, pdfVersion='1.4')

    doc.canvasmaker = canvas_maker

    story = []

    # Title page: first H1 as title if present
    title = None
    for it in items:
        if it['type'] == 'heading' and it['level'] == 1:
            title = it['text']
            break
    if not title:
        title = 'EPL Foul Win Probability Model Report'

    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph(title, styles['TitlePage']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph('Author: EPL Analytics Team', styles['BodyText']))
    story.append(Paragraph(f'Date: {datetime.utcnow().strftime("%Y-%m-%d")}', styles['BodyText']))
    story.append(PageBreak())

    # Table of Contents
    story.append(Paragraph('Table of Contents', styles['TOCTitle']))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(fontSize=12, name='TOCLevel1', leftIndent=20, firstLineIndent=-20, spaceBefore=5),
        ParagraphStyle(fontSize=10, name='TOCLevel2', leftIndent=40, firstLineIndent=-20, spaceBefore=2),
        ParagraphStyle(fontSize=9, name='TOCLevel3', leftIndent=60, firstLineIndent=-20, spaceBefore=1),
    ]
    story.append(Spacer(1, 0.1 * inch))
    story.append(toc)
    story.append(PageBreak())

    # Content
    for it in items:
        if it['type'] == 'heading':
            lvl = it['level']
            style_name = f'Heading{min(lvl,3)}'
            story.append(Paragraph(it['text'], styles[style_name]))
            story.append(Spacer(1, 0.05 * inch))
        elif it['type'] == 'para':
            story.append(Paragraph(it['text'], styles['BodyText']))
        elif it['type'] == 'bullets':
            list_items = [ListItem(Paragraph(b, styles['BodyText']), leftIndent=10) for b in it['items']]
            story.append(ListFlowable(list_items, bulletType='bullet', leftIndent=20))
        elif it['type'] == 'table':
            data = [it['header']] + it['rows']
            tbl = Table(data, hAlign='LEFT')
            tbl.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 0.1 * inch))
        elif it['type'] == 'code':
            story.append(Preformatted(it['text'], styles['Code']))
        else:
            # Fallback to plain paragraph
            story.append(Paragraph(str(it), styles['BodyText']))

    def draw_page_number(canvas_obj, doc_obj):
        canvas_obj.saveState()
        canvas_obj.setFont('Helvetica', 9)
        page_num = f"Page {doc_obj.page}"
        canvas_obj.drawRightString(7.5 * inch, 0.65 * inch, page_num)
        canvas_obj.restoreState()

    # Build PDF
    doc.build(story, onFirstPage=draw_page_number, onLaterPages=draw_page_number)


def main():
    if not os.path.exists(INPUT_MD):
        print(f"Markdown file not found: {INPUT_MD}")
        return
    print(f"Generating PDF: {OUTPUT_PDF}")
    build_pdf(INPUT_MD, OUTPUT_PDF)
    print("Done.")


if __name__ == '__main__':
    main()
