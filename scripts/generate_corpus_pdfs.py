import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONT_NAME = "Nirmala"
FONT_PATH = "C:/Windows/Fonts/Nirmala.ttc"
pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH, subfontIndex=0))
pdfmetrics.registerFont(TTFont(f"{FONT_NAME}-Bold", "C:/Windows/Fonts/Nirmalab.ttf" if os.path.exists("C:/Windows/Fonts/Nirmalab.ttf") else FONT_PATH, subfontIndex=0))

CORPUS_DIR = os.path.join("data", "corpus")

def convert_txt_to_pdf(txt_path, pdf_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        name="SanskritTitle",
        fontName=f"{FONT_NAME}",
        fontSize=18,
        leading=24,
        textColor=colors.HexColor("#7c2d12"), # Deep warm amber/crimson
        alignment=1, # Center
        spaceAfter=12
    )
    
    subtitle_style = ParagraphStyle(
        name="SanskritSubtitle",
        fontName=f"{FONT_NAME}",
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#4b5563"),
        alignment=1,
        spaceAfter=15
    )
    
    body_style = ParagraphStyle(
        name="SanskritBody",
        fontName=FONT_NAME,
        fontSize=12,
        leading=19,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=10,
        firstLineIndent=20
    )
    
    verse_style = ParagraphStyle(
        name="SanskritVerse",
        fontName=FONT_NAME,
        fontSize=12,
        leading=18,
        textColor=colors.HexColor("#991b1b"),
        alignment=1,
        spaceBefore=8,
        spaceAfter=8
    )
    
    story = []
    
    # Header
    header_style = ParagraphStyle(
        name="Header",
        fontName=FONT_NAME,
        fontSize=9,
        textColor=colors.HexColor("#9ca3af"),
        alignment=2
    )
    story.append(Paragraph("JnanaSetu – ज्ञानसेतुः संस्कृतज्ञानसहायकः", header_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb"), spaceAfter=15))
    
    title = lines[0].strip() if lines else "संस्कृतकथा"
    story.append(Paragraph(title, title_style))
    story.append(Paragraph("मूलसंस्कृतग्रन्थसङ्ग्रहः • Authentic Sanskrit Corpus", subtitle_style))
    story.append(Spacer(1, 10))
    
    for line in lines[1:]:
        text = line.strip()
        if not text:
            story.append(Spacer(1, 8))
            continue
            
        if "॥" in text or text.startswith("वरं ") or text.startswith("यथा ") or text.startswith("उद्यमेन ") or text.startswith("बुद्धिर्यस्य ") or text.startswith("माघे ") or text.startswith("त्वयि "):
            story.append(Paragraph(f"<b>{text}</b>", verse_style))
        elif text.startswith("(") and text.endswith(")"):
            moral_style = ParagraphStyle(
                name="SanskritMoral",
                fontName=FONT_NAME,
                fontSize=11,
                leading=16,
                textColor=colors.HexColor("#065f46"),
                alignment=1,
                spaceBefore=6,
                spaceAfter=6
            )
            story.append(Paragraph(f"<i>{text}</i>", moral_style))
        else:
            story.append(Paragraph(text, body_style))
            
    doc.build(story)
    print(f"Generated PDF: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")

def main():
    print("=== Generating Authentic Sanskrit PDF Documents ===")
    txt_files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".txt")]
    for txt_file in sorted(txt_files):
        base_name = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(CORPUS_DIR, txt_file)
        pdf_path = os.path.join(CORPUS_DIR, f"{base_name}.pdf")
        convert_txt_to_pdf(txt_path, pdf_path)
    print("=== All PDF Documents Successfully Generated! ===")

if __name__ == "__main__":
    main()

