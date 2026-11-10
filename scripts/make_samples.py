"""Generate the sample files in ./samples (English + Arabic .docx, mixed .txt).

    python scripts/make_samples.py
"""
from __future__ import annotations

from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Pt, RGBColor

OUT = Path(__file__).resolve().parent.parent / "samples"


def rtl(paragraph):
    """Mark a paragraph (and its runs) right-to-left so Word shows Arabic correctly."""
    ppr = paragraph._p.get_or_add_pPr()
    bidi = OxmlElement("w:bidi")
    ppr.insert(0, bidi)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in paragraph.runs:
        rpr = run._r.get_or_add_rPr()
        rpr.append(OxmlElement("w:rtl"))
    return paragraph


def add_hyperlink(paragraph, url: str, text: str):
    r_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    rpr.extend([color, underline])
    t = OxmlElement("w:t")
    t.text = text
    run.extend([rpr, t])
    link.append(run)
    paragraph._p.append(link)


def english():
    d = docx.Document()
    d.core_properties.author = "Sarah Johnson"
    d.sections[0].header.paragraphs[0].text = "Confidential - prepared by Sarah Johnson"
    d.sections[0].footer.paragraphs[0].text = "Acme Corporation - internal use only"
    d.add_heading("Meeting Minutes", level=1)

    p = d.add_paragraph("On 12 March 2024, ")
    r = p.add_run("Bar")
    r.bold = True
    r = p.add_run("ack Obama")  # entity split across two runs with different formatting
    r.italic = True
    p.add_run(" met officials from the United Nations in Baghdad to discuss the new program.")

    d.add_paragraph("The Iraqi delegation, led by Dr. Ahmed Hassan, a senior engineer at Microsoft, "
                    "arrived in Amman on Monday.")
    p = d.add_paragraph("Contact: john.smith@acme.com, phone +1 (212) 555-0100. Portal: ")
    add_hyperlink(p, "https://portal.acme.com/login", "https://portal.acme.com/login")
    d.add_paragraph("Payment IBAN GB82 WEST 1234 5698 7654 32, card 4111 1111 1111 1111, "
                    "server 192.168.1.20, passport number 548211930.")

    t = d.add_table(rows=3, cols=3)
    t.style = "Table Grid"
    for row, vals in zip(t.rows, [("Name", "Company", "City"), ("Emily Clark", "Google", "London"),
                                  ("Omar Khalid", "Siemens", "Berlin")]):
        for cell, v in zip(row.cells, vals):
            cell.text = v
    run = t.rows[0].cells[0].paragraphs[0].runs[0]
    run.bold = True

    p = d.add_paragraph()
    r = p.add_run("Later, Obama thanked Ahmed Hassan and Emily Clark for their support.")
    r.font.color.rgb = RGBColor(0x2A, 0x78, 0xD6)
    r.font.size = Pt(12)
    d.save(OUT / "sample_en.docx")


def arabic():
    d = docx.Document()
    d.core_properties.author = "سامر الخطيب"
    hp = d.sections[0].header.paragraphs[0]
    hp.add_run("سرّي - إعداد المهندس سامر الخطيب")
    rtl(hp)
    fp = d.sections[0].footer.paragraphs[0]
    fp.add_run("وثيقة داخلية - شركة الأفق للاستشارات")
    rtl(fp)
    rtl(d.add_heading("محضر اجتماع", level=1))

    p = d.add_paragraph("في ١٢ آذار ٢٠٢٤ التقى الرئيس ")
    r = p.add_run("محمد الأحمد")
    r.bold = True
    p.add_run(" بممثلين عن الأمم المتحدة في مقر وزارة الخارجية بدمشق.")
    rtl(p)
    rtl(d.add_paragraph("حضر الاجتماع الدكتور خالد النجار، مدير شركة سيرياتيل، والمهندسة ليلى حسن "
                        "من جامعة دمشق، إضافة إلى وفد سعودي."))
    rtl(d.add_paragraph("للتواصل: البريد khaled.najjar@example.com والهاتف +963 933 123 456 أو 0944123456."))
    rtl(d.add_paragraph("رقم الحساب SA03 8000 0000 6080 1016 7519 ورقم الهوية الوطنية 01020304050."))

    t = d.add_table(rows=3, cols=3)
    t.style = "Table Grid"
    for row, vals in zip(t.rows, [("الاسم", "الجهة", "المدينة"), ("أحمد العلي", "بنك سورية الدولي", "حلب"),
                                  ("سارة يوسف", "منظمة الصحة العالمية", "بيروت")]):
        for cell, v in zip(row.cells, vals):
            cell.text = v
            rtl(cell.paragraphs[0])

    rtl(d.add_paragraph("وقد شكر محمد الأحمد الحضور، وأكد خالد النجار أن التقرير سيُرسل إلى الرياض "
                        "بتاريخ 15/04/2024."))
    d.save(OUT / "sample_ar.docx")


def mixed_txt():
    (OUT / "sample_mixed.txt").write_text(
        "تقرير المتابعة - Follow-up report\n"
        "راجع المريض أحمد السالم عيادة الدكتورة ريم حداد في مستشفى الشامي بتاريخ 03/09/2026.\n"
        "Patient Ahmad Salem was referred to Dr. Reem Haddad at Al-Shami Hospital in Damascus.\n"
        "Diagnosis: type 2 diabetes. Prescribed metformin 500 mg.\n"
        "للتواصل: ahmad.salem@example.com أو 0933111222.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    english()
    arabic()
    mixed_txt()
    print("samples written to", OUT)
