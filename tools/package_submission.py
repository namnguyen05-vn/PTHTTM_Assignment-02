from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables"
OUT.mkdir(exist_ok=True)

report_docx = ROOT / "Bao_cao_ASG_02_hoan_chinh_sanitized.docx"
report_pdf = ROOT / "output" / "pdf" / "Bao_cao_ASG_02_hoan_chinh.pdf"
parts = [
    ("Báo cáo tổng hợp", report_pdf),
    ("Notebook 1 — Diabetes", ROOT / "submission_pdfs" / "ASG_02_1.pdf"),
    ("Notebook 2 — Housing", ROOT / "submission_pdfs" / "ASG_02_2.pdf"),
    ("Notebook 3 — E-commerce", ROOT / "submission_pdfs" / "ASG_02_3.pdf"),
]

final_pdf = OUT / "ASG_02_NguyenNgocHoangNam_B23DCCN585.pdf"
writer = PdfWriter()
start_pages = []
for title, filename in parts:
    start_pages.append((title, len(writer.pages)))
    reader = PdfReader(filename)
    for page in reader.pages:
        writer.add_page(page)
for title, page_number in start_pages:
    writer.add_outline_item(title, page_number)
writer.add_metadata({
    "/Title": "Assignment 02 — Intelligent Systems Development",
    "/Author": "Nguyễn Ngọc Hoàng Nam — B23DCCN585",
})
with final_pdf.open("wb") as handle:
    writer.write(handle)

final_docx = OUT / "Bao_cao_ASG_02_hoan_chinh.docx"
shutil.copy2(report_docx, final_docx)

project_zip = OUT / "ASG_02_Project_Complete.zip"
include_files = [
    "ASG_02_1.ipynb", "ASG_02_2.ipynb", "ASG_02_3.ipynb",
    "diabetes_dataset.csv", "VN_housing_dataset.csv", "Womens Clothing E-Commerce Reviews.csv",
    "main.py", "index.html", "requirements.txt", "README.md", ".gitignore",
]
include_dirs = ["models", "artifacts", "figures", "demo", "tests", "tools"]

with zipfile.ZipFile(project_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for filename in include_files:
        path = ROOT / filename
        archive.write(path, Path("ASG_02") / filename)
    for dirname in include_dirs:
        for path in sorted((ROOT / dirname).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, Path("ASG_02") / path.relative_to(ROOT))
    archive.write(final_docx, Path("ASG_02") / "submission" / final_docx.name)
    archive.write(final_pdf, Path("ASG_02") / "submission" / final_pdf.name)

print(final_pdf)
print(final_docx)
print(project_zip)
