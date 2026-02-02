from pathlib import Path

from docling.document_converter import DocumentConverter
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
INPUTS_DIR = (
    BASE_DIR
    / "data"
    / "inputs"
)
OUTPUT_DIR = Path("processed_data")

converter = DocumentConverter()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".txt", ".csv"}

for file_path in tqdm(INPUTS_DIR.rglob("*")):
    print("filepath: ", file_path)
    if not file_path.is_file():
        continue
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        continue

    # Preserve directory of inputs
    relative_path = file_path.relative_to(INPUTS_DIR)
    output_path = OUTPUT_DIR / relative_path.with_suffix(".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to Markdown
    print(f"Converting: {relative_path}")
    result = converter.convert(file_path)
    markdown = result.document.export_to_markdown()

    # output_file = OUTPUT_DIR / f"{file_path.stem}.md"
    print("output_file: ", output_path)
    output_path.write_text(markdown, encoding="utf-8")

print("✅ Conversion complete")
