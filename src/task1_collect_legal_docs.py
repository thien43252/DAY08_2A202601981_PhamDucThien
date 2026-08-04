"""
Task 1 - Thu thap van ban chinh sach/quy dinh dich vu dai hoc.

The source documents are official UET/VNU PDFs.  Files are stored in
``data/landing/legal/`` so Task 3 can convert them to Markdown.

Usage::

    python -m src.task1_collect_legal_docs
    python -m src.task1_collect_legal_docs --force

The downloader is intentionally idempotent: an existing, valid PDF is kept
unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "landing" / "legal"
MIN_FILE_SIZE = 1024
DEFAULT_TIMEOUT = (10, 60)  # connect, read (seconds)


@dataclass(frozen=True)
class LegalDocument:
    filename: str
    url: str
    description: str


DOCUMENTS: tuple[LegalDocument, ...] = (
    LegalDocument(
        "tuition-fees-uet.pdf",
        "https://uet.vnu.edu.vn/wp-content/uploads/2021/03/"
        "Quy-dinh-ve-viec-nop-hoc-phi-tai-truong-DHCN-2021_final.pdf",
        "Quy dinh ve viec nop hoc phi tai UET",
    ),
    LegalDocument(
        "scholarship-regulations-vnu-uet.pdf",
        "https://uet.vnu.edu.vn/wp-content/uploads/2024/10/"
        "Signed.Signed.Signed.2024_10_4_QUY-DINH-VE-HOC-BONG.pdf",
        "Quy dinh hoc bong cua DHQGHN ap dung cho sinh vien UET",
    ),
    LegalDocument(
        "dormitory-regulations-vnu-uet.pdf",
        "https://student.ulis.vnu.edu.vn/files/uploads/2025/09/"
        "2023_02_16_Quy-che-cong-tac-SV-noi-tru-tai-DHQGHN.pdf",
        "Quy che cong tac sinh vien noi tru tai KTX DHQGHN",
    ),
    LegalDocument(
        "course-registration-regulations-vnu-uet.pdf",
        "https://student.ulis.vnu.edu.vn/files/uploads/2025/09/"
        "3626_21.10.2022.-Quy-che-dao-tao-dai-hoc-tai-DHQGHN-ap-dung-tu-khoa-QH2022-5.pdf",
        "Quy che dao tao dai hoc, bao gom dang ky hoc phan",
    ),
)


def setup_directory(output_dir: Path = DATA_DIR) -> Path:
    """Create and return a landing directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def is_valid_pdf(path: Path, min_size: int = MIN_FILE_SIZE) -> bool:
    """Return True only when *path* looks like a non-empty PDF."""
    if not path.is_file() or path.stat().st_size <= min_size:
        return False
    try:
        with path.open("rb") as stream:
            return stream.read(5) == b"%PDF-"
    except OSError:
        return False


def download_file(
    document: LegalDocument,
    *,
    output_dir: Path = DATA_DIR,
    force: bool = False,
    session: requests.Session | None = None,
) -> Path:
    """Download one official document and validate its PDF signature."""
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / document.filename

    if not force and is_valid_pdf(destination):
        return destination

    client = session or requests.Session()
    response = client.get(
        document.url,
        headers={"User-Agent": "university-services-rag/1.0"},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    content = response.content
    if len(content) <= MIN_FILE_SIZE or not content.startswith(b"%PDF-"):
        raise ValueError(
            f"Nguon khong tra ve PDF hop le cho {document.filename} "
            f"(content-type={response.headers.get('content-type', 'unknown')})"
        )

    # Write only after validation, preventing a failed HTML/error response
    # from replacing a previously good landing file.
    destination.write_bytes(content)
    return destination


def collect_documents(
    documents: Iterable[LegalDocument] = DOCUMENTS,
    *,
    output_dir: Path = DATA_DIR,
    force: bool = False,
) -> list[Path]:
    """Download all configured documents and return their local paths."""
    setup_directory(output_dir)
    downloaded: list[Path] = []
    with requests.Session() as session:
        for document in documents:
            path = download_file(
                document, output_dir=output_dir, force=force, session=session
            )
            if not is_valid_pdf(path):
                raise ValueError(f"Tep tai ve khong hop le: {path}")
            downloaded.append(path)
            print(f"OK {document.filename} - {document.description}")
    return downloaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Tai lai va ghi de cac PDF da ton tai.",
    )
    args = parser.parse_args()
    paths = collect_documents(force=args.force)
    print(f"Da san sang {len(paths)} tai lieu trong: {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
