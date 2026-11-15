"""Pinned, source-only acquisition and inspection of the Broll iGlu archive."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

PACKAGE = "iglu"
VERSION = "4.2.2"
FILENAME = f"{PACKAGE}_{VERSION}.tar.gz"
PRIMARY_URL = f"https://cran.r-project.org/src/contrib/{FILENAME}"
ARCHIVE_URL = f"https://cran.r-project.org/src/contrib/Archive/{PACKAGE}/{FILENAME}"
TARBALL_SHA256 = "d63a14b7fb49a9fb031fb3e597e36f1914f3f31bfb02b9ed656fcfa763c2aa59"
RDA_MEMBER = "iglu/data/example_data_5_subject.rda"
RDA_SHA256 = "1eeb7e14883faa78745ee141592f6b06d63613614c7ba29e947f4055535742de"
OBJECT_NAME = "example_data_5_subject"
DOCUMENT_MEMBERS = (
    "iglu/DESCRIPTION",
    "iglu/man/example_data_5_subject.Rd",
    "iglu/README.md",
)
ALLOWED_READ_MEMBERS = frozenset((RDA_MEMBER, *DOCUMENT_MEMBERS))
ALLOWED_DOWNLOAD_HOSTS = frozenset({"cran.r-project.org"})
MAX_TARBALL_BYTES = 32 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 30

MANIFEST_COLUMNS = (
    "dataset", "artifact_kind", "package", "version", "logical_source", "source_url",
    "member_name", "file_name", "size_bytes", "sha256", "validated_at_utc",
    "requested_url", "resolved_url", "adult_eligibility_status",
    "eligible_age_range_years", "participant_level_ages", "adult_evidence_type",
    "disease_label_scope", "oral_medication_status", "parent_study_insulin_therapy",
    "parent_study_osa_selection", "five_subject_selection_mechanism",
    "source_timezone_evidence", "unit_evidence", "disease_label_evidence",
    "device_evidence", "software_license", "data_license_status",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_tarball(path: Path) -> None:
    if sha256_file(path) != TARBALL_SHA256:
        raise ValueError("Broll package tarball SHA256 mismatch")


def _validate_members(members: Iterable[tarfile.TarInfo]) -> dict[str, tarfile.TarInfo]:
    result = {}
    for member in members:
        name = member.name
        parsed = PurePosixPath(name)
        if (not name or parsed.is_absolute() or ".." in parsed.parts or
                "\\" in name or any(part in ("", ".") for part in name.split("/")[:-1])):
            raise ValueError(f"unsafe tar member path: {name!r}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"unsupported tar member type: {name!r}")
        if name in result:
            raise ValueError(f"duplicate tar member: {name!r}")
        result[name] = member
    if RDA_MEMBER not in result or not result[RDA_MEMBER].isfile():
        raise ValueError(f"missing expected RDA member: {RDA_MEMBER}")
    for name in DOCUMENT_MEMBERS:
        if name not in result or not result[name].isfile():
            raise ValueError(f"missing expected documentation member: {name}")
    return result


def inspect_archive(path: Path) -> tuple[list[str], dict[str, bytes]]:
    validate_tarball(path)
    with tarfile.open(path, "r:gz") as archive:
        members = _validate_members(archive.getmembers())
        contents = {}
        for name in ALLOWED_READ_MEMBERS:
            stream = archive.extractfile(members[name])
            if stream is None:
                raise ValueError(f"cannot read tar member: {name}")
            with stream:
                contents[name] = stream.read()
    if hashlib.sha256(contents[RDA_MEMBER]).hexdigest() != RDA_SHA256:
        raise ValueError("Broll embedded RDA SHA256 mismatch")
    return list(members), contents


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    requested_url: str = ""
    resolved_url: str = ""


def _validate_download_url(url: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS or
            parsed.username is not None or parsed.password is not None or
            parsed.port not in (None, 443) or parsed.fragment):
        raise ValueError(f"disallowed Broll download URL: {url}")


class _RestrictedRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_download_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def publish_verified_download(directory: Path, urls: tuple[str, ...] = (PRIMARY_URL, ARCHIVE_URL)) -> DownloadResult:
    """Download to a temporary file and publish only verified bytes without replacement."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / FILENAME
    if destination.exists():
        validate_tarball(destination)
        return DownloadResult(destination)
    last_error = None
    for url in urls:
        temporary = None
        try:
            _validate_download_url(url)
            with tempfile.NamedTemporaryFile(dir=directory, prefix=f".{FILENAME}.", delete=False) as output:
                temporary = Path(output.name)
                with build_opener(_RestrictedRedirectHandler()).open(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                    resolved_url = response.geturl()
                    _validate_download_url(resolved_url)
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None and int(content_length) > MAX_TARBALL_BYTES:
                        raise ValueError("Broll download Content-Length exceeds size limit")
                    downloaded = 0
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        downloaded += len(chunk)
                        if downloaded > MAX_TARBALL_BYTES:
                            raise ValueError("Broll download exceeds size limit")
                        output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            validate_tarball(temporary)
            try:
                os.link(temporary, destination)  # atomic publication; never replaces an existing file
            except FileExistsError:
                validate_tarball(destination)
            return DownloadResult(destination, url, resolved_url)
        except (OSError, ValueError) as error:
            last_error = error
            if destination.exists() and sha256_file(destination) != TARBALL_SHA256:
                raise ValueError("refusing to overwrite a conflicting package tarball") from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError("all pinned iGlu download URLs failed or failed checksum") from last_error


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    for row in rows:
        if set(row) != set(columns):
            raise ValueError(f"manifest columns differ: {set(row) ^ set(columns)}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def package_evidence(contents: dict[str, bytes]) -> list[dict[str, str]]:
    description = contents["iglu/DESCRIPTION"].decode("utf-8")
    rd = contents["iglu/man/example_data_5_subject.Rd"].decode("utf-8")
    readme = contents["iglu/README.md"].decode("utf-8")
    if "Package: iglu" not in description or "Version: 4.2.2" not in description:
        raise ValueError("package metadata does not match pinned package")
    def evidence(field: str, value: str, status: str, member: str, note: str) -> dict[str, str]:
        return dict(field=field, value=value, status=status, evidence_member=member, note=note)
    license_line = next((line.split(":", 1)[1].strip() for line in description.splitlines()
                         if line.startswith("License:")), "unknown")
    return [
        evidence("software_license", license_line, "documented" if license_line != "unknown" else "unknown", "iglu/DESCRIPTION", "Package license only; not a separate data license."),
        evidence("data_reuse_license", "no separate data-specific license identified", "unresolved standalone data license", "iglu/DESCRIPTION;iglu/man/example_data_5_subject.Rd;iglu/README.md;Broll et al.", "The GPL-2 software license, CC BY article license, and public data availability do not establish separate data terms."),
        evidence("glucose_unit", "mg/dL" if "glucose level as measured by CGM (mg/dL)" in rd else "unknown", "documented" if "glucose level as measured by CGM (mg/dL)" in rd else "unknown", "iglu/man/example_data_5_subject.Rd", "Unit of gl column."),
        evidence("device", "Dexcom G4" if "Dexcom G4 CGM measurements" in rd else "unknown", "documented" if "Dexcom G4 CGM measurements" in rd else "unknown", "iglu/man/example_data_5_subject.Rd", "README also identifies Dexcom G4." if "Dexcom G4 CGM" in readme else "Rd documentation identifies device."),
        evidence("disease_group", "Type II diabetes" if "5 subjects with Type II diabetes" in rd else "unknown", "documented" if "5 subjects with Type II diabetes" in rd else "unknown", "iglu/man/example_data_5_subject.Rd", "Dataset-level description; no row-level disease label field."),
        evidence("adult_eligibility", "confirmed at parent-study level", "linked parent-study eligibility", "Broll et al.;Gaynanova et al.;HYPNOS design and eligibility report", "Eligible age 21–75 years; individual ages unavailable in the released object."),
        evidence("eligible_age_range_years", "21–75", "linked parent-study eligibility", "HYPNOS design and eligibility report", "Parent-study criterion, not individual-level ages."),
        evidence("participant_level_ages", "unavailable", "source object schema", RDA_MEMBER, "No age column in the released object."),
        evidence("disease_label_scope", "dataset/study level", "documented", "iglu/man/example_data_5_subject.Rd", "No participant-level disease-label field."),
        evidence("oral_medication_status", "unknown", "unresolved", "iglu/man/example_data_5_subject.Rd", "Do not infer use from the insulin exclusion."),
        evidence("parent_study_insulin_therapy", "excluded", "linked parent-study eligibility", "Gaynanova et al.;HYPNOS design and eligibility report", "No insulin therapy in parent study."),
        evidence("parent_study_osa_selection", "untreated moderate-to-severe", "linked parent-study eligibility", "Gaynanova et al.;HYPNOS design and eligibility report", "Analysis ODI and trial AHI thresholds are distinct."),
        evidence("five_subject_selection_mechanism", "unknown", "unresolved", "Broll et al.;iglu/man/example_data_5_subject.Rd", "No selection rule for the five released participants documented."),
        evidence("timezone", "pending RDA attribute inventory", "pending source inspection", "iglu/data/example_data_5_subject.rda", "Package documentation does not establish timezone."),
    ]
