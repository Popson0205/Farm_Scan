"""Files FarmScan keeps: drone orthomosaics and the soil dataset. In the platform's S3-compatible bucket, under
S3_PREFIX (farm/), when S3 is set up; otherwise in backend/uploads (development and tests).

Raster libraries read local files, so a stored file is fetched into a local cache before it is read (and stays there
while the service runs)."""
from pathlib import Path

from config import settings

LOCAL = Path(__file__).resolve().parent / "uploads"
CACHE = Path(__file__).resolve().parent / ".file-cache"
_s3 = None


def _client():
    global _s3
    if _s3 is None:
        import boto3
        from botocore.client import Config
        endpoint = settings.s3_endpoint if "://" in settings.s3_endpoint else f"https://{settings.s3_endpoint}"
        region = settings.s3_region or ("auto" if "r2.cloudflarestorage.com" in endpoint else "us-east-1")
        _s3 = boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=settings.s3_access_key,
                           aws_secret_access_key=settings.s3_secret_key, region_name=region,
                           config=Config(signature_version="s3v4", s3={"addressing_style": "path"}))
    return _s3


def key(*parts: str) -> str:
    return settings.s3_prefix + "/".join(p.strip("/") for p in parts)


def put_file(path: Path, k: str, content_type: str = "application/octet-stream") -> None:
    if settings.s3_on:
        _client().upload_file(str(path), settings.s3_bucket, k, ExtraArgs={"ContentType": content_type})
        dest = CACHE / k                     # keep the local copy as the cached one
    else:
        dest = LOCAL / k
    dest.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve() != dest.resolve():
        path.replace(dest)


def local_path(k: str) -> Path | None:
    """A local copy of a stored file (fetched from the bucket the first time), or None if there is no such file."""
    if not settings.s3_on:
        p = LOCAL / k
        return p if p.exists() else None
    p = CACHE / k
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    try:
        _client().download_file(settings.s3_bucket, k, str(tmp))
    except Exception:
        tmp.unlink(missing_ok=True)
        return None
    tmp.replace(p)
    return p


def delete(k: str) -> None:
    for base in (LOCAL, CACHE):
        (base / k).unlink(missing_ok=True)
    if settings.s3_on:
        try:
            _client().delete_object(Bucket=settings.s3_bucket, Key=k)
        except Exception:
            pass
