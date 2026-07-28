"""Download the full ABCD corpus (10k conversations, ~30MB gzipped).

The sample of three conversations in data/ is enough to develop against.
Fetch the full set only when you want real distributions in the metrics.

Builds an explicit SSL context from certifi when it is available. A stock
python.org or pyenv install on macOS has no certificate store wired up, and the
default context then fails with CERTIFICATE_VERIFY_FAILED.
"""

import gzip
import shutil
import ssl
import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/asappresearch/abcd/raw/master/data/abcd_v1.1.json.gz"
OUT = Path(__file__).resolve().parents[1] / "data" / "abcd_v1.1.json"
GZ = OUT.with_suffix(".json.gz")


def _context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {URL}")
    try:
        with urllib.request.urlopen(URL, context=_context()) as resp, open(GZ, "wb") as f:
            shutil.copyfileobj(resp, f)
    except urllib.error.URLError as exc:
        # urlopen wraps the SSL failure in URLError, so catching
        # SSLCertVerificationError directly never fires.
        if not isinstance(exc.reason, ssl.SSLError):
            raise
        sys.exit(
            f"TLS verification failed: {exc.reason}\n\n"
            "On macOS this usually means Python has no certificate store:\n"
            "  /Applications/Python\\ 3.10/Install\\ Certificates.command\n"
            "or\n"
            "  pip install --upgrade certifi\n"
            "  export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')\n\n"
            f"Fastest workaround, curl uses the system store:\n"
            f"  curl -L -o data/abcd_v1.1.json.gz {URL}\n"
            "  gunzip data/abcd_v1.1.json.gz"
        )

    with gzip.open(GZ, "rb") as f_in, open(OUT, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    GZ.unlink()

    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"set TARS_ABCD={OUT}")


if __name__ == "__main__":
    main()
