
# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
# Source lives in the code repository; artifacts have a separate root.
import sys as _sys
from pathlib import Path as _Path


SIGNATURES = {
    "in": (
        ["person", "object", "container", "place", "machine"],
        ["container", "place", "region"],
    ),
    "carries": (["person"], ["object", "container"]),
    "member": (["person"], ["team"]),
    "based": (["team"], ["place"]),
    "assigned": (["person"], ["place", "region"]),
    "isa": (["object"], ["category"]),
    "subtype": (["category"], ["category"]),
    "has": (["category"], ["property"]),
    "property": (["object"], ["property"]),
    "trained": (["person"], ["machine"]),
    "licensed": (["person"], ["machine"]),
    "may": (["person"], ["machine"]),
    "adjacent": (["place"], ["place"]),
}
