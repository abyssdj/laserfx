from PIL import ImageFont
_DIR = "/usr/share/fonts/opentype/inter/"
_cache = {}


def families(root=None):
    return ["Inter", "DejaVu Sans"]


def _pil(f, ss=1):
    key = (f.family, f.size, f.weight, ss)
    if key not in _cache:
        file = "Inter-Bold.otf" if f.weight == "bold" else "Inter-Regular.otf"
        _cache[key] = ImageFont.truetype(_DIR + file, max(1, int(round(f.size * 96 / 72 * ss))))
    return _cache[key]


class Font:
    def __init__(self, root=None, family="Inter", size=10, weight="normal"):
        self.family, self.size, self.weight = family, size, weight

    def measure(self, text):
        return int(_pil(self).getlength(text))
