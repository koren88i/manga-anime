"""
On-demand API client for the One Piece Fandom Wiki (MediaWiki API).

Provides structured data for episodes and characters without bulk scraping.
Each function makes one HTTP GET request and parses the response.
Also includes a hardcoded episode-to-chapter mapping (no network needed).

Usage:
    from one_piece_wiki_api import get_episode_characters, get_episode_summary, ...

    chars = get_episode_characters(1)
    # [{"name": "Gol D. Roger", "wiki_page": "Gol D. Roger", "annotation": "flashback"}, ...]

    summary = get_episode_summary(1)
    # "The episode opens with..."

    info = get_character_info("Monkey D. Luffy")
    # {"name": "Monkey D. Luffy", "image_url": "https://...", "japanese_name": "...", ...}

    ch = get_episode_chapters(1)
    # {"episode": 1, "chapters": [2], "is_filler": False}

    ep = get_chapter_episodes(2)
    # {"chapter": 2, "episodes": [1]}
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import re
import json
import urllib.request
import urllib.parse
from html.parser import HTMLParser


BASE_URL = "https://onepiece.fandom.com/api.php"

# Reuse a single opener with a browser-like User-Agent to avoid 403s
_opener = urllib.request.build_opener()
_opener.addheaders = [("User-Agent", "MangaAnimePipeline/1.0")]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _api_get(params):
    """Make a GET request to the MediaWiki API and return parsed JSON."""
    params["format"] = "json"
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    with _opener.open(url, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_section_wikitext(page, section_index):
    """Return raw wikitext for a specific section of a page."""
    data = _api_get({
        "action": "parse",
        "page": page,
        "prop": "wikitext",
        "section": str(section_index),
    })
    return data["parse"]["wikitext"]["*"]


def _get_section_index(page, heading):
    """Find the section index for a given heading on a page."""
    data = _api_get({
        "action": "parse",
        "page": page,
        "prop": "sections",
    })
    for sec in data["parse"]["sections"]:
        if sec["line"].strip().lower() == heading.strip().lower():
            return int(sec["index"])
    return None


def _get_rendered_html(page, section=None):
    """Return rendered HTML for a page (or a section of it)."""
    params = {"action": "parse", "page": page, "prop": "text"}
    if section is not None:
        params["section"] = str(section)
    data = _api_get(params)
    return data["parse"]["text"]["*"]


def _strip_wikitext_markup(text):
    """Remove common wikitext markup, returning plain text."""
    # Remove [[File:...|...]] image embeds (must be before link stripping)
    text = re.sub(r"\[\[File:[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    # Remove [[ ]] links, keeping display text
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    # Remove '' (italic) and ''' (bold)
    text = re.sub(r"'{2,3}", "", text)
    # Remove {{ }} templates (simple, non-nested)
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    # Remove <ref>...</ref>
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/?>", "", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


class _InfoboxParser(HTMLParser):
    """Parse the portable-infobox <aside> from rendered HTML into a dict."""

    def __init__(self):
        super().__init__()
        self.fields = {}
        self.image_url = None
        self._in_aside = False
        self._in_label = False
        self._in_value = False
        self._current_label = ""
        self._current_value = ""
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "")

        if tag == "aside" and "portable-infobox" in cls:
            self._in_aside = True
            self._depth = 0

        if not self._in_aside:
            return

        if tag == "aside":
            self._depth += 1

        # Capture the first infobox image
        if tag == "img" and self.image_url is None and "pi-image-thumbnail" in cls:
            self.image_url = attrs_dict.get("src", "")

        if "pi-data-label" in cls or "pi-header" in cls:
            self._in_label = True
            self._current_label = ""

        if "pi-data-value" in cls or "pi-font" in cls:
            self._in_value = True
            self._current_value = ""

        # Treat <br> and <li> as separators
        if tag in ("br", "li") and self._in_value:
            self._current_value += "; "

        # Capture linked images as fallback
        if tag == "img" and self.image_url is None and self._in_aside:
            src = attrs_dict.get("src", "")
            if src and "static.wikia" in src:
                self.image_url = src

    def handle_endtag(self, tag):
        if not self._in_aside:
            return

        if tag == "aside":
            self._depth -= 1
            if self._depth <= 0:
                self._in_aside = False

        if self._in_label and tag in ("h3", "div", "span"):
            self._in_label = False

        if self._in_value and tag in ("div", "td", "span"):
            label = self._current_label.strip().rstrip(":")
            value = self._current_value.strip().rstrip(";").strip()
            if label and value:
                self.fields[label] = value
            self._in_value = False

    def handle_data(self, data):
        if self._in_label:
            self._current_label += data
        if self._in_value:
            self._current_value += data

    def get_clean_fields(self):
        """Return fields with wiki reference noise cleaned out."""
        cleaned = {}
        for label, value in self.fields.items():
            # Remove [1], [2], etc. reference markers
            value = re.sub(r"\[[\d]+\]", "", value)
            # Remove orphaned [ from partial refs
            value = re.sub(r"\[\s*$", "", value)
            value = re.sub(r"\[\s*;", ";", value)
            # Clean up double semicolons and trailing separators
            value = re.sub(r";\s*;", ";", value)
            value = value.strip().rstrip(";").strip()
            if value:
                cleaned[label] = value
        return cleaned


# ---------------------------------------------------------------------------
# Episode-to-Chapter mapping (source: listfist.com)
# Empty list = filler / anime-original episode
# ---------------------------------------------------------------------------

_EPISODE_TO_CHAPTERS = {
    1: [2],
    2: [3, 4, 5],
    3: [5, 6, 7],
    4: [1, 8],
    5: [8, 9, 10, 11],
    6: [11, 12, 13, 14, 15],
    7: [15, 16, 17, 18],
    8: [19, 20, 21],
    9: [23, 24],
    10: [24, 25, 26],
    11: [26, 27],
    12: [28, 29, 30],
    13: [30, 31, 32],
    14: [33, 34],
    15: [35, 36, 37],
    16: [38, 39],
    17: [39, 40, 41],
    18: [22, 42],
    19: [5, 42],
    20: [42, 43],
    21: [44, 45],
    22: [46, 47],
    23: [48, 49],
    24: [50, 51, 52],
    25: [53, 54, 55],
    26: [56, 57, 58, 59],
    27: [59, 60, 61, 62],
    28: [62, 63, 64],
    29: [65, 66],
    30: [67, 68],
    31: [69, 70, 71],
    32: [70, 71, 72, 73],
    33: [73, 74, 75],
    34: [75, 76, 77],
    35: [77, 78],
    36: [78, 79, 80],
    37: [80, 81],
    38: [82, 83, 84],
    39: [84, 85, 86],
    40: [86, 87, 88],
    41: [88, 89, 90],
    42: [90, 91, 92],
    43: [92, 93, 94],
    44: [95],
    45: [96],
    46: [35, 37, 39, 40, 42, 43, 46, 47, 48, 50, 51, 53, 54],
    47: [36, 55, 57, 58, 59, 60, 62, 63, 65, 66, 67, 68, 71, 72, 73, 74, 75],
    48: [96, 97, 98],
    49: [97],
    50: [],
    51: [],
    52: [98, 99],
    53: [100],
    54: [101],
    55: [101],
    56: [],
    57: [],
    58: [],
    59: [],
    60: [],
    61: [101],
    62: [102, 103],
    63: [103, 104, 105],
    64: [106, 107],
    65: [107, 108, 109, 110],
    66: [110, 111, 112],
    67: [113, 114],
    68: [83, 85, 87, 93, 95, 99, 101, 102, 104, 106, 108, 110, 112, 113],
    69: [114, 116, 118, 119],
    70: [115, 116, 117],
    71: [116, 118],
    72: [117, 118, 119],
    73: [119, 120],
    74: [121, 122],
    75: [123, 124],
    76: [124, 125, 126],
    77: [127, 128, 129],
    78: [129, 130, 131],
    79: [128, 131, 132],
    80: [132, 133, 134],
    81: [134, 135, 136],
    82: [136, 137, 138],
    83: [138, 139],
    84: [139, 140],
    85: [140, 141, 142, 143],
    86: [141, 143, 144, 145],
    87: [145, 146, 147, 148],
    88: [147, 148, 149, 150],
    89: [150, 151, 152],
    90: [152, 153],
    91: [142, 154, 155],
    92: [155, 156, 157],
    93: [158],
    94: [157, 158],
    95: [159],
    96: [159, 160, 161],
    97: [162],
    98: [162],
    99: [],
    100: [163, 164],
    101: [159],
    102: [],
    103: [160, 161, 162, 163, 164, 165],
    104: [165, 166],
    105: [167, 168],
    106: [169, 170],
    107: [171, 172],
    108: [172, 173, 174],
    109: [175, 176],
    110: [176, 177, 178],
    111: [179, 180],
    112: [180, 181, 182, 183],
    113: [182, 183, 184],
    114: [184, 185, 186],
    115: [186, 187, 188],
    116: [188, 189, 190],
    117: [190, 191, 192],
    118: [192, 193],
    119: [194, 195],
    120: [196, 197],
    121: [198, 199],
    122: [200, 201],
    123: [202, 203],
    124: [204, 205],
    125: [206, 207, 208],
    126: [208, 209, 210],
    127: [211, 212],
    128: [213, 214],
    129: [215, 216],
    130: [217, 218],
    131: [],
    132: [],
    133: [],
    134: [],
    135: [],
    136: [],
    137: [],
    138: [],
    139: [],
    140: [],
    141: [],
    142: [],
    143: [218],
    144: [218, 219, 220],
    145: [221],
    146: [222, 223, 224],
    147: [225, 226],
    148: [227, 228],
    149: [229, 230],
    150: [231, 232],
    151: [232, 233, 234],
    152: [235, 236],
    153: [237, 238],
    154: [239, 240],
    155: [241],
    156: [242],
    157: [243],
    158: [244],
    159: [245],
    160: [246, 247],
    161: [247],
    162: [248, 249],
    163: [249, 250],
    164: [251],
    165: [252],
    166: [253],
    167: [254],
    168: [255, 256],
    169: [256, 257, 258],
    170: [258, 259],
    171: [260, 261],
    172: [261, 262, 263],
    173: [263, 264],
    174: [263, 264, 265, 266, 268, 272],
    175: [266, 267],
    176: [15, 268, 269],
    177: [270, 271],
    178: [271, 272],
    179: [272, 273],
    180: [274, 275],
    181: [275, 276, 277],
    182: [276, 277, 278, 279],
    183: [280, 281],
    184: [282, 283],
    185: [283, 284],
    186: [284, 285, 286],
    187: [286, 287, 288],
    188: [289, 290, 291],
    189: [291, 292, 293],
    190: [294, 295],
    191: [295, 296, 297],
    192: [297, 298],
    193: [299, 300],
    194: [301],
    195: [302, 303],
    196: [],
    197: [],
    198: [],
    199: [],
    200: [],
    201: [],
    202: [],
    203: [303],
    204: [],
    205: [],
    206: [],
    207: [303, 304, 305],
    208: [305, 306],
    209: [306, 307],
    210: [308, 309],
    211: [310, 311],
    212: [312, 313],
    213: [],
    214: [313],
    215: [],
    216: [313],
    217: [313, 314],
    218: [315, 316],
    219: [317, 318],
    220: [],
    221: [],
    222: [],
    223: [],
    224: [],
    225: [303],
    226: [318],
    227: [319, 320],
    228: [320, 321, 322],
    229: [322, 323, 324],
    230: [323, 324, 325],
    231: [325, 326],
    232: [326, 327],
    233: [328, 329],
    234: [329, 330, 331],
    235: [331, 332],
    236: [332, 333],
    237: [334, 335],
    238: [335, 336, 337],
    239: [337, 338, 339],
    240: [339, 340],
    241: [340, 341, 342],
    242: [340, 341, 342],
    243: [343, 344, 345],
    244: [346],
    245: [347, 348],
    246: [349, 350],
    247: [350, 351, 352],
    248: [352, 353, 354],
    249: [353, 355, 356],
    250: [356, 357, 358],
    251: [358, 359],
    252: [360, 361],
    253: [361, 362],
    254: [363, 364, 365],
    255: [364, 365],
    256: [358, 366],
    257: [366, 367],
    258: [368, 369],
    259: [369, 370],
    260: [370, 371, 372],
    261: [371, 372, 373],
    262: [373, 374],
    263: [374, 375],
    264: [375, 376],
    265: [376, 377, 388],
    266: [378, 379],
    267: [379, 380],
    268: [381, 382],
    269: [382, 383],
    270: [383, 384],
    271: [384, 385],
    272: [385, 386, 387],
    273: [387, 388],
    274: [389, 390],
    275: [391, 392],
    276: [393, 394, 395],
    277: [395, 396, 397],
    278: [397, 398],
    279: [],
    280: [],
    281: [],
    282: [],
    283: [],
    284: [399, 400],
    285: [400],
    286: [401, 402],
    287: [402, 403],
    288: [403, 404],
    289: [405, 406],
    290: [406, 407],
    291: [],
    292: [],
    293: [407, 408],
    294: [409],
    295: [410, 411],
    296: [411, 412],
    297: [413, 414],
    298: [414, 415, 416],
    299: [416, 417],
    300: [417, 418],
    301: [419, 420],
    302: [420],
    303: [],
    304: [421],
    305: [422],
    306: [423, 424],
    307: [424, 425],
    308: [425, 426],
    309: [426, 427],
    310: [428],
    311: [429],
    312: [430],
    313: [431],
    314: [432],
    315: [433],
    316: [433, 434, 435],
    317: [],
    318: [],
    319: [435],
    320: [358, 435],
    321: [436],
    322: [437],
    323: [438],
    324: [439, 440],
    325: [434, 440, 441],
    326: [442],
    327: [],
    328: [],
    329: [],
    330: [],
    331: [],
    332: [],
    333: [],
    334: [],
    335: [],
    336: [],
    337: [442],
    338: [443],
    339: [443, 444],
    340: [445, 446],
    341: [446, 447],
    342: [447, 448, 449],
    343: [448, 449],
    344: [449, 450],
    345: [451],
    346: [452],
    347: [453],
    348: [454],
    349: [455],
    350: [456],
    351: [457],
    352: [458],
    353: [458, 459],
    354: [459],
    355: [460],
    356: [461],
    357: [462],
    358: [463, 464],
    359: [463, 464],
    360: [465],
    361: [466],
    362: [466, 467],
    363: [468],
    364: [469],
    365: [470],
    366: [471],
    367: [472],
    368: [473],
    369: [474],
    370: [475, 476],
    371: [476, 477],
    372: [478, 479],
    373: [480, 481],
    374: [481, 482],
    375: [483],
    376: [484],
    377: [485],
    378: [486],
    379: [487],
    380: [487, 488],
    381: [489, 490],
    382: [],
    383: [],
    384: [],
    385: [490],
    386: [490, 491],
    387: [492, 493],
    388: [493, 494],
    389: [494, 495],
    390: [496],
    391: [497],
    392: [498],
    393: [499],
    394: [500],
    395: [501],
    396: [502],
    397: [503],
    398: [504],
    399: [505, 506],
    400: [506, 507],
    401: [507, 508],
    402: [509, 510],
    403: [510, 511],
    404: [511, 512],
    405: [513],
    406: [],
    407: [],
    408: [514],
    409: [515],
    410: [516],
    411: [517],
    412: [518],
    413: [519],
    414: [520],
    415: [521],
    416: [522],
    417: [522, 523],
    418: [523],
    419: [523, 524],
    420: [524],
    421: [524],
    422: [525, 526],
    423: [526],
    424: [527],
    425: [528, 530],
    426: [],
    427: [],
    428: [],
    429: [],
    430: [528, 529],
    431: [530],
    432: [530, 531],
    433: [532],
    434: [533],
    435: [533, 534],
    436: [535],
    437: [535, 536],
    438: [536, 537],
    439: [537, 538],
    440: [538, 539],
    441: [539],
    442: [540],
    443: [540, 541],
    444: [541, 542],
    445: [542, 543],
    446: [543, 544],
    447: [544],
    448: [545],
    449: [545, 546],
    450: [546, 547],
    451: [548],
    452: [549],
    453: [548, 549, 555, 556],
    454: [543, 544, 552, 554],
    455: [545, 546, 550, 551],
    456: [557, 558, 559, 560],
    457: [],
    458: [],
    459: [549, 550, 551],
    460: [551],
    461: [551, 552],
    462: [552, 553],
    463: [553, 554],
    464: [554, 555],
    465: [556],
    466: [557],
    467: [558],
    468: [559],
    469: [560],
    470: [561],
    471: [562],
    472: [563],
    473: [564],
    474: [565],
    475: [566],
    476: [567],
    477: [568],
    478: [569],
    479: [570, 571],
    480: [571],
    481: [572],
    482: [573],
    483: [574],
    484: [574, 575, 576],
    485: [576],
    486: [577],
    487: [578],
    488: [579],
    489: [580],
    490: [581],
    491: [581, 582],
    492: [],
    493: [582, 583],
    494: [583, 584],
    495: [584],
    496: [585],
    497: [585],
    498: [584, 585],
    499: [],
    500: [585, 586],
    501: [586, 587],
    502: [587, 588],
    503: [588, 589],
    504: [1, 589],
    505: [590],
    506: [],
    507: [591],
    508: [592],
    509: [592],
    510: [591, 593],
    511: [594],
    512: [593, 594],
    513: [595],
    514: [595, 596],
    515: [596, 597],
    516: [597],
    517: [598],
    518: [598, 599],
    519: [599, 600],
    520: [600],
    521: [601],
    522: [602],
    523: [603, 604],
    524: [604, 605],
    525: [605, 606],
    526: [606, 607],
    527: [607, 608],
    528: [609],
    529: [610],
    530: [611],
    531: [612],
    532: [613],
    533: [614],
    534: [614, 615],
    535: [616],
    536: [617],
    537: [618],
    538: [619],
    539: [620],
    540: [621, 622],
    541: [620, 621, 622],
    542: [],
    543: [623],
    544: [623, 624],
    545: [624, 625],
    546: [625, 626],
    547: [627],
    548: [628],
    549: [629],
    550: [629, 630],
    551: [631],
    552: [632],
    553: [633],
    554: [634],
    555: [634, 635],
    556: [634, 635, 636],
    557: [636],
    558: [636, 637],
    559: [638],
    560: [639],
    561: [640],
    562: [641],
    563: [642, 643, 644],
    564: [642, 644],
    565: [643, 644, 645],
    566: [645, 646],
    567: [645, 647],
    568: [648],
    569: [649],
    570: [650],
    571: [651],
    572: [652],
    573: [653],
    574: [654, 655],
    575: [],
    576: [],
    577: [],
    578: [],
    579: [655],
    580: [655, 656],
    581: [656, 657, 658],
    582: [657, 658],
    583: [658, 659],
    584: [659],
    585: [660],
    586: [661],
    587: [662, 663],
    588: [662, 663],
    589: [664],
    590: [],
    591: [665],
    592: [666],
    593: [667],
    594: [668],
    595: [669],
    596: [670],
    597: [671, 672],
    598: [672],
    599: [673],
    600: [674],
    601: [675],
    602: [676],
    603: [677],
    604: [678, 679],
    605: [679],
    606: [680],
    607: [681],
    608: [682],
    609: [683],
    610: [684],
    611: [685],
    612: [686],
    613: [687],
    614: [688],
    615: [689],
    616: [690, 693],
    617: [691, 692],
    618: [692],
    619: [693, 694],
    620: [694],
    621: [695],
    622: [696],
    623: [697],
    624: [698],
    625: [698, 699],
    626: [699],
    627: [],
    628: [],
    629: [],
    630: [701],
    631: [701, 702],
    632: [702, 703],
    633: [703, 704],
    634: [704, 705, 706],
    635: [705, 706],
    636: [706, 707],
    637: [707, 708],
    638: [709],
    639: [710],
    640: [710, 711, 712],
    641: [711, 712],
    642: [706, 711, 712],
    643: [713, 714],
    644: [714],
    645: [715],
    646: [716],
    647: [717],
    648: [718, 726],
    649: [718, 719],
    650: [720],
    651: [721],
    652: [722],
    653: [722, 723],
    654: [706, 723],
    655: [724],
    656: [725],
    657: [725, 726, 727, 731],
    658: [726],
    659: [727],
    660: [728],
    661: [728, 729],
    662: [730],
    663: [731],
    664: [732],
    665: [733],
    666: [734],
    667: [735],
    668: [736],
    669: [736, 737],
    670: [736, 737, 738],
    671: [738, 739],
    672: [739, 740],
    673: [740],
    674: [741],
    675: [741, 742],
    676: [742],
    677: [743],
    678: [744],
    679: [744, 745],
    680: [745],
    681: [746],
    682: [746, 747],
    683: [747],
    684: [748],
    685: [749, 750],
    686: [749, 750],
    687: [750, 751],
    688: [751, 752],
    689: [752],
    690: [752, 753],
    691: [753, 754, 755],
    692: [754, 755],
    693: [755],
    694: [756],
    695: [756, 757],
    696: [757, 758],
    697: [758],
    698: [759],
    699: [760],
    700: [761],
    701: [762],
    702: [763],
    703: [764],
    704: [765],
    705: [766],
    706: [767],
    707: [768],
    708: [769, 770],
    709: [769, 770],
    710: [769, 771],
    711: [769, 770, 771, 772, 780],
    712: [772, 773],
    713: [773],
    714: [774, 779],
    715: [775, 779],
    716: [775, 776],
    717: [776, 777],
    718: [777, 778],
    719: [778],
    720: [779],
    721: [780],
    722: [781],
    723: [781, 782],
    724: [782, 783],
    725: [783],
    726: [784],
    727: [785],
    728: [785, 786],
    729: [786, 787],
    730: [787, 788],
    731: [788, 789],
    732: [789],
    733: [790],
    734: [791],
    735: [792],
    736: [793],
    737: [794],
    738: [794, 795],
    739: [795],
    740: [796],
    741: [796, 797],
    742: [797, 798],
    743: [798, 799],
    744: [799],
    745: [800],
    746: [801],
    747: [802],
    748: [],
    749: [],
    750: [],
    751: [802],
    752: [802, 803],
    753: [803, 804],
    754: [805],
    755: [806],
    756: [807],
    757: [807, 808],
    758: [808, 809],
    759: [809, 810],
    760: [808, 809, 810],
    761: [810],
    762: [811, 812],
    763: [812, 813],
    764: [813, 814],
    765: [814, 815],
    766: [815],
    767: [816],
    768: [817],
    769: [817, 818],
    770: [818],
    771: [819],
    772: [819, 820],
    773: [820, 821],
    774: [821],
    775: [],
    776: [822],
    777: [823],
    778: [823],
    779: [824],
    780: [],
    781: [],
    782: [],
    783: [824, 825],
    784: [825, 826],
    785: [826, 827],
    786: [827],
    787: [828],
    788: [828, 829],
    789: [829],
    790: [830],
    791: [831],
    792: [831, 832],
    793: [832, 833],
    794: [833, 834],
    795: [834],
    796: [835],
    797: [835, 836],
    798: [836, 837],
    799: [837, 838],
    800: [838, 839],
    801: [839],
    802: [839, 840],
    803: [840],
    804: [841],
    805: [841, 842],
    806: [842, 843],
    807: [843],
    808: [843, 844],
    809: [845],
    810: [845],
    811: [846],
    812: [846, 847],
    813: [847, 848],
    814: [848, 849],
    815: [848, 849],
    816: [850],
    817: [850, 851],
    818: [851],
    819: [852],
    820: [852, 853],
    821: [853, 854],
    822: [854],
    823: [855],
    824: [855, 856],
    825: [856],
    826: [857],
    827: [857, 858],
    828: [858, 859],
    829: [859, 860],
    830: [860],
    831: [861],
    832: [862],
    833: [863],
    834: [864],
    835: [865],
    836: [866],
    837: [867],
    838: [868],
    839: [868, 869],
    840: [869, 870],
    841: [870, 871],
    842: [871, 872],
    843: [872, 873],
    844: [873, 874],
    845: [874],
    846: [875],
    847: [875, 876],
    848: [876, 877],
    849: [877, 878],
    850: [878],
    851: [879],
    852: [879, 880],
    853: [880, 881],
    854: [881, 882],
    855: [882, 883],
    856: [883],
    857: [884],
    858: [884, 885],
    859: [885, 886],
    860: [886, 887],
    861: [887, 888],
    862: [888, 889],
    863: [889, 890],
    864: [890],
    865: [890, 891, 894],
    866: [891, 892],
    867: [892, 893],
    868: [893, 894],
    869: [894],
    870: [895],
    871: [896, 902],
    872: [897],
    873: [898],
    874: [899],
    875: [900],
    876: [901],
    877: [902],
    878: [903],
    879: [903],
    880: [904],
    881: [905],
    882: [905],
    883: [905],
    884: [906],
    885: [864, 865, 866, 868, 870, 871, 888, 903, 904, 906],
    886: [907],
    887: [907],
    888: [908],
    889: [908],
    890: [909],
    891: [910],
    892: [909, 910],
    893: [910, 911],
    894: [911, 912],
    895: [],
    896: [],
    897: [912],
    898: [913],
    899: [913, 915],
    900: [914],
    901: [914, 915],
    902: [915, 916],
    903: [916],
    904: [916, 917],
    905: [917],
    906: [918],
    907: [],
    908: [918, 919],
    909: [919],
    910: [920],
    911: [920, 921],
    912: [921, 922],
    913: [922],
    914: [922, 923],
    915: [923, 924],
    916: [924],
    917: [925],
    918: [925, 926],
    919: [926],
    920: [927, 928],
    921: [927, 928],
    922: [928, 929],
    923: [928, 929, 930],
    924: [930],
    925: [931],
    926: [931, 932],
    927: [932, 933],
    928: [933],
    929: [934],
    930: [934, 935],
    931: [935, 936],
    932: [936],
    933: [937],
    934: [937, 938],
    935: [938, 939],
    936: [939, 940],
    937: [940],
    938: [941],
    939: [941, 942],
    940: [942, 943],
    941: [943],
    942: [944],
    943: [944, 945],
    944: [945],
    945: [946],
    946: [947],
    947: [947, 948],
    948: [948, 949],
    949: [949],
    950: [950],
    951: [950, 951],
    952: [951, 952],
    953: [952, 953],
    954: [953],
    955: [954],
    956: [955],
    957: [956],
    958: [957],
    959: [958, 959],
    960: [959, 960],
    961: [961, 962],
    962: [962, 963],
    963: [963, 964],
    964: [964, 965],
    965: [965, 966],
    966: [966],
    967: [966, 967],
    968: [967],
    969: [968],
    970: [968, 969],
    971: [969, 970],
    972: [970, 971],
    973: [971],
    974: [972],
    975: [972, 973],
    976: [973, 974],
    977: [974],
    978: [975],
    979: [975, 976],
    980: [976],
    981: [977],
    982: [977, 978],
    983: [977, 978],
    984: [979],
    985: [979, 980],
    986: [980],
    987: [981],
    988: [981, 982],
    989: [982],
    990: [983],
    991: [983, 984],
    992: [984],
    993: [985],
    994: [985, 986],
    995: [986],
    996: [987],
    997: [987, 988],
    998: [988],
    999: [989],
    1000: [989, 990],
    1001: [990],
    1002: [991],
    1003: [991, 992],
    1004: [992],
    1005: [993],
    1006: [993, 994],
    1007: [994],
    1008: [995],
    1009: [995, 996],
    1010: [996, 997],
    1011: [997],
    1012: [998],
    1013: [998, 999],
    1014: [999, 1000],
    1015: [1000],
    1016: [1001],
    1017: [1001, 1002],
    1018: [1003],
    1019: [1004],
    1020: [1004, 1005],
    1021: [1005, 1006],
    1022: [1006],
    1023: [1007],
    1024: [1007, 1008],
    1025: [1008, 1009],
    1026: [1009],
    1027: [1010],
    1028: [1010, 1011],
    1029: [],
    1030: [],
    1031: [1011],
    1032: [1012],
    1033: [1012, 1013],
    1034: [1013, 1014],
    1035: [1014],
    1036: [1015],
    1037: [1015, 1016],
    1038: [1016, 1017],
    1039: [1017, 1018],
    1040: [1018],
    1041: [1019],
    1042: [1019, 1020],
    1043: [1020, 1021],
    1044: [1021],
    1045: [1022],
    1046: [1022, 1023],
    1047: [1023, 1024],
    1048: [1024],
    1049: [1025],
    1050: [1025, 1026],
    1051: [1026, 1027],
    1052: [1027, 1028],
    1053: [1028],
    1054: [1029],
    1055: [1029, 1030],
    1056: [1030, 1031],
    1057: [1031],
    1058: [1032],
    1059: [1032, 1033],
    1060: [1033, 1034],
    1061: [1034, 1035],
    1062: [1035, 1036],
    1063: [1036],
    1064: [1037],
    1065: [1038],
    1066: [1039],
    1067: [1040],
    1068: [1041],
    1069: [1041, 1042],
    1070: [1043],
    1071: [1044],
    1072: [1045, 1046],
    1073: [1046],
    1074: [1047],
    1075: [1048],
    1076: [1049],
    1077: [1050],
    1078: [1051],
    1079: [1052],
    1080: [1053],
    1081: [873, 875, 1054],
    1082: [1055],
    1083: [1056],
    1084: [],
    1085: [1057],
    1086: [1058],
    1087: [1059],
    1088: [1058, 1059, 1060],
    1089: [1060, 1061],
    1090: [1061],
    1091: [1062],
    1092: [1062, 1063],
    1093: [1062, 1063, 1064],
    1094: [1064, 1065],
    1095: [1065],
    1096: [1066],
    1097: [1066, 1067],
    1098: [1067, 1068],
    1099: [1068, 1069],
    1100: [1069],
    1101: [1070],
    1102: [1070, 1071],
    1103: [1071, 1072],
    1104: [1072, 1073],
    1105: [1073],
    1106: [1074],
    1107: [1074, 1075],
    1108: [1075, 1076, 1077],
    1109: [1076],
    1110: [1077],
    1111: [1078],
    1112: [1079],
    1113: [1080],
    1114: [1073, 1080, 1081],
    1115: [1081],
    1116: [1082],
    1117: [1083],
    1118: [1084],
    1119: [1085],
    1120: [1086],
    1121: [1087],
}

# Reverse mapping: chapter -> episodes (built lazily)
_CHAPTER_TO_EPISODES = None


def _build_chapter_to_episodes():
    global _CHAPTER_TO_EPISODES
    if _CHAPTER_TO_EPISODES is not None:
        return
    _CHAPTER_TO_EPISODES = {}
    for ep, chapters in _EPISODE_TO_CHAPTERS.items():
        for ch in chapters:
            _CHAPTER_TO_EPISODES.setdefault(ch, []).append(ep)
    for ch in _CHAPTER_TO_EPISODES:
        _CHAPTER_TO_EPISODES[ch].sort()


# ---------------------------------------------------------------------------
# Public API -- Episodes
# ---------------------------------------------------------------------------

def get_episode_characters(episode_num):
    """
    Get the list of characters appearing in an episode.

    Returns:
        list of dicts: [{"name": str, "wiki_page": str, "annotation": str|None}, ...]
    """
    page = f"Episode_{episode_num}"
    idx = _get_section_index(page, "Characters in Order of Appearance")
    if idx is None:
        return []

    wikitext = _get_section_wikitext(page, idx)
    characters = []
    for line in wikitext.split("\n"):
        line = line.strip()
        if not line.startswith("*"):
            continue
        line = line.lstrip("*").strip()

        # Extract annotation like ''(flashback)''
        annotation = None
        ann_match = re.search(r"''\(([^)]+)\)''", line)
        if ann_match:
            annotation = ann_match.group(1)
            line = line[:ann_match.start()].strip()

        # Extract wiki link [[Page|Display]] or [[Page]]
        link_match = re.search(r"\[\[([^|\]]+)(?:\|([^\]]+))?\]\]", line)
        if link_match:
            wiki_page = link_match.group(1).strip()
            display = (link_match.group(2) or wiki_page).strip()
            characters.append({
                "name": display,
                "wiki_page": wiki_page,
                "annotation": annotation,
            })

    return characters


def get_episode_summary(episode_num):
    """
    Get the long summary for an episode.

    Returns:
        str: Plain-text episode summary.
    """
    page = f"Episode_{episode_num}"
    idx = _get_section_index(page, "Long Summary")
    if idx is None:
        return ""

    wikitext = _get_section_wikitext(page, idx)
    # Remove the heading line
    lines = wikitext.split("\n")
    lines = [l for l in lines if not l.startswith("==")]
    text = "\n".join(lines)
    return _strip_wikitext_markup(text)


def get_episode_short_summary(episode_num):
    """
    Get the short summary for an episode.

    Returns:
        str: Plain-text short summary.
    """
    page = f"Episode_{episode_num}"
    idx = _get_section_index(page, "Short Summary")
    if idx is None:
        return ""

    wikitext = _get_section_wikitext(page, idx)
    lines = wikitext.split("\n")
    lines = [l for l in lines if not l.startswith("==")]
    text = "\n".join(lines)
    return _strip_wikitext_markup(text)


def get_episode_infobox(episode_num):
    """
    Get episode metadata from the infobox (section 0 wikitext).

    Returns:
        dict with keys like: title, translation, kanji, romaji, airdate,
        char_debuts, location_debuts, tech_debuts, devil_fruit_debuts, etc.
    """
    page = f"Episode_{episode_num}"
    wikitext = _get_section_wikitext(page, 0)

    result = {"episode": episode_num}
    # Parse pipe-delimited fields from the Episode Box template
    for match in re.finditer(r"\|\s*(\w[\w\s]*?)\s*=\s*(.+)", wikitext):
        key = match.group(1).strip().lower().replace(" ", "_")
        value = match.group(2).strip()
        if value:
            result[key] = _strip_wikitext_markup(value)

    return result



def get_episode_chapters(episode_num):
    """
    Get the manga chapters adapted by an anime episode.

    Uses a hardcoded mapping table sourced from listfist.com.
    Covers episodes 1-1121. Returns None for unknown episodes.

    Args:
        episode_num: Episode number (int)

    Returns:
        dict: {"episode": int, "chapters": [int, ...], "is_filler": bool}
        or None if the episode is not in the mapping.
    """
    chapters = _EPISODE_TO_CHAPTERS.get(episode_num)
    if chapters is None:
        return None
    return {
        "episode": episode_num,
        "chapters": list(chapters),
        "is_filler": len(chapters) == 0,
    }


def get_chapter_episodes(chapter_num):
    """
    Get the anime episodes that adapt a manga chapter (reverse lookup).

    Args:
        chapter_num: Chapter number (int)

    Returns:
        dict: {"chapter": int, "episodes": [int, ...]}
        or None if the chapter is not covered by any episode.
    """
    _build_chapter_to_episodes()
    episodes = _CHAPTER_TO_EPISODES.get(chapter_num)
    if episodes is None:
        return None
    return {
        "chapter": chapter_num,
        "episodes": list(episodes),
    }


# ---------------------------------------------------------------------------
# Public API -- Characters
# ---------------------------------------------------------------------------

def get_character_info(name):
    """
    Get structured character info from the rendered infobox.

    Args:
        name: Character name as it appears in wiki URLs
              (e.g. "Monkey D. Luffy", "Roronoa Zoro", "Koby")

    Returns:
        dict with parsed infobox fields + image_url.
        Field names are lowercased with spaces replaced by underscores.
    """
    wiki_page = name.replace(" ", "_")

    # Get rendered HTML (section 0) and parse the portable-infobox
    html = _get_rendered_html(wiki_page, section=0)
    parser = _InfoboxParser()
    parser.feed(html)

    result = {"name": name, "wiki_page": wiki_page}

    if parser.image_url:
        result["image_url"] = parser.image_url

    for label, value in parser.get_clean_fields().items():
        key = label.lower().replace(" ", "_").replace(":", "")
        result[key] = value

    return result


def get_character_image(name, thumb_size=300):
    """
    Get just the character's thumbnail image URL.

    Args:
        name: Character name (e.g. "Monkey D. Luffy")
        thumb_size: Thumbnail width in pixels (default 300)

    Returns:
        str: Image URL, or None if not found.
    """
    wiki_page = name.replace(" ", "_")
    data = _api_get({
        "action": "query",
        "titles": wiki_page,
        "prop": "pageimages",
        "pithumbsize": str(thumb_size),
    })
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        thumb = page.get("thumbnail", {})
        if thumb.get("source"):
            return thumb["source"]
    return None


# ---------------------------------------------------------------------------
# Public API -- Arcs
# ---------------------------------------------------------------------------

def _arc_page(arc_name):
    """Normalize arc name to wiki page title (e.g. 'Romance Dawn' -> 'Romance_Dawn_Arc')."""
    name = arc_name.strip()
    if not name.endswith("Arc"):
        name = name + " Arc"
    return name.replace(" ", "_")


def get_arc_summary(arc_name):
    """
    Get the arc's narrative summary with named sub-sections.

    Args:
        arc_name: e.g. "Romance Dawn Arc" or "Romance Dawn"

    Returns:
        list of dicts: [{"heading": str, "text": str}, ...]
        Each entry is a named story beat within the arc.
    """
    page = _arc_page(arc_name)
    # Summary is always section 1
    idx = _get_section_index(page, "Summary")
    if idx is None:
        return []

    # Get the sections list to find sub-sections under Summary
    data = _api_get({"action": "parse", "page": page, "prop": "sections"})
    sections = data["parse"]["sections"]

    # Find all sub-sections (level 3) under Summary
    summary_parts = []
    in_summary = False
    for sec in sections:
        if sec["line"].strip().lower() == "summary":
            in_summary = True
            continue
        if in_summary:
            if int(sec["level"]) <= 2:
                break  # hit next top-level section
            if int(sec["level"]) == 3:
                wikitext = _get_section_wikitext(page, int(sec["index"]))
                # Remove the heading line
                lines = wikitext.split("\n")
                lines = [l for l in lines if not l.startswith("==")]
                text = _strip_wikitext_markup("\n".join(lines))
                summary_parts.append({
                    "heading": sec["line"].strip(),
                    "text": text,
                })

    # If no sub-sections, get the whole Summary section
    if not summary_parts:
        wikitext = _get_section_wikitext(page, idx)
        lines = wikitext.split("\n")
        lines = [l for l in lines if not l.startswith("==")]
        text = _strip_wikitext_markup("\n".join(lines))
        if text:
            summary_parts.append({"heading": "Summary", "text": text})

    return summary_parts


def get_arc_ranges(arc_name):
    """
    Get the arc's chapter/episode/volume ranges from the infobox.

    Args:
        arc_name: e.g. "Romance Dawn Arc" or "Romance Dawn"

    Returns:
        dict with keys: arc, chapters, episodes, volumes, year,
        prev_arc, next_arc.  Range values are dicts with 'start',
        'end', 'count' where parseable.
    """
    page = _arc_page(arc_name)
    html = _get_rendered_html(page, section=0)
    parser = _InfoboxParser()
    parser.feed(html)

    fields = parser.get_clean_fields()
    result = {"arc": arc_name}

    def _parse_range(raw):
        """Parse '1-7, 7 chapters' or '1-4, 4 episodes' into structured dict."""
        if not raw:
            return None
        info = {"raw": raw}
        # Match patterns like "1-7" or "1-4"
        range_match = re.search(r"(\d+)\s*-\s*(\d+)", raw)
        if range_match:
            info["start"] = int(range_match.group(1))
            info["end"] = int(range_match.group(2))
        else:
            # Single number like "1"
            single = re.search(r"(\d+)", raw)
            if single:
                info["start"] = int(single.group(1))
                info["end"] = int(single.group(1))
        # Match count like "7 chapters" or "4 episodes"
        count_match = re.search(r"(\d+)\s+(?:chapter|episode|volume)", raw, re.I)
        if count_match:
            info["count"] = int(count_match.group(1))
        return info

    for label, value in fields.items():
        label_lower = label.lower()
        if "chapter" in label_lower:
            result["chapters"] = _parse_range(value)
        elif "episode" in label_lower:
            result["episodes"] = _parse_range(value)
        elif "volume" in label_lower:
            result["volumes"] = _parse_range(value)
        elif "year" in label_lower or "released" in label_lower:
            result["year"] = value
        elif "previous" in label_lower or "prev" in label_lower:
            result["prev_arc"] = _strip_wikitext_markup(value)
        elif "next" in label_lower:
            result["next_arc"] = _strip_wikitext_markup(value)

    return result



def get_story_arcs():
    """
    Get the full saga -> arc -> episode/chapter mapping from the Story Arcs page.

    Returns:
        list of dicts: [
            {"saga": "East Blue Saga", "arcs": [
                {"name": "Romance Dawn Arc", "wiki_page": "Romance_Dawn_Arc",
                 "chapters": "1-7", "episodes": "1-4"},
                ...
            ]},
            ...
        ]
    """
    data = _api_get({"action": "parse", "page": "Story_Arcs", "prop": "sections"})
    sections = data["parse"]["sections"]

    sagas = []
    current_saga = None

    # Sagas are level 3 (H3), arcs are level 4 (H4) on the Story Arcs page
    # But section structure may vary -- detect by hierarchy
    for sec in sections:
        level = int(sec["level"])
        name = sec["line"].strip()
        index = int(sec["index"])

        # Skip non-story sections
        if name in ("Overview", "Main Story Arcs", "Short-Term Focused Cover Page Serials",
                     "Anime-Only Arcs", "TV Specials", "Recaps", "Crossovers",
                     "Translation and Dub Issues", "References", "External links",
                     "Site Navigation"):
            continue
        # Also skip if index is past the main story arcs
        if "Saga" in name and "Rookies" not in name and "New World" not in name:
            current_saga = {"saga": name, "arcs": []}
            sagas.append(current_saga)
        elif current_saga is not None and "Arc" in name:
            # Fetch the arc subsection to extract chapter/episode ranges
            wikitext = _get_section_wikitext("Story_Arcs", index)
            arc_info = {"name": name, "wiki_page": name.replace(" ", "_")}

            # Parse "Chapters: N (X-Y)" and "Episodes: N (X-Y)"
            ch_match = re.search(
                r"[Cc]hapters?\]\]?\s*:\s*(\d+)\s*\((\d+)\s*-\s*(\d+)\)", wikitext
            )
            if ch_match:
                arc_info["chapters"] = f"{ch_match.group(2)}-{ch_match.group(3)}"
                arc_info["chapter_count"] = int(ch_match.group(1))

            ep_match = re.search(
                r"[Ee]pisodes?\]\]?\s*:\s*(\d+)\s*\((\d+)\s*-\s*(\d+)\)", wikitext
            )
            if ep_match:
                arc_info["episodes"] = f"{ep_match.group(2)}-{ep_match.group(3)}"
                arc_info["episode_count"] = int(ep_match.group(1))

            current_saga["arcs"].append(arc_info)

    return sagas


def normalize_arc_id(arc_name):
    """
    Normalize a wiki arc name to a standardized snake_case arc_id.

    Examples:
        "Romance Dawn Arc"   -> "romance_dawn"
        "East Blue"          -> "east_blue"
        "Alabasta Arc"       -> "alabasta"
        "Drum Island Arc"    -> "drum_island"
        "Water 7 Arc"        -> "water_7"
        "Whole Cake Island"  -> "whole_cake_island"

    Args:
        arc_name: Arc name string from the wiki (e.g. "Romance Dawn Arc")

    Returns:
        str: snake_case arc identifier, or "" if input is empty.
    """
    if not arc_name:
        return ""
    # Strip trailing "Arc" or "Saga"
    name = arc_name.strip()
    name = re.sub(r"\s+(Arc|Saga)$", "", name, flags=re.IGNORECASE)
    # Lowercase and replace non-alphanumeric with underscores
    name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return name


def get_arc_for_episode(episode_num):
    """
    Look up which arc and saga an episode belongs to.

    Args:
        episode_num: Episode number (int)

    Returns:
        dict: {"saga": str, "arc": str, "wiki_page": str,
               "episodes": "X-Y", "chapters": "X-Y"} or None
    """
    sagas = get_story_arcs()
    for saga in sagas:
        for arc in saga["arcs"]:
            ep_str = arc.get("episodes", "")
            if not ep_str:
                continue
            parts = ep_str.split("-")
            if len(parts) == 2:
                try:
                    start, end = int(parts[0]), int(parts[1])
                    if start <= episode_num <= end:
                        return {
                            "saga": saga["saga"],
                            "arc": arc["name"],
                            "wiki_page": arc["wiki_page"],
                            "episodes": ep_str,
                            "chapters": arc.get("chapters", ""),
                        }
                except ValueError:
                    continue
    return None


# ---------------------------------------------------------------------------
# Convenience: get everything for an episode in one call
# ---------------------------------------------------------------------------

def get_episode_data(episode_num):
    """
    Fetch all available wiki data for an episode.

    Returns:
        dict with keys: characters, summary, short_summary, infobox
    """
    return {
        "episode": episode_num,
        "characters": get_episode_characters(episode_num),
        "summary": get_episode_summary(episode_num),
        "short_summary": get_episode_short_summary(episode_num),
        "infobox": get_episode_infobox(episode_num),
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import io

    # Force UTF-8 output on Windows (cp1252 can't print Japanese chars)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

    episode = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    char_name = sys.argv[2] if len(sys.argv) > 2 else "Monkey D. Luffy"

    print(f"=== Episode {episode} Characters ===")
    for c in get_episode_characters(episode):
        ann = f" ({c['annotation']})" if c["annotation"] else ""
        print(f"  - {c['name']}{ann}  [{c['wiki_page']}]")

    print(f"\n=== Episode {episode} Short Summary ===")
    print(get_episode_short_summary(episode))

    print(f"\n=== Episode {episode} Long Summary ===")
    summary = get_episode_summary(episode)
    print(summary[:500] + "..." if len(summary) > 500 else summary)

    print(f"\n=== Episode {episode} Infobox ===")
    infobox = get_episode_infobox(episode)
    for k, v in infobox.items():
        print(f"  {k}: {v}")

    print(f"\n=== Episode {episode} Chapter Mapping ===")
    ch_data = get_episode_chapters(episode)
    if ch_data:
        filler_str = " (FILLER)" if ch_data["is_filler"] else ""
        print(f"  Chapters: {ch_data['chapters']}{filler_str}")
    else:
        print("  (not in mapping)")

    # Show reverse lookup for first chapter
    if ch_data and ch_data["chapters"]:
        first_ch = ch_data["chapters"][0]
        rev = get_chapter_episodes(first_ch)
        if rev:
            print(f"  Chapter {first_ch} -> Episodes: {rev['episodes']}")

    print(f"\n=== Character: {char_name} ===")
    info = get_character_info(char_name)
    for k, v in info.items():
        val = str(v)
        print(f"  {k}: {val[:120]}{'...' if len(val) > 120 else ''}")

    print(f"\n=== Character Image: {char_name} ===")
    print(f"  {get_character_image(char_name)}")

    print(f"\n=== Arc for Episode {episode} ===")
    arc_lookup = get_arc_for_episode(episode)
    if arc_lookup:
        for k, v in arc_lookup.items():
            print(f"  {k}: {v}")
        arc_name = arc_lookup["arc"]
    else:
        arc_name = "Romance Dawn Arc"
        print("  (not found, defaulting to Romance Dawn Arc)")

    print(f"\n=== Arc Ranges: {arc_name} ===")
    ranges = get_arc_ranges(arc_name)
    for k, v in ranges.items():
        print(f"  {k}: {v}")

    print(f"\n=== Arc Summary: {arc_name} ===")
    for part in get_arc_summary(arc_name):
        text = part["text"]
        preview = text[:200] + "..." if len(text) > 200 else text
        print(f"  [{part['heading']}] {preview}")

