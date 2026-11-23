"""Shared pieces for the decision-model tests.

- Synthetic workloads shaped like DecisionRAG's navigation stages (clusters, page titles, paragraphs).
- "Needle" workloads: a real question with its gold page and related distractors hidden among filler titles.
- GPU memory sampling via nvidia-smi.
- Small HTTP helpers.

Standard library only.
"""
import json
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request

# Question wording from DecisionRAG App. B.1 (page, paragraph) and B.5 (cluster: the paper quotes it as
# "the page with the information needed to answer the user's question [is] likely in Topic nn").
Q_GROUP = "Is the page with the information needed to answer the user's question likely in Topic {n}?"
Q_PAGE = "Is the information needed to answer the user's question on Page {n}?"
Q_PARA = "Does Paragraph {n} contain information needed to answer the user's question?"

GROUP_SIZE = 106  # median DecisionRAG cluster size on Natural Questions
PARAS_PER_PAGE = 8

# Stage-shaped workloads, smallest first so an out-of-memory failure comes last.
#   stage1_N: N cluster descriptions (paper: 1,000 clusters in 8 requests of ~125)
#   stage2_N: N page titles grouped under cluster descriptions (paper: ~17,500 titles in requests of <=20k tokens;
#             stage2_2500 is ~20k tokens)
#   stage3_N: N paragraphs grouped under page titles (paper: ~230 paragraphs in one request; stage3_230 is ~40k tokens)
SCALE_WORKLOADS = ["stage1_125", "stage2_625", "stage2_1250", "stage2_2500", "stage3_115", "stage3_230"]

# Real questions with a gold page and related but less relevant pages. Used as a sanity check of
# formatting and plumbing, and to compare the two models' rankings on identical inputs.
NEEDLES = [
    {"query": "who wrote the song i will always love you",
     "gold": "I Will Always Love You",
     "distractors": ["Whitney Houston", "The Bodyguard (1992 film)", "Love song", "I Will Survive", "Country music"]},
    {"query": "when did the berlin wall come down",
     "gold": "Fall of the Berlin Wall",
     "distractors": ["Checkpoint Charlie", "East Germany", "Great Wall of China", "Berlin", "Cold War"]},
    {"query": "what is the capital city of australia",
     "gold": "Canberra",
     "distractors": ["Sydney", "Melbourne", "Australian Capital Territory", "Capital city", "Austria"]},
    {"query": "who painted the ceiling of the sistine chapel",
     "gold": "Sistine Chapel ceiling",
     "distractors": ["Leonardo da Vinci", "Raphael", "St. Peter's Basilica", "Fresco", "The Last Supper (Leonardo)"]},
    {"query": "how many players are on a rugby union team",
     "gold": "Rugby union",
     "distractors": ["Rugby league", "American football", "Rugby World Cup", "Association football", "Cricket"]},
    {"query": "what is the boiling point of water at sea level",
     "gold": "Boiling point",
     "distractors": ["Water", "Melting point", "Sea level", "Steam", "Celsius"]},
]

FIRST = ("Anna Boris Carlos Daniel Elena Farah Georg Hana Ivan Julia Kenji Lena Marco Nadia Omar Paula "
         "Quentin Rosa Stefan Tara Umar Vera Walter Ximena Yusuf Zofia").split()
LAST = ("Abbott Bauer Castillo Dubois Eriksen Fischer Garcia Haddad Ivanova Jensen Kowalski Larsen Moreau "
        "Novak Okafor Petrov Quinn Rossi Schmidt Tanaka Ueda Varga Weber Xu Yilmaz Zhang").split()
PLACES = ("Aberdeen Bologna Cork Dresden Edmonton Fresno Ghent Hobart Izmir Jaipur Krakow Lyon Malmo Nantes "
          "Oaxaca Porto Quebec Rotterdam Salzburg Tampere Utrecht Valencia Winnipeg Xiamen York Zagreb "
          "Manitoba Bavaria Andalusia Tasmania Ontario Galicia").split()
NOUNS = ("railway bridge cathedral harbour river festival treaty dynasty canal museum regiment monastery "
         "observatory lighthouse orchestra parliament university academy expedition fortress tramway "
         "reservoir library brewery theatre stadium").split()
PLURALS = ["railway stations", "bridges", "castles", "lakes", "islands", "mountains", "museums", "airports",
           "universities", "newspapers", "football clubs", "radio stations", "rivers", "churches"]
ADJS = "northern ancient royal grand lost silent golden imperial eastern hidden lower upper old new great little".split()
VERBS = "founded rebuilt expanded designed restored renamed acquired documented surveyed opened".split()
ROLES = "politician footballer singer painter novelist physicist architect cyclist composer botanist".split()
SUFFIXES = ["FC", "River", "County", "Airport", "Railway Station", "University"]


def make_title(rng):
    k = rng.randrange(8)
    if k == 0:
        return f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    if k == 1:
        return f"{rng.choice(PLACES)} {rng.choice(NOUNS).title()}"
    if k == 2:
        return f"{rng.choice(NOUNS).title()} of {rng.choice(PLACES)}"
    if k == 3:
        return f"The {rng.choice(ADJS).title()} {rng.choice(NOUNS).title()} ({rng.randint(1930, 2020)} film)"
    if k == 4:
        return f"List of {rng.choice(PLURALS)} in {rng.choice(PLACES)}"
    if k == 5:
        return f"{rng.choice(PLACES)} {rng.choice(SUFFIXES)}"
    if k == 6:
        return f"{rng.choice(ADJS).title()} {rng.choice(NOUNS)}"
    return f"{rng.choice(FIRST)} {rng.choice(LAST)} ({rng.choice(ROLES)})"


def make_sentence(rng):
    return (f"The {rng.choice(ADJS)} {rng.choice(NOUNS)} of {rng.choice(PLACES)} was {rng.choice(VERBS)} "
            f"by {rng.choice(FIRST)} {rng.choice(LAST)} in {rng.randint(1700, 2020)}.")


def make_text(rng, words):
    parts, n = [], 0
    while n < words:
        s = make_sentence(rng)
        parts.append(s)
        n += len(s.split())
    return " ".join(parts)


def make_description(rng):
    """About 30 words, like DecisionRAG's one-sentence cluster descriptions."""
    return (f"Pages about {rng.choice(PLURALS)} and {rng.choice(NOUNS)}s in {rng.choice(PLACES)} and "
            f"{rng.choice(PLACES)}, along with {rng.choice(ROLES)}s, {rng.choice(ROLES)}s and the "
            f"{rng.choice(ADJS)} {rng.choice(NOUNS)} history of the {rng.randint(170, 201)}0s.")


def page_state(query, titles, rng):
    """Page titles numbered in order and grouped under cluster descriptions, as in DecisionRAG stage 2."""
    lines = [f"User question: {query}"]
    for i, title in enumerate(titles):
        if i % GROUP_SIZE == 0:
            lines += ["", f"Group: {make_description(rng)}"]
        lines.append(f"Page {i + 1}: {title}")
    return "\n".join(lines)


def needle_workload(needle, n, seed, gold_pos=None):
    """n page titles with the gold page and its distractors hidden among filler.

    gold_pos (1-based) fixes where the gold page goes; otherwise all special titles are shuffled in.
    """
    rng = random.Random(seed)
    special = [needle["gold"]] + needle["distractors"]
    filler = [make_title(rng) for _ in range(n - len(special))]
    if gold_pos is None:
        titles = filler + special
        rng.shuffle(titles)
    else:
        others = filler + needle["distractors"]
        rng.shuffle(others)
        titles = others[:gold_pos - 1] + [needle["gold"]] + others[gold_pos - 1:]
    return {
        "query": needle["query"],
        "titles": titles,
        "gold_index": titles.index(needle["gold"]),
        "state": page_state(needle["query"], titles, rng),
        "questions": [Q_PAGE.format(n=i + 1) for i in range(n)],
    }


def scale_workload(name, seed):
    stage, n = name.split("_")
    n = int(n)
    rng = random.Random(seed)
    query = NEEDLES[0]["query"]
    if stage == "stage1":
        lines = [f"User question: {query}", ""]
        lines += [f"Topic {i + 1}: {make_description(rng)}" for i in range(n)]
        state, questions = "\n".join(lines), [Q_GROUP.format(n=i + 1) for i in range(n)]
    elif stage == "stage2":
        titles = [make_title(rng) for _ in range(n)]
        state, questions = page_state(query, titles, rng), [Q_PAGE.format(n=i + 1) for i in range(n)]
    elif stage == "stage3":
        lines = [f"User question: {query}"]
        for i in range(n):
            if i % PARAS_PER_PAGE == 0:
                lines += ["", f"Page: {make_title(rng)}"]
            lines.append(f"Paragraph {i + 1}: {make_text(rng, 120)}")
        state, questions = "\n".join(lines), [Q_PARA.format(n=i + 1) for i in range(n)]
    else:
        raise ValueError(f"unknown workload {name}")
    return {"name": name, "state": state, "questions": questions}


class GpuMonitor:
    """Samples GPU memory in use while the block runs. peak_mib stays None without nvidia-smi."""

    def __init__(self, interval=0.25):
        self.interval = interval
        self.peak_mib = None
        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def sample():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            vals = [int(x) for x in out.split() if x.strip().isdigit()]
            return max(vals) if vals else None
        except Exception:
            return None

    def _run(self):
        while not self._stop.is_set():
            v = self.sample()
            if v is not None:
                self.peak_mib = v if self.peak_mib is None else max(self.peak_mib, v)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def gpu_info():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None


RETRY_HTTP_CODES = {425, 429, 503}


def http_json(method, url, payload=None, timeout=600, headers=None, retries=3):
    """JSON request. Retries dropped connections and 425/429/503 with backoff; other HTTP errors raise at once."""
    data = None if payload is None else json.dumps(payload).encode()
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:1000]
            if e.code not in RETRY_HTTP_CODES or attempt == retries:
                raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from None
        except (ConnectionError, urllib.error.URLError):
            if attempt == retries:
                raise
        time.sleep(2 ** attempt)


def http_text(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode(errors="replace")
