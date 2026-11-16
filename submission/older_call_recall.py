"""仅从预测前的借阅建立索书号索引。支持历史索书号序列或匿名读者画像。"""
from collections import Counter, defaultdict
from math import log2, sqrt

from data_io import TIME, USER, attribute_value

CALL = "索书号"


def ranked(scores):
    return sorted(scores, key=lambda item: (-scores[item], item))


class OlderCallRecall:
    def __init__(self, events, cutoff):
        history = events.loc[events[TIME] < cutoff].sort_values(TIME, kind="stable").copy()
        history[CALL] = history[CALL].map(attribute_value)
        history = history.loc[history[CALL].ne("")]
        self.author = {}
        self.by_author = defaultdict(set)
        self.by_class = defaultdict(set)
        for item, author in history.drop_duplicates(CALL, keep="last")[[CALL, "责任者"]].itertuples(index=False):
            self.author[item] = attribute_value(author)
            if self.author[item]:
                self.by_author[self.author[item]].add(item)
            self.by_class[item.split("/")[0]].add(item)

        self.books = defaultdict(set)
        self.readers = defaultdict(set)
        self.hot = Counter()
        self.local = defaultdict(Counter)
        self.profile = {}
        pairs = history.drop_duplicates([USER, CALL], keep="last")
        for user, item, time, dept in pairs[[USER, CALL, TIME, "单位"]].itertuples(index=False):
            self.books[user].add(item)
            self.readers[item].add(user)
            days = (cutoff-time).days
            weight = 0.2 + 0.8 * 2**(-days / 120)
            self.hot[item] += weight
            if attribute_value(dept):
                self.local[dept][item] += weight
        for u, g in history.groupby(USER):
            self.profile[u] = g[CALL].drop_duplicates(keep="last").tolist()
        self.popular = ranked(self.hot)

    def channels(self, sequence, dept=""):
        seen = set(sequence)
        seeds = list(dict.fromkeys(reversed(sequence)))[:20]
        peer = Counter()
        content = Counter()
        for rank, seed in enumerate(seeds, 1):
            weight = 1 / sqrt(rank)
            users = self.readers.get(seed, ())
            for u in sorted(users):
                contribution = weight / (sqrt(len(users)) * log2(2+len(self.books[u])))
                for item in self.books[u] - seen:
                    peer[item] += contribution / sqrt(len(self.readers[item]))
            for family, scale in ((self.by_class.get(seed.split("/")[0], ()), 1),
                                  (self.by_author.get(self.author.get(seed), ()), 2)):
                for item in family:
                    if item not in seen:
                        content[item] += weight * scale * log2(2+self.hot[item]) / sqrt(len(family))
        local = self.local.get(dept, {})
        return {name: [item for item in items if item not in seen] for name, items in
                (("hot", self.popular), ("college", ranked(local)), ("cf", ranked(peer)),
                 ("content", ranked(content)))}


def blend(channels, weights, limit=100):
    scores = Counter()
    for name, weight in weights.items():
        for rank, item in enumerate(channels[name][:limit], 1):
            scores[item] += weight / (20 + rank)
    return ranked(scores)[:limit]


# 固定权重，所有开发窗口使用同一配方。
WEIGHTS = {"hot": 1., "college": .5, "cf": .75, "content": 1.}

