"""仅使用历史事件身份的读者级别偏好；不合并原始身份标签。"""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from baseline import USER, ITEM, TIME, attribute_value


@dataclass(frozen=True)
class ReaderLevelConfig:
    enabled: bool = False
    min_users: int = 3
    strength: float = 20.0
    prior_weight: float = 0.25

    def __post_init__(self):
        if self.min_users < 1 or self.strength <= 0 or not 0 <= self.prior_weight <= 1:
            raise ValueError('读者级别参数不合法。')


class ReaderLevelPreference:
    version = 'reader-level-v1'

    def __init__(self, context, config, categories):
        self.context, self.config = context, config
        self.categories = np.asarray(categories)
        if len(categories) != len(context.items):
            raise ValueError('类别与馆藏长度不一致。')
        self.profiles = {}
        pairs, global_pairs = defaultdict(dict), {}
        support = defaultdict(set)
        for row in context.history.sort_values(TIME, kind='stable').to_dict('records'):
            if row[TIME] >= context.cutoff:
                raise ValueError('禁止使用预测时点及之后的事件。')
            user, item = row[USER], context.item_index[row[ITEM]]
            level = attribute_value(row.get('读者级别', ''))
            weight = 2 ** (-(context.cutoff - row[TIME]).total_seconds()
                           / (86400 * context.config.half_life_days))
            pair = (user, item)
            global_pairs[pair] = max(global_pairs.get(pair, 0), weight)
            if level:
                self.profiles[user] = level
                pairs[level][pair] = max(pairs[level].get(pair, 0), weight)
                support[level].add(user)
        self.support = {key: len(value) for key, value in support.items()}

        def vector(values):
            result = np.zeros(len(context.items))
            for (_, item), weight in values.items():
                result[item] += weight
            return result

        global_counts = vector(global_pairs)
        self.global_probability = global_counts / global_counts.sum()
        self.counts = {level: vector(values) for level, values in pairs.items()
                       if self.support[level] >= config.min_users}

    def probabilities(self, user, profile=None):
        explicit = attribute_value((profile or {}).get('读者级别', ''))
        level = explicit or self.profiles.get(user, '')
        counts = self.counts.get(level)
        active = self.config.enabled and counts is not None
        probability = self.global_probability.copy()
        if active:
            probability = (counts + self.config.strength * probability) / (counts.sum() + self.config.strength)
        by_category = defaultdict(float)
        for category, value in zip(self.categories, probability):
            by_category[category] += value
        category_probability = np.asarray([by_category[c] for c in self.categories])
        return probability, category_probability, int(active)
