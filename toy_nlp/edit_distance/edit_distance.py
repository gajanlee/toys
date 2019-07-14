#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File    :   edit_distance.py
@Time    :   2019/07/10 21:35:46
@Author  :   gajanlee
@Contact :   lee_jiazh@163.com
@Desc    :   Reference: http://blog.notdot.net/2010/07/Damn-Cool-Algorithms-Levenshtein-Automata
'''

import bisect

class NFA(object):
    EPSILON = object()
    ANY = object()

    def __init__(self, start_state):
        self.transitions = {}
        self.final_states = set()
        self._start_state = start_state

    @property
    def start_state(self):
        return frozenset(self._expand_state(set([self._start_state])))

    def add_transition(self, src, input, dest):
        self.transitions.setdefault(src, {}).setdefault(input, set()).add(dest)

    def add_final_state(self, state):
        self.final_states.add(state)

    def is_final(self, states):
        return self.final_states.intersection(states)

    def _expand_state(self, states):
        frontier = set(states)
        while frontier:
            state = frontier.pop()
            new_states = self.transitions.get(state, {}).get(NFA.EPSILON, set()) - states

            frontier.update(new_states)
            states.update(new_states)

        return states

    def next_state(self, states, input):
        dest_states = set()

        for state in states:
            state_transitions = self.transitions.get(state, {})
            dest_states |= state_transitions.get(input, set())
            dest_states |= state_transitions.get(NFA.ANY, set())
        
        # 所有可空路径
        return frozenset(self._expand_state(dest_states))

    def get_inputs(self, states):
        inputs = set()
        for state in states:
            inputs.update(self.transitions.get(state, {}).keys())
        return inputs

    def to_dfa(self):
        dfa = DFA(self.start_state)
        frontier = [self.start_state]
        seen = set()
        while frontier:
            current = frontier.pop()
            inputs = self.get_inputs(current)
            for input in inputs:
                if input == NFA.EPSILON: continue

                new_state = self.next_state(current, input)
                if new_state not in seen:
                    frontier.append(new_state)
                    seen.add(new_state)
                    if self.is_final(new_state):
                        dfa.add_final_state(new_state)
                
                if input == NFA.ANY:
                    dfa.set_default_transition(current, new_state)    
                else:
                    dfa.add_transition(current, input, new_state)
        return dfa

def levenshtein_automata(term, k):
    nfa = NFA((0, 0))

    # i花了多少步
    # e是错了多少次，两个合起来描述一个状态
    for i, c in enumerate(term):
        for e in range(k + 1):
            nfa.add_transition((i, e), c, (i + 1, e))

            if e < k:
                # 删除
                nfa.add_transition((i, e), NFA.ANY, (i, e + 1))
                # 插入
                nfa.add_transition((i, e), NFA.EPSILON, (i + 1, e + 1))
                # 替换
                nfa.add_transition((i, e), NFA.ANY, (i + 1, e + 1))

    # 消耗了全部的字符就是结束，len(term)
    for e in range(k + 1):
        if e < k:
            nfa.add_transition((len(term), e), NFA.ANY, (len(term), e + 1))
        nfa.add_final_state((len(term), e))
    return nfa


def test_nfa():
    # 根据可容忍的编辑距离构建自动机
    nfa = levenshtein_automata("food", 1)

    test_string = "fod"

    states = [(0, 0)]
    for char in test_string:
        states = nfa.next_state(states, char)
        if nfa.final_states & states:
            print(f"接受字符 {test_string}")
            return

    print(f"不接受字符 {test_string}")

test_nfa()















class DFA(object):

    def __init__(self, start_state):
        self.start_state = start_state
        self.transitions = {}
        self.final_states = set()
        self.defaults = {}

    def add_transition(self, src, input, dest):
        self.transitions.setdefault(src, {})[input] = dest

    def set_default_transition(self, src, dest):
        self.defaults[src] = dest

    def add_final_state(self, state):
        self.final_states.add(state)

    def is_final(self, state):
        return state in self.final_states

    def next_state(self, src, input):
        # 获取当前状态
        state_transitions = self.transitions.get(src, {})
        # 根据输入字符转移状态
        return state_transitions.get(input, self.defaults.get(src, None))

    def next_valid_string(self, input):
        state = self.start_state
        stack = []

        for i, char in enumerate(input):
            stack.append((input[:i], state, char))
            state = self.next_state(state, char)

            if not state: break
        else:
            stack.append((input[:i+1], state, None))
        
        if self.is_final(state):
            # 合法字符串
            return input
        
        # wall following search
        # 寻找可能的接收状态
        while stack:

            path, state, char = stack.pop()
            x = self.find_next_edge(state, char)
            if x:
                path += x
                state = self.next_state(state, x)
                if self.is_final(state): return path
                
                stack.append((path, state, None))
        return None

    # 找到下一个可能的字符
    def find_next_edge(self, src, char):
        if char is None:
            char = '\0'
        else:
            char = chr(ord(char) + 1)

        state_transitions = self.transitions.get(src, {})

        if char in state_transitions or src in self.defaults:
            return char
        
        labels = sorted(state_transitions.keys())
        pos = bisect.bisect_left(labels, char)

        if pos < len(labels):
            return labels[pos]
        return None


def find_all_matches(word, k, lookup_func):
    lev = levenshtein_automata(word, k).to_dfa()
    match = lev.next_valid_string(u'\0')
    while match:
        next = lookup_func(match)
        if not next:
            return
        if match == next:
            yield match
            next = next + u'\0'
        match = lev.next_valid_string(next)

import bisect
import random

class Matcher(object):
  def __init__(self, l):
    self.l = l
    self.probes = 0
  def __call__(self, w):
    self.probes += 1
    pos = bisect.bisect_left(self.l, w)
    if pos < len(self.l):
      return self.l[pos]
    else:
      return None

words = [x.strip().lower() for x in open("/usr/share/dict/british-english")]

words100 = [x for x in words if random.random() <= 0.01]
words.sort()
m = Matcher(words)
ws = find_all_matches('food', 1, m)
print(list(ws))




def minDistance(word1, word2):
    """
    :type word1: str
    :type word2: str
    :rtype: int
    """
    m, n = len(word1), len(word2)
    if m == 0:return n
    if n == 0:return m
    dp = [[0]*(n+1) for _ in range(m+1)]  # 初始化dp和边界
    for i in range(1, m+1): dp[i][0] = i
    for j in range(1, n+1): dp[0][j] = j
    for i in range(1, m+1):  # 计算dp
        for j in range(1, n+1):
            # 字符相同，编辑距离继承不变
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1] + 1, dp[i][j - 1] + 1, dp[i - 1][j] + 1)
    return dp[m][n]

print(minDistance("foode", "good"))