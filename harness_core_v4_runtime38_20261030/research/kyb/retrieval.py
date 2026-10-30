"""Minimum lexical evidence for candidates, never semantic answer certification."""
import re


STOP_WORDS = frozenset('''a an the and or but if then than as at by for from in into of on onto to with
is are was were be been being do does did have has had can could may might must shall should will would
i me my we our you your he she it its they their them this that these those there here which what when
where who how why all any both each either every few many more most much other same some such only own
also so very just about above below between through during before after until while again further
am not no nor without over under up down out off once too s t
please find read show query history historical record records experience experiences example examples
measure measuring current previous recent supplied existing'''.split())


def content_terms(text):
    # Function/request words are removed before the query-length bound. Axes x/y/z remain searchable.
    return list(dict.fromkeys(term for term in re.findall(r'\w+', text.casefold(), re.UNICODE)
                              if term not in STOP_WORDS and not term.isdecimal()))


def sufficient_overlap(query_terms, document):
    # A multiword question cannot be admitted by one ordinary shared word. Single-keyword searches
    # still work. This is deliberately only a lower bound on lexical relevance, not an answer test.
    return len(set(query_terms) & set(content_terms(document))) >= min(2, len(query_terms))
