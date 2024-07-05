import heapq
import itertools

# priority queue helpers when also have a dictionary to map to locations
# from here: https://docs.python.org/3/library/heapq.html

REMOVED = ","  # "<removed-candidate>"  # placeholder for a removed task


def heap_add_update_candidate(candidate, votes, pq, entry_finder):
    "Add a new candidate or update the priority of an existing candidate"
    if candidate in entry_finder:
        heap_remove_candidate(candidate, entry_finder)
    entry = [votes, candidate]
    entry_finder[candidate] = entry
    heapq.heappush(pq, entry)


def heap_remove_candidate(candidate, entry_finder):
    "Mark an existing candidate as REMOVED.  Raise KeyError if not found."
    entry = entry_finder.pop(candidate)
    entry[-1] = REMOVED


class max_min_heap:
    def add_update_candidate(self, candidate, votes):
        heap_add_update_candidate(candidate, -votes, self.max_heap, self.max_heap_entry_finder)
        heap_add_update_candidate(candidate, votes, self.min_heap, self.min_heap_entry_finder)

    def remove_candidate(self, candidate):
        heap_remove_candidate(candidate, self.max_heap_entry_finder)
        heap_remove_candidate(candidate, self.min_heap_entry_finder)

    def pop_least_candidate(self):
        "Remove and return the lowest priority candidate. Raise KeyError if empty."
        while self.min_heap:
            priority, candidate = heappop(self.min_heap)
            if candidate is not REMOVED:
                del self.min_heap_entry_finder[candidate]
                return candidate
        raise KeyError("pop from an empty priority queue")

    def find_worst_candidate(self):
        cand = self.min_heap[0][1]  #  candidate; don't need votes for this one
        # print(self.min_heap)
        while cand is REMOVED:
            heapq.heappop(self.min_heap)  #
            cand = self.min_heap[0][1]
        return cand

    def find_best_candidate(self):
        cand = self.max_heap[0][1]  #  candidate; don't need votes for this one
        while cand is REMOVED:
            heapq.heappop(self.max_heap)  #
            cand = self.max_heap[0][1]
        return -self.max_heap[0][0], self.max_heap[0][1]  # -votes, candidate

    def __init__(self, vote_count):
        self.max_heap_entry_finder = {}
        self.min_heap_entry_finder = {}
        self.max_heap = []
        self.min_heap = []
        for candidate in vote_count:
            self.add_update_candidate(candidate, vote_count[candidate])
