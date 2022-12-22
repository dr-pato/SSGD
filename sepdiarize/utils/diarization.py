from collections import defaultdict
import numpy as np
import datetime
import re


def load_rttm(rttm_file):
    ref_turns = defaultdict(list)
    with open(rttm_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            spk = parts[7]
            start = float(parts[3])
            end = start + float(parts[4])
            ref_turns[parts[1]].append((spk, start, end))
    return ref_turns


def write_rttm(ref_turns, id, rttm_file):
    with open(rttm_file, 'w') as f:
        for t in ref_turns:
           line = 'SPEAKER {:s} 1 {:.6f} {:.6f} <NA> <NA> {:s} <NA> <NA>\n'.format(id, t[1], t[2] - t[1], t[0])
           f.write(line)


def delete_shorter(preds, factor):
    def helper(segments, th=np.inf):

        tmp = []
        for s, e in segments:
            if (e - s) > th:
                tmp.append([s, e])
        return tmp

    preds_dict = {}
    for entry in preds:
        event_name = entry[0]
        if event_name not in preds_dict.keys():
            preds_dict[event_name] = [[entry[1], entry[2]]]
        else:
            preds_dict[event_name].append([entry[1], entry[2]])

    for k in preds_dict.keys():
        preds_dict[k] = helper(preds_dict[k], th=factor)

    out = []
    for k in preds_dict.keys():
        for segs in preds_dict[k]:
            out.append([k, segs[0], segs[1]])
    return out


def merge_intervals(intervals, delta= 0.0):
    """
    A simple algorithm can be used:
    1. Sort the intervals in increasing order
    2. Push the first interval on the stack
    3. Iterate through intervals and for each one compare current interval
       with the top of the stack and:
       A. If current interval does not overlap, push on to stack
       B. If current interval does overlap, merge both intervals in to one
          and push on to stack
    4. At the end return stack
    """
    if not intervals:
        return intervals
    intervals = sorted(intervals, key=lambda x: x[0])

    merged = [intervals[0]]
    for current in intervals[1:]:
        previous = merged[-1]
        if (current[0] - delta) <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)
    return merged


def collapse_segments(segments):
    segments = np.array(segments)
    i = 0
    while i < (len(segments) - 1):
        cur_end = segments[i, 1]
        next_start = segments[i+1, 0]
        if cur_end >= next_start:
            if segments[i, 1] < segments[i+1, 1]:
                segments[i, 1] = segments[i+1, 1]
            segments = np.delete(segments, i+1, 0)
        else:
            i += 1

    return segments