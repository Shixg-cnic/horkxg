#!/usr/bin/env python3
"""Check whether a binary CSR stores both directions of every edge."""

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count reciprocal, one-way, duplicate, and self-loop CSR edges."
    )
    parser.add_argument("graph_dir", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--indices-dtype", choices=("int32", "int64"), default="int64"
    )
    parser.add_argument("--chunk-entries", type=int, default=8_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indptr_path = args.graph_dir / f"{args.name}_indptr.bin"
    indices_path = args.graph_dir / f"{args.name}_indices.bin"

    indptr = np.memmap(indptr_path, dtype=np.int64, mode="r")
    if indptr.size < 1:
        raise ValueError("indptr is empty")
    num_vertices = int(indptr.size - 1)
    num_entries = int(indptr[-1])
    index_dtype = np.dtype(args.indices_dtype)
    expected_bytes = num_entries * index_dtype.itemsize
    actual_bytes = indices_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"indices size mismatch: expected {expected_bytes} bytes for "
            f"{num_entries} {index_dtype} entries, got {actual_bytes}; "
            "select the correct --indices-dtype"
        )
    if num_vertices >= 2**32:
        raise ValueError("packed edge keys require fewer than 2^32 vertices")

    indices = np.memmap(indices_path, dtype=index_dtype, mode="r")
    keys = np.empty(num_entries, dtype=np.uint64)
    chunk = max(1, args.chunk_entries)

    for first_vertex in range(0, num_vertices, chunk):
        last_vertex = min(num_vertices, first_vertex + chunk)
        begin = int(indptr[first_vertex])
        end = int(indptr[last_vertex])
        degrees = np.diff(indptr[first_vertex : last_vertex + 1])
        sources = np.repeat(
            np.arange(first_vertex, last_vertex, dtype=np.uint64), degrees
        )
        destinations = np.asarray(indices[begin:end], dtype=np.uint64)
        if destinations.size and int(destinations.max()) >= num_vertices:
            raise ValueError("indices contain a vertex id outside [0, n)")
        keys[begin:end] = (sources << np.uint64(32)) | destinations

    keys.sort()
    unique_keys, multiplicities = np.unique(keys, return_counts=True)
    sources = unique_keys >> np.uint64(32)
    destinations = unique_keys & np.uint64(0xFFFFFFFF)
    non_loop = sources != destinations
    loop_entries = int(multiplicities[~non_loop].sum())
    loop_unique = int((~non_loop).sum())

    directed_keys = unique_keys[non_loop]
    directed_counts = multiplicities[non_loop]
    reciprocal_unique = 0
    multiplicity_mismatches = 0

    for begin in range(0, directed_keys.size, chunk):
        current = directed_keys[begin : begin + chunk]
        reverse = ((current & np.uint64(0xFFFFFFFF)) << np.uint64(32)) | (
            current >> np.uint64(32)
        )
        positions = np.searchsorted(directed_keys, reverse)
        found = positions < directed_keys.size
        found_indices = np.flatnonzero(found)
        if found_indices.size:
            found[found_indices] = (
                directed_keys[positions[found_indices]] == reverse[found_indices]
            )
        reciprocal_unique += int(found.sum())
        matched = np.flatnonzero(found)
        if matched.size:
            multiplicity_mismatches += int(
                np.count_nonzero(
                    directed_counts[begin : begin + current.size][matched]
                    != directed_counts[positions[matched]]
                )
            )

    unique_directed = int(directed_keys.size)
    one_way_unique = unique_directed - reciprocal_unique
    bidirectional_pairs = reciprocal_unique // 2
    unique_undirected = bidirectional_pairs + one_way_unique
    duplicate_entries = num_entries - int(unique_keys.size)

    print("=========== CSR Symmetry Check ===========")
    print(f"vertices: {num_vertices}")
    print(f"adjacency_entries: {num_entries}")
    print(f"self_loop_entries: {loop_entries}")
    print(f"unique_self_loops: {loop_unique}")
    print(f"duplicate_entries: {duplicate_entries}")
    print(f"unique_directed_nonloop_edges: {unique_directed}")
    print(f"reciprocal_unique_directed_edges: {reciprocal_unique}")
    print(f"bidirectional_undirected_pairs: {bidirectional_pairs}")
    print(f"one_way_unique_directed_edges: {one_way_unique}")
    print(f"unique_undirected_edges: {unique_undirected}")
    print(f"reverse_multiplicity_mismatches: {multiplicity_mismatches}")
    print(f"topology_is_symmetric: {'yes' if one_way_unique == 0 else 'no'}")
    print(
        "multiset_is_symmetric: "
        f"{'yes' if one_way_unique == 0 and multiplicity_mismatches == 0 else 'no'}"
    )


if __name__ == "__main__":
    main()
