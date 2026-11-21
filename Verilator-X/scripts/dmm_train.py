#!/usr/bin/env python3
"""
DMM decision tree trainer for 5 networks:
  VGG16, ResNet50, AlexNet, MobileNetV2, GoogLeNet.

- op_type: 0=conv, 1=pool, 2=residual, 3=depthwise
- label:   0=LINE, 1=BLOCK, 2=POOL_STREAM, 3=ELEMENTWISE
- Rule:
    pool                -> 2 (POOL_STREAM)
    residual            -> 3 (ELEMENTWISE)
    depthwise           -> wbytes = in_ch * K^2 ; wbytes > buf*1024 ? LINE : BLOCK
    conv                -> wbytes = out_ch * K^2 * in_ch ; wbytes > buf*1024 ? LINE : BLOCK

- 5000 train / 3000 test, global dedup, no overlap.
"""
import csv
import os
import sys
import collections

try:
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
except ImportError as e:
    print("[ERROR] missing dependency: " + str(e))
    print("Install with: sudo apt-get install -y python3-numpy python3-sklearn")
    sys.exit(1)

FEATURES = ["op_type", "in_h", "in_w", "in_ch", "out_ch", "kernel",
            "pe_rows", "pe_cols", "buffer_kb", "tiles"]
CLASSES  = ["LINE", "BLOCK", "POOL_STREAM", "ELEMENTWISE"]

TRAIN_CSV = os.path.join("data", "train.csv")
TEST_CSV  = os.path.join("data", "test.csv")
CPP_OUT   = os.path.join("tb", "dmm_tree_generated.cpp")

N_TRAIN = 5000
N_TEST  = 3000


# ---------------------------------------------------------------
# Label rule
# ---------------------------------------------------------------
def rule_label(op, in_h, in_w, in_ch, out_ch, kernel, pe, buf):
    if op == 1:
        return 2                                  # POOL_STREAM
    if op == 2:
        return 3                                  # ELEMENTWISE
    if op == 3:                                   # depthwise
        wbytes = in_ch * kernel * kernel
        return 0 if wbytes > buf * 1024 else 1
    # conv
    wbytes = out_ch * kernel * kernel * in_ch
    return 0 if wbytes > buf * 1024 else 1


def make_row(op, size, ich, och, ker, pe, buf):
    tiles = (size * size + pe - 1) // pe
    y = rule_label(op, size, size, ich, och, ker, pe, buf)
    return [op, size, size, ich, och, ker, pe, pe, buf, tiles], y


# ---------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------
def gen_fallback(n_train=N_TRAIN, n_test=N_TEST):
    rng = np.random.RandomState(42)

    SIZE_CANDS   = [4, 8, 16, 32, 64, 128]
    IN_CH_CANDS  = [3, 8, 16, 32, 64, 128, 256, 512]
    OUT_CH_CANDS = [8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    KER_CANDS    = [1, 3, 5, 7]
    PE_CANDS     = [8, 16, 32, 64, 128, 256]
    BUF_CANDS    = [8, 16, 32, 64, 128, 256, 512]

    seen = set()

    def one_conv():
        size = int(rng.choice(SIZE_CANDS))
        ich  = int(rng.choice(IN_CH_CANDS))
        och  = int(rng.choice(OUT_CH_CANDS))
        ker  = int(rng.choice(KER_CANDS))
        pe   = int(rng.choice(PE_CANDS))
        buf  = int(rng.choice(BUF_CANDS))
        return make_row(0, size, ich, och, ker, pe, buf)

    def one_pool():
        size = int(rng.choice(SIZE_CANDS))
        ch   = int(rng.choice(IN_CH_CANDS))
        pe   = int(rng.choice(PE_CANDS))
        buf  = int(rng.choice(BUF_CANDS))
        return make_row(1, size, ch, ch, 2, pe, buf)

    def one_residual():
        size = int(rng.choice(SIZE_CANDS))
        ch   = int(rng.choice(IN_CH_CANDS))
        pe   = int(rng.choice(PE_CANDS))
        buf  = int(rng.choice(BUF_CANDS))
        return make_row(2, size, ch, ch, 0, pe, buf)

    def one_depthwise():
        size = int(rng.choice(SIZE_CANDS))
        ch   = int(rng.choice(IN_CH_CANDS))
        ker  = int(rng.choice([3, 5]))
        pe   = int(rng.choice(PE_CANDS))
        buf  = int(rng.choice(BUF_CANDS))
        return make_row(3, size, ch, ch, ker, pe, buf)

    def gen(n, probs=(0.60, 0.15, 0.10, 0.15)):
        X, y = [], []
        attempts = 0
        max_attempts = n * 200
        while len(X) < n and attempts < max_attempts:
            attempts += 1
            r = rng.rand()
            if r < probs[0]:
                x, yy = one_conv()
            elif r < probs[0] + probs[1]:
                x, yy = one_pool()
            elif r < probs[0] + probs[1] + probs[2]:
                x, yy = one_residual()
            else:
                x, yy = one_depthwise()
            key = tuple(int(v) for v in x)
            if key in seen:
                continue
            seen.add(key)
            X.append(x)
            y.append(yy)
        if len(X) < n:
            print("[warn] only generated " + str(len(X)) + "/" + str(n) +
                  " samples after " + str(attempts) + " attempts")
        return X, y

    trX, trY = gen(n_train)
    teX, teY = gen(n_test)
    return trX, trY, teX, teY


# ---------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------
def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)


def write_csv(path, X, y):
    ensure_dir(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FEATURES + ["label"])
        for xi, yi in zip(X, y):
            w.writerow([int(v) for v in xi] + [int(yi)])


def load_csv(path):
    X, y = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                X.append([float(row[k]) for k in FEATURES])
                y.append(int(row["label"]))
            except (KeyError, ValueError):
                continue
    return X, y


def csv_has_rows(path, min_rows):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return sum(1 for ln in f if ln.strip()) - 1 >= min_rows


def print_dist(tag, y):
    dist = collections.Counter(int(v) for v in y)
    print(tag + " label distribution:")
    for c, name in enumerate(CLASSES):
        print("  {:<14} {}".format(name, dist.get(c, 0)))


# ---------------------------------------------------------------
# C++ export
# ---------------------------------------------------------------
def export_cpp(clf, path):
    t = clf.tree_
    lines = ['#include "dmm_tree_generated.h"',
             '',
             'DataflowTemplate dmm_tree_infer_impl(const DMMFeatures& f) {']

    def emit(node, depth):
        pad = "    " * depth
        if t.children_left[node] == -1:
            v = np.array(t.value[node]).flatten()
            c = int(np.argmax(v))
            lines.append(pad + "return DataflowTemplate::" + CLASSES[c] + ";")
            return
        fname = FEATURES[t.feature[node]]
        thr = float(t.threshold[node])
        lines.append(pad + "if (f." + fname + " <= {:.6f}f) {{".format(thr))
        emit(t.children_left[node],  depth + 1)
        lines.append(pad + "} else {")
        emit(t.children_right[node], depth + 1)
        lines.append(pad + "}")

    emit(0, 1)
    lines.append("}")

    ensure_dir(path)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    if csv_has_rows(TRAIN_CSV, N_TRAIN) and csv_has_rows(TEST_CSV, N_TEST):
        print("[load] " + TRAIN_CSV + " + " + TEST_CSV)
        trX, trY = load_csv(TRAIN_CSV)
        teX, teY = load_csv(TEST_CSV)
    else:
        print("[gen] generating " + str(N_TRAIN) + "/" + str(N_TEST) + " dataset")
        trX, trY, teX, teY = gen_fallback()
        write_csv(TRAIN_CSV, trX, trY)
        write_csv(TEST_CSV,  teX, teY)
        print("[gen] wrote " + TRAIN_CSV + " (" + str(len(trX)) + " rows)")
        print("[gen] wrote " + TEST_CSV  + " (" + str(len(teX)) + " rows)")

    print("Train samples: " + str(len(trX)))
    print("Test  samples: " + str(len(teX)))
    print_dist("Train", trY)
    print_dist("Test ", teY)

    # sklearn 0.17 requires 2D ndarray
    trX = np.array(trX, dtype=float)
    trY = np.array(trY, dtype=int)
    teX = np.array(teX, dtype=float)
    teY = np.array(teY, dtype=int)

    clf = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=5,
        random_state=42
    )
    clf.fit(trX, trY)
    pred = clf.predict(teX)
    acc = accuracy_score(teY, pred)
    print("Test accuracy: {:.4f}".format(acc))

    for c, name in enumerate(CLASSES):
        tot = int(np.sum(teY == c))
        ok  = int(np.sum((teY == c) & (pred == c)))
        if tot:
            print("  {:<14} {}/{} = {:.2f}".format(name, ok, tot, ok / float(tot)))

    n_lines = export_cpp(clf, CPP_OUT)
    print("Generated " + CPP_OUT + " (" + str(n_lines) + " lines)")


if __name__ == "__main__":
    main()
