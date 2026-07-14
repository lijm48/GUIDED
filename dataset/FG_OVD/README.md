# dataset/FG_OVD/

This directory should contain the FG-OVD benchmark data.

Download the original FG-OVD benchmark from: https://github.com/therosFG/FG-OVD

Download our generated subject-and-atomic-phrase benchmark files from [this link](TODO), and place the extracted `with_subject_and_atomic_phrases/` directory under `FG_OVD/benchmarks/`.

These files add an explicit subject and decomposed atomic phrases for each category, e.g. `a black dog with a white tail` -> subject `dog`, atomic phrases `a black dog` and `a dog with a white tail`.

Expected structure after preparation:
```
FG_OVD/
└── benchmarks/
    ├── 1_attributes.json
    ├── 2_attributes.json
    ├── 3_attributes.json
    ├── color.json
    ├── material.json
    ├── pattern.json
    ├── shuffle_negatives.json
    ├── transparency.json
    └── with_subject_and_atomic_phrases/  # downloaded generated files
```
