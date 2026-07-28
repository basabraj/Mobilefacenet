# Enrolled faces (recognition gallery)

Add one subfolder per person, named after them, containing a few clear
photos of their face (JPEG/PNG, any size — the pipeline detects and crops
the face automatically):

```
enrolled_faces/
├── Rahim/
│   ├── 1.jpg
│   └── 2.jpg
└── Karim/
    └── 1.jpg
```

`live_face_pipeline.py` loads this folder once at startup, computes a
MobileFaceNet embedding for every enrolled photo, and matches each live
detected face against this gallery by cosine similarity. More photos per
person (different angles/lighting) improve match reliability.
