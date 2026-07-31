# Feature Superposition Visualizations

## Beginner Three.js lab

[vector-distribution.html](vector-distribution.html) starts from a rotatable
three-dimensional sphere and connects it to the distribution of vector angles
in higher-dimensional spaces. It highlights the 89°–91° band by default and
lets you compare:

- angles from every feature vector to one fixed query;
- angles between every pair of feature vectors;
- literal 3D geometry with angle-preserving 3D sketches of 8D through 4096D
  vectors.

The sphere stays honest about its limitation: above three dimensions it
preserves each vector's angle to the query, but it cannot preserve every
feature-to-feature angle on a 3D screen. The adjacent distribution is computed
from the full high-dimensional vectors.

Serve the directory from the repository root:

```bash
.venv/bin/python -m http.server 8000 \
  --directory visualizations
```

Then open:

```text
http://localhost:8000/vector-distribution.html
```

Three.js is loaded from a version-pinned CDN module, so the first load requires
an internet connection.

## Python mathematical reference

This Python visualization connects four ideas:

1. For unit vectors, a dot product is cosine similarity:
   $\mathbf{q}\cdot\mathbf{f}=\cos\theta$.
2. A representation can superpose active feature directions:
   $\mathbf{x}=\sum_j a_j\mathbf{f}_j$.
3. The same directions can decode the mixture:

   ```math
   \mathbf{f}_i\cdot\mathbf{x}
   = a_i
   + \sum_{j\ne i}a_j(\mathbf{f}_i\cdot\mathbf{f}_j).
   ```

   The first term is the wanted feature. The sum is cross-talk from other
   active features.

4. Random unit vectors in $d$ dimensions have cosine similarities concentrated
   near zero, with a typical scale of $1/\sqrt{d}$. A rough all-pairs estimate
   therefore allows a near-orthogonal codebook whose size grows like
   $\exp(d\epsilon^2/4)$ for a fixed overlap tolerance $\epsilon$.

The exponential statement is about the number of available feature
**directions**, not the number of arbitrary real values that can be stored and
recovered simultaneously. Superposition works best when only a sparse subset
of features is active; increasing the active-feature slider makes the resulting
cross-talk visible.

## Run it

From the repository root, install the optional visualization dependencies once:

```bash
uv pip install --python .venv/bin/python 'matplotlib>=3.9' 'numpy>=2.0'
```

Then open the interactive figure:

```bash
.venv/bin/python \
  visualizations/superposition.py
```

Move the angle slider to see the dot product change. Then change dimensions,
available feature directions, and simultaneously active features to compare
signal with interference. `New directions` draws a new random codebook.

To save a deterministic image without opening a window:

```bash
.venv/bin/python \
  visualizations/superposition.py \
  --save results/superposition.png
```

Run the embedded numerical checks with:

```bash
.venv/bin/python \
  visualizations/superposition.py \
  --self-test
```
