# Creating the repository

1. On GitHub, create a new **public** repository named `Pix2Geomodel-v2` under `ARhaman`.
   Do not initialise it with a README — this folder has one.

2. From this folder:

```bash
git init
git add .
git commit -m "Pix2Geomodel v2: leakage-free evaluation protocol, baselines and controls"
git branch -M main
git remote add origin https://github.com/ARhaman/Pix2Geomodel-v2.git
git push -u origin main
```

3. Add a note at the top of the old `ARhaman/GeoPix` README so readers of the 2026
   Geoenergy paper are not misled:

> **Note.** This repository accompanies Al-Fakih et al. (2026), *Geoenergy Science and
> Engineering* 258, 214342, and implements the 256 × 256 workflow described there. The
> leakage-free evaluation protocol, conventional baselines and controls introduced in the
> follow-up study are in [Pix2Geomodel-v2](https://github.com/ARhaman/Pix2Geomodel-v2).

   Also remove or resolve the `TODO` placeholders in that README; a reviewer who follows
   the data-availability link and finds unfinished text will read it as the state of the
   work.

4. Tag a release and mint a DOI. Most journals in this area now expect an archived,
   immutable snapshot rather than a mutable repository:

```bash
git tag -a v2.0.0 -m "Version submitted to Results in Engineering"
git push origin v2.0.0
```

   Then connect the repository to Zenodo (`zenodo.org` → GitHub → enable the repository)
   and re-push the tag. Zenodo issues a DOI for the release.

5. Put the DOI in the manuscript's Data Availability statement, replacing the bare URL.
   Suggested wording is in `docs/DATA_AVAILABILITY.md`.
