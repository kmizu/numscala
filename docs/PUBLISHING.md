# Publishing numscala to Maven Central

The build is ready for Maven Central through the **Sonatype Central Portal**
(<https://central.sonatype.com>) using [sbt-ci-release](https://github.com/sbt/sbt-ci-release):

* coordinates: `com.github.kmizu` % `numscala_3` (`"com.github.kmizu" %% "numscala" % "<version>"`)
* version: derived from the git tag by sbt-dynver (`v0.1.0` → `0.1.0`; untagged commits are
  `-SNAPSHOT`s that go to the Central snapshot repository)
* the POM carries everything Central requires (name, description, url, BSD-3-Clause
  licence, SCM, developer); sources and Scaladoc jars are produced by `publishSigned`,
  and every artifact is GPG-signed.

What cannot live in the repository are the credentials. They are needed once.

## Current status (2026-10-01)

| Item | Status |
|---|---|
| GitHub repository `kmizu/numscala` | renamed from `kmizu/num-scala` (old URLs redirect), CI green (JDK 17/21) |
| Signing key | the maintainer's common release key, RSA 4096 `E0FA067379B91CFA168154C9C8BFC42B047CB4C1` (`Kota Mizushima <kmizu.main@gmail.com>`), on keyserver.ubuntu.com and keys.openpgp.org; passphrase kept locally in `~/.config/release-signing/pgp-passphrase` |
| `PGP_SECRET`, `PGP_PASSPHRASE` secrets | set |
| `SONATYPE_USERNAME`, `SONATYPE_PASSWORD` secrets | set |
| **0.1.0** | `com.github.kmizu:numscala_3:0.1.0`, package `com.github.kmizu.numscala` |
| **0.1.1** | adds the backquoted floor-division operator ``a `//` b`` / ``a `//=` b`` |
| **0.2.0** | NEP 50 weak scalars: `float32Array + 2.0` stays float32, `int8Array + 1` stays int8 (operators, in-place, ufuncs). Breaking: `np.add(int32Array, 3L)` is int32 (was int64) |
| **0.3.0** | Float32 CPU kernel layer `com.github.kmizu.numscala.cpu` (NS-CPU-001: `gemmInto`, row gather/scatter/coalesce, elementwise ops, affine scan, `Workspace`); float32 `np.matmul` runs on it without copies (about 7x faster). Fix: matmul no longer drops `0 * inf` / `0 * nan` (now NaN, as in NumPy). The optional JDK 25 `vector25` backend is not published |
| **0.4.0** | `ParallelF32.gemmInto` / `gemmRowsInto` (worker-count-independent parallel GEMM on a caller-owned executor); vectorised deterministic exp for sigmoid/SiLU/log-sum-exp; faster NT GEMM; fix: vector kernels could randomly stay on C1 (~60x slower). First release of `com.github.kmizu:numscala-vector25_3` (JDK 25, `--add-modules=jdk.incubator.vector`), published from JDK 25 via `CI_RELEASE` |
| **0.5.0** | `gemmTileInto`; 2-D tiled `ParallelF32` (GEMVs parallelise by columns, products below `minWork` skip the executor); vector25 6x2 GEMM micro-kernel (up to 1.45x); `tools/bitcheck`. gemm/gather/scan bits identical to 0.4.0 |
| superseded | `com.github.kmizu:num-scala_3:0.1.0` (package `numscala`) was published first under the old name. Maven Central artifacts cannot be deleted, so it remains available but is no longer maintained. |

Releasing a new version is now just `git tag -a vX.Y.Z -m ... && git push origin vX.Y.Z`. Since 0.4.0 the
release publishes two artifacts, `numscala_3` and `numscala-vector25_3`; check both on Central afterwards.

## 1. Namespace

`com.github.kmizu` must be a verified namespace of your Central Portal account
(Account → Namespaces). Accounts migrated from the old OSSRH (`oss.sonatype.org`)
keep their verified namespaces. If it is not verified, either verify it (Central asks
you to create a temporary public GitHub repository with a given name) or switch to the
automatically verifiable `io.github.kmizu` by changing `ThisBuild / organization` in
`build.sbt`.

## 2. Central Portal user token

Central Portal → Account → *Generate User Token*. You get a username and a password.

## 3. GPG key

```bash
gpg --full-generate-key                       # RSA 4096
gpg --list-secret-keys --keyid-format LONG    # note the key id
gpg --keyserver keyserver.ubuntu.com --send-keys <KEY_ID>
gpg --armor --export-secret-keys <KEY_ID> | base64 -w0   # value for PGP_SECRET
```

## 4. GitHub secrets

In the repository settings (Secrets and variables → Actions) add:

| Secret | Value |
|---|---|
| `SONATYPE_USERNAME` | token username from step 2 |
| `SONATYPE_PASSWORD` | token password from step 2 |
| `PGP_SECRET` | base64 of the armored secret key (step 3) |
| `PGP_PASSPHRASE` | the key's passphrase |

## 5. Release

```bash
git tag -a v0.1.0 -m "numscala 0.1.0"
git push origin v0.1.0
```

`.github/workflows/release.yml` runs the tests and then `sbt ci-release`, which signs,
bundles and uploads the artifacts and releases the deployment on the Central Portal.
Artifacts appear on Maven Central typically within 10–30 minutes.

### Releasing from a local machine instead

```bash
export SONATYPE_USERNAME=... SONATYPE_PASSWORD=... PGP_PASSPHRASE=...
export PGP_SECRET=$(gpg --armor --export-secret-keys <KEY_ID> | base64 -w0)
git tag v0.1.0
sbt ci-release
```

### Dry run (no credentials needed)

```bash
sbt publishLocal        # installs into ~/.ivy2/local
sbt publishM2           # installs into ~/.m2/repository
```
