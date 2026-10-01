# Publishing num-scala to Maven Central

The build is ready for Maven Central through the **Sonatype Central Portal**
(<https://central.sonatype.com>) using [sbt-ci-release](https://github.com/sbt/sbt-ci-release):

* coordinates: `com.github.kmizu` % `num-scala_3` (`"com.github.kmizu" %% "num-scala" % "<version>"`)
* version: derived from the git tag by sbt-dynver (`v0.1.0` → `0.1.0`; untagged commits are
  `-SNAPSHOT`s that go to the Central snapshot repository)
* the POM carries everything Central requires (name, description, url, BSD-3-Clause
  licence, SCM, developer); sources and Scaladoc jars are produced by `publishSigned`,
  and every artifact is GPG-signed.

What cannot live in the repository are the credentials. They are needed once.

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
git tag -a v0.1.0 -m "num-scala 0.1.0"
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
