// numscala: a (nearly complete) port of NumPy to Scala 3.

ThisBuild / scalaVersion := "3.3.7"
ThisBuild / organization := "com.github.kmizu"
ThisBuild / organizationName := "Kota Mizushima"
ThisBuild / homepage := Some(url("https://github.com/kmizu/numscala"))
ThisBuild / licenses := Seq("BSD-3-Clause" -> url("https://opensource.org/licenses/BSD-3-Clause"))
ThisBuild / scmInfo := Some(
  ScmInfo(
    url("https://github.com/kmizu/numscala"),
    "scm:git:https://github.com/kmizu/numscala.git",
    Some("scm:git:git@github.com:kmizu/numscala.git")
  )
)
ThisBuild / developers := List(
  Developer(
    id = "kmizu",
    name = "Kota Mizushima",
    email = "kmizu.main@gmail.com",
    url = url("https://github.com/kmizu")
  )
)
ThisBuild / versionScheme := Some("early-semver")

lazy val root = (project in file("."))
  .settings(
    name := "numscala",
    description := "A nearly complete port of NumPy (n-dimensional arrays, ufuncs, linalg, fft, random, ...) to Scala 3",
    scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-release", "17"),
    libraryDependencies += "org.scalameta" %% "munit" % "1.1.1" % Test,
    // suites share global state (np.random.seed, print options), so run them sequentially
    Test / parallelExecution := false,
    Compile / doc / scalacOptions ++= Seq("-project", "numscala"),
    pomIncludeRepository := { _ => false },
    Test / publishArtifact := false
  )

// ---------------------------------------------------------------------------------------------
// Optional modules (NS-CPU-001). The root project above does NOT aggregate them, so `sbt test` and
// `publishLocal` only build the `numscala` artifact on JDK 17+. The release publishes `numscala` and,
// explicitly, `numscala-vector25` (JDK 25; see .github/workflows/release.yml).

val vectorModuleOpts = Seq("--add-modules=jdk.incubator.vector")

/** Vector API (JDK 25, `jdk.incubator.vector`) backend for the Float32 CPU kernels. */
lazy val vector25 = (project in file("vector25"))
  .dependsOn(root % "compile->compile;test->test")
  .settings(
    name := "numscala-vector25",
    description := "Optional JDK 25 Vector API backend for numscala's Float32 CPU kernels",
    // no `-release 17`: the incubator module is only visible when compiling against the full JDK 25 image
    scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked"),
    javacOptions ++= vectorModuleOpts,
    libraryDependencies += "org.scalameta" %% "munit" % "1.1.1" % Test,
    Test / fork := true,
    Test / javaOptions ++= vectorModuleOpts,
    Test / parallelExecution := false,
    run / fork := true,
    run / javaOptions ++= vectorModuleOpts,
    Test / publishArtifact := false,
    pomIncludeRepository := { _ => false }
  )

/** JMH benchmarks for the CPU kernels (never published). */
lazy val benchmarks = (project in file("benchmarks"))
  .dependsOn(root, vector25)
  .enablePlugins(JmhPlugin)
  .settings(
    name := "numscala-benchmarks",
    scalacOptions ++= Seq("-deprecation", "-feature"),
    Jmh / javaOptions ++= vectorModuleOpts,
    publish / skip := true
  )
