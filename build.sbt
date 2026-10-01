// numscala: a (nearly complete) port of NumPy to Scala 3.

ThisBuild / scalaVersion := "3.3.6"
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
