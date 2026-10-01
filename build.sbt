// num-scala: a (nearly complete) port of NumPy to Scala 3.

ThisBuild / scalaVersion := "3.3.6"
ThisBuild / organization := "com.github.kmizu"
ThisBuild / organizationName := "Kota Mizushima"
ThisBuild / homepage := Some(url("https://github.com/kmizu/num-scala"))
ThisBuild / licenses := Seq("BSD-3-Clause" -> url("https://opensource.org/licenses/BSD-3-Clause"))
ThisBuild / scmInfo := Some(
  ScmInfo(
    url("https://github.com/kmizu/num-scala"),
    "scm:git:https://github.com/kmizu/num-scala.git",
    Some("scm:git:git@github.com:kmizu/num-scala.git")
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
    name := "num-scala",
    description := "A nearly complete port of NumPy (n-dimensional arrays, ufuncs, linalg, fft, random, ...) to Scala 3",
    scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked"),
    libraryDependencies += "org.scalameta" %% "munit" % "1.1.1" % Test,
    Test / parallelExecution := true,
    Compile / doc / scalacOptions ++= Seq("-project", "num-scala"),
    pomIncludeRepository := { _ => false },
    Test / publishArtifact := false
  )
