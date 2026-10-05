// Bit-compatibility check between numscala versions (see README.md in this directory).
scalaVersion := "3.3.7"
val v = sys.props.getOrElse("nsv", "0.4.0")
libraryDependencies ++= Seq("com.github.kmizu" %% "numscala" % v, "com.github.kmizu" %% "numscala-vector25" % v)
fork := true
javaOptions ++= Seq("--add-modules=jdk.incubator.vector")
