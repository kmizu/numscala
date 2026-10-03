package com.github.kmizu.numscala.cpu

/** What the running JVM offers to the kernel backends (K6). */
final case class BackendCapabilities(
    javaVersion: String,
    availableProcessors: Int,
    vectorModulePresent: Boolean,
    vectorBackendOnClasspath: Boolean
)

/** The result of a backend selection: the kernels plus why they were chosen (for logs and reports). */
final case class BackendSelection(kernels: F32Kernels, requested: String, chosen: String, reason: String):
  /** One line suitable for a log or benchmark record. */
  def describe: String = s"backend=$chosen requested=$requested (${kernels.describe}); $reason"

/** Backend choice for the Float32 kernels (K6).
  *
  * `scalar` is always available. `vector25` needs the optional `numscala-vector25` module on the
  * classpath and a JDK 25 started with `--add-modules=jdk.incubator.vector`. Asking for it
  * explicitly fails with the reason when it cannot be loaded; only `auto` falls back to `scalar`.
  * The Vector API class is loaded only after the module check succeeds.
  */
object F32Backend:
  /** System property read once by [[default]]: `scalar` (default), `vector25` or `auto`. */
  val Property = "numscala.cpu.backend"

  private val VectorClass = "com.github.kmizu.numscala.cpu.vector25.VectorF32Kernels$"

  /** Queries the JVM (cheap; does not initialise any Vector API class). */
  def capabilities: BackendCapabilities =
    val module = ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent
    val onClasspath = getClass.getClassLoader.getResource(VectorClass.replace('.', '/') + ".class") != null
    BackendCapabilities(System.getProperty("java.version"), Runtime.getRuntime.availableProcessors(), module, onClasspath)

  /** Selects a backend by name (`scalar`, `vector25`, `auto`); each call re-queries the JVM. */
  def select(requested: String): BackendSelection = requested.trim.toLowerCase match
    case "scalar" => BackendSelection(ScalarF32Kernels, requested, "scalar", "scalar requested")
    case "vector25" =>
      loadVector() match
        case Right(k) => BackendSelection(k, requested, k.name, "vector25 requested and available")
        case Left(why) => throw new IllegalStateException(s"vector25 backend requested but unavailable: $why")
    case "auto" =>
      loadVector() match
        case Right(k) => BackendSelection(k, requested, k.name, "auto: vector25 available")
        case Left(why) => BackendSelection(ScalarF32Kernels, requested, "scalar", s"auto: fell back to scalar ($why)")
    case other => throw new IllegalArgumentException(s"unknown Float32 backend '$other' (expected scalar, vector25 or auto)")

  /** The process-wide selection, made once from the `numscala.cpu.backend` system property (default `scalar`). */
  lazy val default: BackendSelection = select(Option(System.getProperty(Property)).getOrElse("scalar"))

  private def loadVector(): Either[String, F32Kernels] =
    val caps = capabilities
    if !caps.vectorModulePresent then Left("module jdk.incubator.vector is not resolved (start the JVM with --add-modules=jdk.incubator.vector on JDK 25)")
    else if !caps.vectorBackendOnClasspath then Left("numscala-vector25 is not on the classpath")
    else
      try Right(Class.forName(VectorClass).getField("MODULE$").get(null).asInstanceOf[F32Kernels])
      catch case e: (ReflectiveOperationException | LinkageError | ExceptionInInitializerError) => Left(s"loading failed: $e")

/** Formatting of backend and workspace diagnostics (K5/K6). */
object KernelDiagnostics:
  /** A multi-field one-line report: backend, JVM, CPU count and (optionally) workspace counters. */
  def report(selection: BackendSelection, workspace: Workspace | Null = null): String =
    val c = F32Backend.capabilities
    val ws =
      if workspace == null then ""
      else
        val s = workspace.stats
        s" workspace(allocatedBytes=${s.allocatedBytes}, grows=${s.growCount}, bytesPacked=${s.bytesPacked}, " +
          s"bytesConverted=${s.bytesConverted}, floats=${s.floatCapacity}, longs=${s.longCapacity})"
    s"${selection.describe} java=${c.javaVersion} cpus=${c.availableProcessors}$ws"
