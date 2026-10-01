package com.github.kmizu.numscala

import java.io.{BufferedOutputStream, ByteArrayInputStream, ByteArrayOutputStream}
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.util.zip.{CRC32, ZipEntry, ZipFile, ZipOutputStream}
import scala.collection.immutable.ListMap
import scala.jdk.CollectionConverters.*

/** The arrays of an `.npz` archive, keyed by name in archive order (NumPy's `NpzFile`). */
final class NpzFile private[numscala] (private val entries: ListMap[String, NDArray[?]])
    extends scala.collection.immutable.AbstractMap[String, NDArray[?]]:
  /** The array names, like `NpzFile.files`. */
  def files: Seq[String] = entries.keys.toSeq
  def get(key: String): Option[NDArray[?]] = entries.get(key).orElse(entries.get(key.stripSuffix(".npy")))
  def iterator: Iterator[(String, NDArray[?])] = entries.iterator
  def removed(key: String): Map[String, NDArray[?]] = entries.removed(key)
  def updated[V1 >: NDArray[?]](key: String, value: V1): Map[String, V1] = entries.updated(key, value)
  /** The array `key` converted to dtype `dtype` (see [[NpIO.load]]). */
  def get[T](key: String, dtype: DType[T]): NDArray[T] =
    NpIO.convertLoaded(apply(key), dtype)
  override def default(key: String): NDArray[?] =
    throw new NoSuchElementException(s"$key is not a file in the archive")
  override def className: String = "NpzFile"

private[numscala] object NpIO:
  type PathLike = String | Path

  def toPath(f: PathLike, ext: String = ""): Path = f match
    case p: Path => p
    case s: String => Paths.get(if ext.nonEmpty && !s.endsWith(ext) then s + ext else s)

  def convertLoaded[T](a: NDArray[?], dtype: DType[T]): NDArray[T] =
    if a.dtype eq dtype then a.asInstanceOf[NDArray[T]]
    else if NpMiscTypes.canCast(a.dtype, dtype, "same_kind") then a.astypeDyn(dtype).asInstanceOf[NDArray[T]]
    else
      throw new IllegalArgumentException(
        s"Cannot cast array data from dtype('${a.dtype}') to dtype('$dtype') according to the rule 'same_kind'"
      )

  def readBytes(f: PathLike): Array[Byte] =
    val p = toPath(f)
    val raw = Files.readAllBytes(p)
    if p.toString.endsWith(".gz") then
      new java.util.zip.GZIPInputStream(new ByteArrayInputStream(raw)).readAllBytes()
    else raw

  def readLines(src: PathLike | Seq[String], encoding: String): Seq[String] = src match
    case s: Seq[?] => s.asInstanceOf[Seq[String]].flatMap(l => NpIOText.lines(l)).toSeq
    case f => NpIOText.lines(new String(readBytes(f.asInstanceOf[PathLike]), encoding))

  def isZip(b: Array[Byte]): Boolean = b.length >= 4 && b(0) == 'P' && b(1) == 'K' && b(2) == 3 && b(3) == 4

  def writeZip(f: PathLike, arrays: Seq[(String, NDArray[?])], compress: Boolean): Unit =
    val names = arrays.map(_._1)
    if names.distinct.length != names.length then
      throw new IllegalArgumentException(s"Cannot use un-named variables and keyword ${names.diff(names.distinct).head}")
    val out = new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(toPath(f, ".npz"))))
    try
      for (name, arr) <- arrays do
        val bytes = NpIONpy.toBytes(arr)
        val e = new ZipEntry(name + ".npy")
        if compress then e.setMethod(ZipEntry.DEFLATED)
        else
          val crc = new CRC32()
          crc.update(bytes)
          e.setMethod(ZipEntry.STORED)
          e.setSize(bytes.length.toLong)
          e.setCompressedSize(bytes.length.toLong)
          e.setCrc(crc.getValue)
        out.putNextEntry(e)
        out.write(bytes)
        out.closeEntry()
    finally out.close()

  def namedArgs(args: Seq[NDArray[?] | (String, NDArray[?])]): Seq[(String, NDArray[?])] =
    var k = -1
    val pos = args.collect { case a: NDArray[?] => k += 1; (s"arr_$k", a) }
    val named = args.collect { case (n: String, a: NDArray[?]) => (n, a) }
    named ++ pos

  /** NumPy's `_ensure_ndmin_ndarray`. */
  def ensureNdmin[T](a0: NDArray[T], ndmin: Int): NDArray[T] =
    if ndmin < 0 || ndmin > 2 then throw new IllegalArgumentException(s"Illegal value of ndmin keyword: $ndmin")
    var a = a0
    if a.ndim > ndmin then a = a.squeeze()
    if a.ndim < ndmin then
      if ndmin == 1 then a = a.reshape(1)
      else if a.ndim == 0 then a = a.reshape(1, 1)
      else a = a.reshape(1, a.size).T.copy()
    a

  def resolveCols(usecols: Int | Seq[Int] | Null): Option[Seq[Int]] = usecols match
    case null => None
    case i: Int => Some(Seq(i))
    case s: Seq[?] => Some(s.asInstanceOf[Seq[Int]])

  def commentList(c: String | Seq[String] | Null): Seq[String] = c match
    case null => Nil
    case s: String => Seq(s)
    case s: Seq[?] => s.asInstanceOf[Seq[String]]

  def selectCols(fields: Seq[String], cols: Option[Seq[Int]], row: Int): Seq[(String, Int)] = cols match
    case None => fields.zipWithIndex
    case Some(cs) =>
      cs.map { c =>
        val j = if c < 0 then c + fields.length else c
        if j < 0 || j >= fields.length then
          throw new IllegalArgumentException(s"invalid column index $c at row $row with ${fields.length} columns")
        (fields(j), j)
      }

  def finish[T](rows: Seq[Array[T]], d: DType[T], ndmin: Int, unpack: Boolean): NDArray[T] =
    val ncol = rows.headOption.map(_.length).getOrElse(0)
    val flat = d.newArray(rows.length * ncol)
    var k = 0
    rows.foreach { r => System.arraycopy(r, 0, flat, k, ncol); k += ncol }
    val a0 = if rows.isEmpty then NDArray.fromArray(flat, Array(0))(using d) else NDArray.fromArray(flat, Array(rows.length, ncol))(using d)
    val a = ensureNdmin(a0, ndmin)
    if unpack then a.T else a

  /** Splits `s` into raw fields for `fromfile` / text `sep`. */
  def splitSep(s: String, sep: String): Seq[String] =
    if sep.trim.isEmpty then s.trim.split("\\s+").toSeq.filter(_.nonEmpty)
    else
      val t = s.trim
      if t.isEmpty then Nil
      else t.split("\\s*" + java.util.regex.Pattern.quote(sep.trim) + "\\s*", -1).toSeq.map(_.trim)

  def rawDecode[T](bytes: Array[Byte], d: DType[T], count: Int, offset: Int): NDArray[T] =
    if d.isString then throw new IllegalArgumentException("cannot read raw binary data of dtype str (itemsize unknown)")
    if offset < 0 || offset > bytes.length then
      throw new IllegalArgumentException("offset must be non-negative and no greater than buffer length")
    val item = d.itemSize
    val avail = bytes.length - offset
    val n =
      if count < 0 then
        if avail % item != 0 then throw new IllegalArgumentException("buffer size must be a multiple of element size")
        avail / item
      else
        if count.toLong * item > avail then throw new IllegalArgumentException("buffer is smaller than requested size")
        count
    val descr = NpIONpy.Descr(d, ByteOrder.LITTLE_ENDIAN, d.kind, item)
    val data = NpIONpy.decodeData(ByteBuffer.wrap(bytes, offset, n * item), descr, n)
    NDArray.fromArray(data.asInstanceOf[Array[T]], Array(n))(using d)

  def rowValues(a: NDArray[?]): Array[Any] = a.toArray.asInstanceOf[Array[?]].map {
    case b: Byte => b.toLong
    case s: Short => s.toLong
    case i: Int => i.toLong
    case f: Float => f.toDouble
    case other => other
  }

/** Input/output routines: `.npy`/`.npz` files, text files and raw binary buffers. */
trait NpIO:
  import NpIO.*

  // ------------------------------------------------------------------ npy / npz

  /** `np.save`: writes `arr` in `.npy` format (`.npy` is appended to a `String` path without it). */
  def save(file: String | Path, arr: NDArray[?]): Unit =
    Files.write(toPath(file, ".npy"), NpIONpy.toBytes(arr))

  /** `np.load` for a `.npy` file; the dtype is whatever the file declares. Use [[load_npz]] for `.npz`. */
  def load(file: String | Path): NDArray[?] =
    val bytes = readBytes(file)
    if isZip(bytes) then
      throw new IllegalArgumentException(s"$file is an .npz archive; use np.load_npz to read it")
    NpIONpy.fromBytes(bytes)

  /** Typed `np.load`: loads a `.npy` file and converts it to `dtype` (casting rule `same_kind`). */
  def load[T](file: String | Path, dtype: DType[T]): NDArray[T] = convertLoaded(load(file), dtype)

  /** `np.load` for an `.npz` archive written by `savez` / `savez_compressed`. */
  def load_npz(file: String | Path): NpzFile =
    val zf = new ZipFile(toPath(file).toFile)
    try
      val entries = zf.entries().asScala.toSeq.filter(!_.isDirectory).map { e =>
        val in = zf.getInputStream(e)
        try
          val bytes = in.readAllBytes()
          val name = if e.getName.endsWith(".npy") then e.getName.dropRight(4) else e.getName
          name -> NpIONpy.fromBytes(bytes)
        finally in.close()
      }
      new NpzFile(ListMap.from(entries))
    finally zf.close()

  /** `np.savez`: positional arrays are stored as `arr_0`, `arr_1`, ...; `("name", a)` pairs under their name. */
  def savez(file: String | Path, args: (NDArray[?] | (String, NDArray[?]))*): Unit =
    writeZip(file, namedArgs(args), compress = false)

  /** `np.savez(file, **kwds)`. */
  def savez(file: String | Path, kwds: Map[String, NDArray[?]]): Unit = writeZip(file, kwds.toSeq, compress = false)

  /** `np.savez_compressed`: like [[savez]] but with deflate compression. */
  def savez_compressed(file: String | Path, args: (NDArray[?] | (String, NDArray[?]))*): Unit =
    writeZip(file, namedArgs(args), compress = true)

  /** `np.savez_compressed(file, **kwds)`. */
  def savez_compressed(file: String | Path, kwds: Map[String, NDArray[?]]): Unit =
    writeZip(file, kwds.toSeq, compress = true)

  // ------------------------------------------------------------------ text

  /** `np.savetxt`: writes a 1-D or 2-D array as text. `fmt` is a Python `%` format (one per column,
    * a sequence of them, or a whole-row format); complex values are written as ` (re+imj)`.
    */
  def savetxt[T](
      fname: String | Path,
      X: NDArray[T],
      fmt: String | Seq[String] = "%.18e",
      delimiter: String = " ",
      newline: String = "\n",
      header: String = "",
      footer: String = "",
      comments: String = "# ",
      encoding: String = "UTF-8"
  ): Unit =
    val x2 = X.ndim match
      case 1 => X.reshape(X.size, 1)
      case 2 => X
      case n => throw new IllegalArgumentException(s"Expected 1D or 2D array, got ${n}D array instead")
    val ncol = x2.shapeArr(1)
    val isComplex = X.dtype.isComplex
    val rowFmt: String = fmt match
      case s: Seq[?] =>
        val fs = s.asInstanceOf[Seq[String]]
        if fs.length != ncol then throw new IllegalArgumentException(s"fmt has wrong shape.  ${fs.mkString("[", ", ", "]")}")
        val per = if isComplex then fs.map(f => s" ($f+${f}j)") else fs
        per.mkString(delimiter)
      case f: String =>
        val n = NpIOPyFormat.countSpecs(f)
        if n == 1 then Seq.fill(ncol)(if isComplex then s" ($f+${f}j)" else f).mkString(delimiter)
        else if isComplex && n != 2 * ncol then throw new IllegalArgumentException(s"fmt has wrong number of % formats:  $f")
        else if !isComplex && n != ncol then throw new IllegalArgumentException(s"fmt has wrong number of % formats:  $f")
        else f
    val sb = new StringBuilder
    if header.nonEmpty then sb.append(comments).append(header.replace("\n", "\n" + comments)).append(newline)
    for i <- 0 until x2.shapeArr(0) do
      val row = x2(i, ::)
      val vals: Seq[Any] =
        if isComplex then row.toSeq.flatMap { v => val c = X.dtype.toComplex(v); Seq(c.re, c.im) }
        else rowValues(row).toSeq
      val line = NpIOPyFormat.format(rowFmt, vals)
      sb.append(if isComplex then line.replace("+-", "-") else line).append(newline)
    if footer.nonEmpty then sb.append(comments).append(footer.replace("\n", "\n" + comments)).append(newline)
    Files.write(toPath(fname), sb.toString.getBytes(encoding))

  /** `np.loadtxt`: reads a text file (or a `Seq` of lines) into an array of `dtype` (default float64).
    * `delimiter = null` splits on whitespace; `usecols` may be negative; `quotechar` enables quoted fields.
    */
  def loadtxt[T](
      fname: String | Path | Seq[String],
      dtype: DType[T] | Null = null,
      comments: String | Seq[String] | Null = "#",
      delimiter: String | Null = null,
      skiprows: Int = 0,
      usecols: Int | Seq[Int] | Null = null,
      unpack: Boolean = false,
      ndmin: Int = 0,
      max_rows: Int = -1,
      quotechar: Char | Null = null,
      encoding: String = "UTF-8"
  )(using dd: DefaultDType[T]): NDArray[T] =
    val d: DType[T] = if dtype == null then dd.dtype else dtype.asInstanceOf[DType[T]]
    val q: Option[Char] = quotechar match
      case c: Char => Some(c)
      case _ => None
    val cmts = commentList(comments)
    val cols = resolveCols(usecols)
    val rows = scala.collection.mutable.ArrayBuffer.empty[Array[T]]
    var ncols = -1
    val it = readLines(fname, encoding).iterator.drop(skiprows)
    while it.hasNext && (max_rows < 0 || rows.length < max_rows) do
      val line = NpIOText.stripComment(it.next(), cmts, q)
      if line.trim.nonEmpty then
        val fields = NpIOText.split(line, delimiter, q)
        val sel = selectCols(fields, cols, rows.length)
        if ncols >= 0 && sel.length != ncols then
          throw new IllegalArgumentException(
            s"the number of columns changed from $ncols to ${sel.length} at row ${rows.length + 1}; " +
              "use `usecols` to select a subset and avoid this error"
          )
        ncols = sel.length
        val r = d.newArray(sel.length)
        var j = 0
        for (s, col) <- sel do
          r(j) =
            try NpIOText.parse(d, s)
            catch
              case _: IllegalArgumentException =>
                throw new IllegalArgumentException(
                  s"could not convert string '${s.trim}' to ${d.name} at row ${rows.length}, column ${col + 1}."
                )
          j += 1
        rows += r
    finish(rows.toSeq, d, ndmin, unpack)

  /** `np.genfromtxt` (without `names`): like `loadtxt`, but missing (empty or listed in
    * `missing_values`) and unparsable fields become `filling_values` (default NaN for floats,
    * -1 for integers, `False` for bool). `delimiter` may also be a field width or widths.
    */
  def genfromtxt[T](
      fname: String | Path | Seq[String],
      dtype: DType[T] | Null = null,
      comments: String | Null = "#",
      delimiter: String | Int | Seq[Int] | Null = null,
      skip_header: Int = 0,
      skip_footer: Int = 0,
      missing_values: String | Seq[String] | Null = null,
      filling_values: Any = null,
      usecols: Int | Seq[Int] | Null = null,
      autostrip: Boolean = false,
      max_rows: Int = -1,
      unpack: Boolean = false,
      ndmin: Int = 0,
      invalid_raise: Boolean = true,
      encoding: String = "UTF-8"
  )(using dd: DefaultDType[T]): NDArray[T] =
    val d: DType[T] = if dtype == null then dd.dtype else dtype.asInstanceOf[DType[T]]
    val cmts = commentList(comments)
    val cols = resolveCols(usecols)
    val missing: Set[String] = (missing_values match
      case null => Seq.empty[String]
      case s: String => s.split(",").toSeq
      case s: Seq[?] => s.asInstanceOf[Seq[String]]
    ).map(_.trim).toSet + ""
    val defaultFill: T = d.kind match
      case 'f' | 'c' => d.fromDouble(Double.NaN)
      case 'i' => d.fromLong(-1L)
      case _ => d.zero
    def fillFor(col: Int): T = filling_values match
      case null => defaultFill
      case s: Seq[?] => if col < s.length then d.coerce(s(col)) else defaultFill
      case m: Map[?, ?] => m.asInstanceOf[Map[Int, Any]].get(col).map(d.coerce).getOrElse(defaultFill)
      case v => d.coerce(v)
    def splitLine(line: String): Seq[String] = delimiter match
      case null => NpIOText.split(line, null, None)
      case s: String => NpIOText.split(line, s, None)
      case w: Int => line.grouped(w).toSeq
      case ws: Seq[?] =>
        var p = 0
        ws.asInstanceOf[Seq[Int]].map { w => val f = line.slice(p, p + w); p += w; f }
    val all = readLines(fname, encoding).drop(skip_header)
    val raw = all.zipWithIndex.flatMap { (l, i) =>
      val s = NpIOText.stripComment(l, cmts, None)
      if s.trim.isEmpty then None else Some((splitLine(s), skip_header + i + 1))
    }
    val kept0 = if skip_footer > 0 then raw.dropRight(skip_footer) else raw
    val kept = if max_rows >= 0 then kept0.take(max_rows) else kept0
    val ncol = kept.headOption.map(r => cols.fold(r._1.length)(_.length)).getOrElse(0)
    val bad = kept.drop(1).filter(r => cols.isEmpty && r._1.length != ncol)
    if bad.nonEmpty && invalid_raise then
      throw new IllegalArgumentException(
        "Some errors were detected !\n" +
          bad.map(r => s"    Line #${r._2} (got ${r._1.length} columns instead of $ncol)").mkString("\n")
      )
    val good = kept.filter(r => cols.nonEmpty || r._1.length == ncol)
    val rows = good.zipWithIndex.map { case ((fields, _), ri) =>
      val sel = selectCols(fields, cols, ri)
      sel.zipWithIndex.map { case ((f0, _), j) =>
        val f = if autostrip || !d.isString then f0.trim else f0
        if missing.contains(f.trim) then fillFor(j)
        else
          try NpIOText.parse(d, f)
          catch case _: IllegalArgumentException => fillFor(j)
      }.toArray(using d.classTag)
    }
    finish(rows, d, ndmin, unpack)

  // ------------------------------------------------------------------ raw binary

  /** `np.fromfile`: reads raw little-endian binary data (`sep = ""`) or text separated by `sep`. */
  def fromfile[T](
      file: String | Path,
      dtype: DType[T] | Null = null,
      count: Int = -1,
      sep: String = "",
      offset: Int = 0
  )(using dd: DefaultDType[T]): NDArray[T] =
    val d: DType[T] = if dtype == null then dd.dtype else dtype.asInstanceOf[DType[T]]
    val bytes = readBytes(file)
    if sep.isEmpty then rawDecode(bytes, d, count, offset)
    else
      if offset != 0 then throw new IllegalArgumentException("'offset' argument only permitted for binary files")
      val parts0 = splitSep(new String(bytes, StandardCharsets.UTF_8), sep)
      val parts = if count >= 0 then parts0.take(count) else parts0
      NDArray.fromArray(parts.map(NpIOText.parse(d, _)).toArray(using d.classTag), Array(parts.length))(using d)

  /** `np.frombuffer`: interprets little-endian bytes as a 1-D array of `dtype` (default float64). */
  def frombuffer[T](
      buffer: Array[Byte],
      dtype: DType[T] | Null = null,
      count: Int = -1,
      offset: Int = 0
  )(using dd: DefaultDType[T]): NDArray[T] =
    val d: DType[T] = if dtype == null then dd.dtype else dtype.asInstanceOf[DType[T]]
    rawDecode(buffer, d, count, offset)

  /** `ndarray.tofile`: writes the elements in C order as raw little-endian bytes (`sep = ""`)
    * or as text items formatted with `format` and separated by `sep`.
    */
  def tofile(a: NDArray[?], fid: String | Path, sep: String = "", format: String = "%s"): Unit =
    val p = toPath(fid)
    if sep.isEmpty then
      if a.dtype.isString then throw new IllegalArgumentException("cannot write str arrays as raw binary")
      Files.write(p, NpIONpy.encodeData(a, a.dtype.itemSize))
    else
      val items = rowValues(a).map(v => NpIOPyFormat.format(format, Seq(v)))
      Files.write(p, items.mkString(sep).getBytes(StandardCharsets.UTF_8))


/** `a.tofile(...)` / `a.tobytes()` method syntax. */
extension (a: NDArray[?])
  /** `ndarray.tofile`, see [[NpIO.tofile]]. */
  def tofile(fid: String | Path, sep: String = "", format: String = "%s"): Unit = np.tofile(a, fid, sep, format)
  /** `ndarray.tobytes()`. */
  def tobytes(): Array[Byte] =
    if a.dtype.isString then NpIONpy.encodeData(a, NpIONpy.descrOf(a)._2)
    else NpIONpy.encodeData(a, a.dtype.itemSize)
