"""
Local SparkSession factory with Delta Lake support.

On Windows, Spark's bundled Hadoop client needs winutils.exe/hadoop.dll to
do basic file operations - this module points HADOOP_HOME at a
version-matched copy bundled in tools/ so no system-wide install or env var
is required. No-op on other platforms.
"""

from __future__ import annotations

import os
import platform
import sys

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from music_man.paths import PROJECT_ROOT

HADOOP_HOME = PROJECT_ROOT / "tools" / "hadoop-3.3.6"
JAVA_HOME = PROJECT_ROOT / "tools" / "jdk-17.0.20.1+1"


def _configure_windows_hadoop() -> None:
    if platform.system() != "Windows":
        return
    if not (HADOOP_HOME / "bin" / "winutils.exe").exists():
        raise FileNotFoundError(
            f"winutils.exe not found at {HADOOP_HOME / 'bin'} - see README for setup."
        )
    # The system java found on PATH may be a 32-bit JVM, which can never
    # load the 64-bit hadoop.dll/winutils.exe above - use a bundled 64-bit
    # JDK instead so this doesn't depend on whatever Java happens to be
    # installed system-wide.
    if not (JAVA_HOME / "bin" / "java.exe").exists():
        raise FileNotFoundError(
            f"Bundled JDK not found at {JAVA_HOME} - see README for setup."
        )
    os.environ["JAVA_HOME"] = str(JAVA_HOME)
    os.environ["HADOOP_HOME"] = str(HADOOP_HOME)
    os.environ["hadoop.home.dir"] = str(HADOOP_HOME)
    for bin_dir in (str(JAVA_HOME / "bin"), str(HADOOP_HOME / "bin")):
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def get_spark(app_name: str = "music_man") -> SparkSession:
    """Returns a local SparkSession configured for Delta Lake reads/writes."""
    _configure_windows_hadoop()

    # PYSPARK_PYTHON: ensures worker subprocesses use this venv's interpreter.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

    # JAVA_TOOL_OPTIONS is picked up by every JVM at launch unconditionally
    # (unlike spark.driver.extraJavaOptions, which local/interactive mode
    # launches too late to honor). Needed for two Windows-specific issues:
    # preferIPv4Stack (JVM<->Python worker loopback handshake resets without
    # it) and java.library.path (Delta's read path loads hadoop.dll via the
    # JVM's native library path, not the OS PATH env var).
    os.environ["JAVA_TOOL_OPTIONS"] = (
        "-Djava.net.preferIPv4Stack=true "
        f"-Djava.library.path={HADOOP_HOME / 'bin'}"
    )

    # spark.driver.memory set via .config() is silently ignored in local
    # mode - the driver JVM is already running by the time SparkConf is
    # applied. --driver-memory via PYSPARK_SUBMIT_ARGS is the documented
    # way to size it before that JVM launches.
    os.environ.setdefault("PYSPARK_SUBMIT_ARGS", "--driver-memory 6g pyspark-shell")

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()
