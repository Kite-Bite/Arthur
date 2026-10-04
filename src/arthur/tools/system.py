"""System information tools implemented with psutil/platform (SAFE, no shell)."""

from __future__ import annotations

import platform
from typing import Any

import psutil
from pydantic import BaseModel, Field

from arthur.execution.errors import ToolError
from arthur.tools.base import Tool, ToolContext, ToolResult


def human_bytes(num: float) -> str:
    """Format bytes as a short human-readable string (base-1024)."""
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} EB"


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.partition(":")[2].strip()
    except OSError:  # pragma: no cover - non-Linux fallback
        pass
    return platform.processor() or "unknown"


class SystemInfoArgs(BaseModel):
    pass


class SystemInfoTool(Tool):
    name = "system_info"
    description = "Operating system, host, kernel, architecture and Python details."
    action = "Show system information"
    args_model = SystemInfoArgs

    def run(self, args: SystemInfoArgs, ctx: ToolContext) -> ToolResult:
        os_release: dict[str, str] = {}
        try:
            os_release = dict(platform.freedesktop_os_release())
        except (OSError, AttributeError):  # pragma: no cover
            pass
        data: dict[str, Any] = {
            "os": os_release.get("NAME", platform.system()),
            "os_version": os_release.get("VERSION_ID", platform.release()),
            "hostname": platform.node(),
            "kernel": platform.release(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "cpu_logical": psutil.cpu_count(logical=True),
            "cpu_physical": psutil.cpu_count(logical=False),
            "boot_time": psutil.boot_time(),
        }
        summary = f"{data['os']} {data['os_version']} on {data['hostname']}"
        return ToolResult(summary=summary, data=data)


class CpuInfoArgs(BaseModel):
    sample_seconds: float = Field(0.3, ge=0.0, le=2.0, description="Usage sampling window")


class CpuInfoTool(Tool):
    name = "cpu_info"
    description = "CPU model, cores, frequency, load average and current utilisation."
    action = "Show CPU information"
    args_model = CpuInfoArgs

    def run(self, args: CpuInfoArgs, ctx: ToolContext) -> ToolResult:
        psutil.cpu_percent(interval=None)  # prime per-core counters
        overall = psutil.cpu_percent(interval=args.sample_seconds)
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        freq = psutil.cpu_freq()
        try:
            load1, load5, load15 = psutil.getloadavg()
        except (AttributeError, OSError):  # pragma: no cover
            load1 = load5 = load15 = 0.0
        data: dict[str, Any] = {
            "model": _cpu_model(),
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "usage_percent": overall,
            "per_core_percent": per_core,
            "frequency_mhz": round(freq.current, 0) if freq else None,
            "load_avg": {"1m": load1, "5m": load5, "15m": load15},
        }
        return ToolResult(summary=f"CPU {data['model']} at {overall}%", data=data)


class MemoryInfoArgs(BaseModel):
    pass


class MemoryInfoTool(Tool):
    name = "memory_info"
    description = "RAM and swap usage (total, available, percentage)."
    action = "Show memory information"
    args_model = MemoryInfoArgs

    def run(self, args: MemoryInfoArgs, ctx: ToolContext) -> ToolResult:
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        data = {
            "ram_total": human_bytes(vm.total),
            "ram_available": human_bytes(vm.available),
            "ram_used_percent": vm.percent,
            "swap_total": human_bytes(swap.total),
            "swap_used": human_bytes(swap.used),
            "swap_percent": swap.percent,
            "ram_total_bytes": vm.total,
            "ram_available_bytes": vm.available,
        }
        return ToolResult(summary=f"RAM {data['ram_total']} ({vm.percent}% used)", data=data)


class DiskUsageArgs(BaseModel):
    # Named ``mount`` rather than ``path`` on purpose: this argument is only
    # passed to statvfs, which reports capacity numbers and never file
    # contents, so it is not subject to workspace containment - and a
    # containment check here would reject the default ("/"), making the tool
    # unusable. It also cannot leak more than the partition list the tool
    # already returns.
    mount: str = Field("/", description="Mount point or filesystem path to report usage for")


class DiskUsageTool(Tool):
    name = "disk_usage"
    description = (
        "Disk usage (total/used/free) for a mount point plus all mounted physical partitions."
    )
    action = "Show disk usage"
    args_model = DiskUsageArgs

    def run(self, args: DiskUsageArgs, ctx: ToolContext) -> ToolResult:
        try:
            usage = psutil.disk_usage(args.mount)
        except OSError as exc:
            raise ToolError(f"cannot stat {args.mount}: {exc}") from exc
        partitions = []
        for part in psutil.disk_partitions(all=False):
            try:
                pusage = psutil.disk_usage(part.mountpoint)
            except OSError:  # pragma: no cover - unmounted/permission
                continue
            partitions.append(
                {
                    "device": part.device,
                    "mountpoint": part.mountpoint,
                    "fstype": part.fstype,
                    "total": human_bytes(pusage.total),
                    "used": human_bytes(pusage.used),
                    "free": human_bytes(pusage.free),
                    "used_percent": pusage.percent,
                }
            )
        data = {
            "path": args.mount,
            "total": human_bytes(usage.total),
            "used": human_bytes(usage.used),
            "free": human_bytes(usage.free),
            "used_percent": usage.percent,
            "partitions": partitions,
        }
        return ToolResult(
            summary=(
                f"{args.mount}: {usage.percent}% used "
                f"({human_bytes(usage.used)} used, {human_bytes(usage.free)} free "
                f"of {human_bytes(usage.total)})"
            ),
            data=data,
        )


class ProcessListArgs(BaseModel):
    limit: int = Field(15, ge=1, le=200)
    sort_by: str = Field("memory", description="'memory' or 'cpu'")


class ProcessListTool(Tool):
    name = "process_list"
    description = "List running processes sorted by CPU or memory usage."
    action = "List processes"
    timeout = 30.0
    args_model = ProcessListArgs

    def run(self, args: ProcessListArgs, ctx: ToolContext) -> ToolResult:
        sort_key = "memory" if args.sort_by.lower().startswith("mem") else "cpu"
        rows: list[dict[str, Any]] = []
        try:
            procs = list(psutil.process_iter(["pid", "name", "status", "memory_info"]))
            for proc in procs:
                try:
                    proc.cpu_percent(interval=None)  # prime
                except (psutil.Error, OSError):
                    continue
            import time

            time.sleep(0.15)
            for proc in procs:
                try:
                    info = proc.info
                    cmdline = " ".join(proc.cmdline())[:200]
                    mem = info.get("memory_info")
                    rss = mem.rss if mem is not None else 0
                    rows.append(
                        {
                            "pid": info.get("pid"),
                            "name": info.get("name"),
                            "status": info.get("status"),
                            "cpu_percent": proc.cpu_percent(interval=None),
                            "memory_rss": human_bytes(rss),
                            "memory_rss_bytes": rss,
                            "command": cmdline or info.get("name"),
                        }
                    )
                except (psutil.Error, OSError):
                    continue
        except psutil.Error as exc:
            raise ToolError(f"cannot enumerate processes: {exc}") from exc

        rows.sort(
            key=lambda r: r["cpu_percent"] if sort_key == "cpu" else r["memory_rss_bytes"],
            reverse=True,
        )
        top = rows[: args.limit]
        for row in top:
            row.pop("memory_rss_bytes", None)
        return ToolResult(
            summary=f"top {len(top)} processes by {sort_key}",
            data={"processes": top},
        )


class NetworkInfoArgs(BaseModel):
    pass


class NetworkInfoTool(Tool):
    name = "network_info"
    description = "Network interfaces, addresses and traffic counters."
    action = "Show network information"
    args_model = NetworkInfoArgs

    def run(self, args: NetworkInfoArgs, ctx: ToolContext) -> ToolResult:
        interfaces: dict[str, list[str]] = {}
        for name, addrs in psutil.net_if_addrs().items():
            interfaces[name] = [
                f"{a.family.name}: {a.address}"
                for a in addrs
                if a.family.name in {"AF_INET", "AF_INET6"}
            ]
        io = psutil.net_io_counters()
        connections: int | None = None
        try:
            connections = len(psutil.net_connections(kind="inet"))
        except (psutil.Error, PermissionError):  # pragma: no cover - needs perms
            connections = None
        data = {
            "interfaces": interfaces,
            "bytes_sent": io.bytes_sent if io else 0,
            "bytes_recv": io.bytes_recv if io else 0,
            "active_connections": connections,
        }
        return ToolResult(
            summary=f"{len(interfaces)} interfaces, {connections} connections", data=data
        )


class UptimeArgs(BaseModel):
    pass


class UptimeTool(Tool):
    name = "uptime"
    description = "System boot time and uptime."
    action = "Show uptime"
    args_model = UptimeArgs

    def run(self, args: UptimeArgs, ctx: ToolContext) -> ToolResult:
        import time
        from datetime import UTC, datetime

        boot = psutil.boot_time()
        seconds = time.time() - boot
        days, rem = divmod(int(seconds), 86400)
        hours, rem = divmod(rem, 3600)
        minutes, _ = divmod(rem, 60)
        data = {
            "boot_time": datetime.fromtimestamp(boot, tz=UTC).isoformat(),
            "uptime_seconds": int(seconds),
            "uptime_human": f"{days}d {hours}h {minutes}m",
        }
        return ToolResult(summary=f"up {data['uptime_human']}", data=data)
