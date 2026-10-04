"""System tools return structured, real host data (SAFE, psutil-backed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arthur.config.schema import Config
from arthur.security.paths import PathPolicy
from arthur.tools.base import ToolContext
from arthur.tools.system import (
    CpuInfoTool,
    DiskUsageTool,
    MemoryInfoTool,
    NetworkInfoTool,
    ProcessListTool,
    SystemInfoTool,
    UptimeTool,
    human_bytes,
)


@pytest.fixture()
def ctx(tmp_path: Path) -> ToolContext:
    config = Config()
    config.security.allowed_roots = [str(tmp_path)]
    return ToolContext(config=config, paths=PathPolicy(config.security))


def test_human_bytes() -> None:
    assert human_bytes(512) == "512.0 B"
    assert human_bytes(2048) == "2.0 KB"
    assert human_bytes(5 * 1024**3) == "5.0 GB"


def test_system_info(ctx: ToolContext) -> None:
    result = SystemInfoTool().run(SystemInfoTool.args_model(), ctx)
    assert result.ok
    data = result.data
    assert data["hostname"]
    assert data["kernel"]
    assert data["python"].startswith("3.")
    assert isinstance(data["cpu_logical"], int)


def test_cpu_info(ctx: ToolContext) -> None:
    result = CpuInfoTool().run(CpuInfoTool.args_model(sample_seconds=0.05), ctx)
    data = result.data
    assert data["model"]
    assert 0 <= data["usage_percent"] <= 100
    assert len(data["per_core_percent"]) >= 1


def test_memory_info(ctx: ToolContext) -> None:
    data = MemoryInfoTool().run(MemoryInfoTool.args_model(), ctx).data
    assert data["ram_total"].endswith(("KB", "MB", "GB", "TB"))
    assert 0 <= data["ram_used_percent"] <= 100


def test_disk_usage(ctx: ToolContext) -> None:
    data = DiskUsageTool().run(DiskUsageTool.args_model(mount="/"), ctx).data
    assert data["used_percent"] >= 0
    assert isinstance(data["partitions"], list)


def test_process_list(ctx: ToolContext) -> None:
    data = ProcessListTool().run(ProcessListTool.args_model(limit=5), ctx).data
    processes = data["processes"]
    assert 1 <= len(processes) <= 5
    assert all("pid" in proc and "name" in proc for proc in processes)


def test_network_info(ctx: ToolContext) -> None:
    data = NetworkInfoTool().run(NetworkInfoTool.args_model(), ctx).data
    assert isinstance(data["interfaces"], dict)
    assert data["bytes_sent"] >= 0


def test_uptime(ctx: ToolContext) -> None:
    data = UptimeTool().run(UptimeTool.args_model(), ctx).data
    assert data["uptime_seconds"] > 0
    assert "d" in data["uptime_human"]


def test_results_are_json_serializable(ctx: ToolContext) -> None:
    for tool_args in (
        SystemInfoTool(),
        MemoryInfoTool(),
        NetworkInfoTool(),
        UptimeTool(),
    ):
        result = tool_args.run(tool_args.args_model(), ctx)
        json.dumps(result.data, default=str)
