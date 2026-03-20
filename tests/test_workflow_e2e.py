"""End-to-end integration tests for the complete Gauss Lean 4 workflow loop.

Each test in this module exercises *all five steps* in sequence:

  Step 1  – ``gauss``        : CLI entry point (SwarmManager singleton)
  Step 2  – ``/project``     : Gauss project initialisation (``lakefile.lean`` → manifest)
  Step 3  – ``/prove`` etc.  : Workflow command parsed to ``workflow_kind``
  Step 4  – spawn             : ``resolve_autoformalize_request`` builds the
                                ``AutoformalizeLaunchPlan`` and a SwarmManager
                                task is spawned with the plan's metadata
  Step 5  – ``/swarm``       : Task tracked, queries, and cancelled via SwarmManager

No real ``claude`` / ``git`` / ``uvx`` processes are launched — external I/O is
patched at the two seams (``_prepare_shared_bundle`` and
``_resolve_backend_runtime``) so the test can run in any CI environment.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import gauss_cli.autoformalize as autoformalize
from gauss_cli.commands import COMMANDS
from gauss_cli.project import GaussProject, initialize_gauss_project, load_gauss_project
from swarm_manager import SwarmManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_swarm():
    """Guarantee each test gets an isolated SwarmManager singleton."""
    SwarmManager.reset()
    yield
    SwarmManager.reset()


@pytest.fixture()
def lean_project(tmp_path: Path) -> GaussProject:
    """Step 1+2: create a real Lean project dir and initialise the Gauss manifest."""
    root = tmp_path / "KernelProject"
    root.mkdir(parents=True)
    (root / "lakefile.lean").write_text(
        "import Lake\nopen Lake DSL\npackage Kernel {}\n",
        encoding="utf-8",
    )
    project = initialize_gauss_project(root, name="KernelProject")
    return project


@pytest.fixture()
def _workflow_config() -> dict[str, Any]:
    return {
        "gauss": {
            "autoformalize": {
                "backend": "claude-code",
                "handoff_mode": "auto",
                "auth_mode": "auto",
            }
        }
    }


# ---------------------------------------------------------------------------
# Helper: build the SharedLeanBundle / AutoformalizeBackendRuntime stubs
# ---------------------------------------------------------------------------

def _make_bundle(
    tmp_path: Path,
    project: GaussProject,
    backend_name: str = "claude-code",
) -> autoformalize.SharedLeanBundle:
    real_home = tmp_path / "home"
    managed_root = tmp_path / backend_name / "managed"
    assets_root = tmp_path / "assets"
    startup_dir = managed_root / "startup"
    mcp_dir = managed_root / "mcp"
    plugin_source = assets_root / "lean4-skills" / "plugins" / "lean4"
    skill_source = plugin_source / "skills" / "lean4"
    scripts_root = plugin_source / "lib" / "scripts"
    references_root = skill_source / "references"
    commands_root = plugin_source / "commands"

    for path in (real_home, startup_dir, mcp_dir, scripts_root, references_root, commands_root):
        path.mkdir(parents=True, exist_ok=True)

    (skill_source / "SKILL.md").write_text("# Lean4\n", encoding="utf-8")
    (scripts_root / "prove.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (references_root / "README.md").write_text("refs\n", encoding="utf-8")
    for cmd_name in ("prove", "draft", "autoprove", "formalize", "autoformalize"):
        (commands_root / f"{cmd_name}.md").write_text(f"# {cmd_name}\n", encoding="utf-8")

    return autoformalize.SharedLeanBundle(
        backend_name=backend_name,
        managed_root=managed_root,
        assets_root=assets_root,
        startup_dir=startup_dir,
        mcp_dir=mcp_dir,
        project=project,
        project_root=project.root,
        lean_root=project.lean_root,
        active_cwd=project.root,
        real_home=real_home,
        plugin_source=plugin_source,
        skill_source=skill_source,
        scripts_root=scripts_root,
        references_root=references_root,
        uv_runner=("/usr/bin/uvx", "--from", autoformalize.LEAN_LSP_MCP_GIT_SPEC, "lean-lsp-mcp"),
    )


def _make_runtime(
    tmp_path: Path,
    bundle: autoformalize.SharedLeanBundle,
    backend_name: str = "claude-code",
) -> autoformalize.AutoformalizeBackendRuntime:
    managed_context = autoformalize.ManagedContext(
        backend_name=backend_name,
        managed_root=bundle.managed_root,
        project_root=bundle.project.root,
        lean_root=bundle.project.lean_root,
        backend_home=bundle.managed_root / "backend-home",
        plugin_root=bundle.managed_root / "backend-home" / ".claude" / "plugins" / "lean4",
        mcp_config_path=bundle.mcp_dir / "lean-lsp.mcp.json",
        startup_context_path=bundle.startup_dir / "context.md",
        assets_root=bundle.assets_root,
        project_manifest_path=bundle.project.manifest_path,
        backend_config_path=bundle.managed_root / "backend-home" / ".claude.json",
    )
    return autoformalize.AutoformalizeBackendRuntime(
        argv=["/usr/bin/claude", "--model", autoformalize.CLAUDE_MODEL],
        child_env={"HOME": str(managed_context.backend_home), "PATH": "/usr/bin"},
        managed_context=managed_context,
    )


# ---------------------------------------------------------------------------
# Helper: run the entire plan-resolution loop
# ---------------------------------------------------------------------------

def _resolve_plan(
    frontend_command: str,
    project: GaussProject,
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch,
) -> autoformalize.AutoformalizeLaunchPlan:
    """Resolve a workflow command through the full pipeline, mocking only I/O."""
    bundle = _make_bundle(tmp_path, project)
    runtime = _make_runtime(tmp_path, bundle)

    monkeypatch.setattr(autoformalize, "_require_executable", lambda n, _m, _e: f"/usr/bin/{n}")
    monkeypatch.setattr(
        autoformalize,
        "_resolve_uv_runner",
        lambda _e: ("/usr/bin/uvx", "--from", autoformalize.LEAN_LSP_MCP_GIT_SPEC, "lean-lsp-mcp"),
    )
    monkeypatch.setattr(autoformalize, "_prepare_shared_bundle", lambda **_kw: bundle)
    monkeypatch.setattr(autoformalize, "_resolve_backend_runtime", lambda **_kw: runtime)
    monkeypatch.setattr(
        autoformalize,
        "build_handoff_request",
        lambda **kw: SimpleNamespace(**kw),
    )

    return autoformalize.resolve_autoformalize_request(
        frontend_command,
        config,
        active_cwd=str(project.root),
        base_env={"HOME": str(tmp_path / "home"), "PATH": "/usr/bin"},
    )


# ---------------------------------------------------------------------------
# Helper: spawn a swarm task from a resolved plan (step 4→5)
# ---------------------------------------------------------------------------

def _spawn_from_plan(
    plan: autoformalize.AutoformalizeLaunchPlan,
) -> Any:
    """Spawn a SwarmManager task using the plan's metadata (no real subprocess)."""
    swarm = SwarmManager()
    fake_thread = MagicMock(spec=threading.Thread)
    fake_thread.start = MagicMock()

    with patch("swarm_manager._run_claude_code_interactive"):
        with patch("threading.Thread", return_value=fake_thread):
            task = swarm.spawn_interactive(
                theorem=plan.user_instruction or plan.backend_command,
                description=f"{plan.workflow_kind} in {plan.project.label}",
                argv=list(plan.handoff_request.argv),
                cwd=plan.handoff_request.cwd,
                env=dict(plan.handoff_request.env),
                workflow_kind=plan.workflow_kind,
                workflow_command=plan.backend_command,
                project_name=plan.project.label,
                project_root=str(plan.project.root),
                backend_name=plan.managed_context.backend_name,
            )
    return task


# ===========================================================================
# Main end-to-end tests — one per workflow kind
# ===========================================================================

class TestCompleteWorkflowLoop:
    """Run the full 5-step Gauss workflow for each of the five commands."""

    @pytest.mark.parametrize(
        ("frontend", "expected_kind", "expected_backend_prefix"),
        [
            ("/prove", "prove", "/lean4:prove"),
            ("/draft", "draft", "/lean4:draft"),
            ("/autoprove", "autoprove", "/lean4:autoprove"),
            ("/formalize", "formalize", "/lean4:formalize"),
            ("/autoformalize", "autoformalize", "/lean4:autoformalize"),
        ],
    )
    def test_complete_loop_for_each_workflow_kind(
        self,
        frontend: str,
        expected_kind: str,
        expected_backend_prefix: str,
        lean_project: GaussProject,
        _workflow_config: dict[str, Any],
        tmp_path: Path,
        monkeypatch,
    ):
        """Step 1–5: project → command parse → plan → spawn → swarm query.

        This test exercises the complete workflow end-to-end for one workflow kind.
        """
        # ── Step 1: CLI is running (SwarmManager singleton is fresh)
        swarm = SwarmManager()
        assert len(swarm.tasks) == 0, "Swarm should start empty"

        # ── Step 2: /project — the lean_project fixture has already created
        #             a real project with lakefile.lean + .gauss/project.yaml
        project = lean_project
        assert project.manifest_path.is_file(), "Project manifest must exist"
        assert (project.lean_root / "lakefile.lean").is_file(), "lakefile.lean must exist"

        # ── Step 3: command is registered in the CLI surface
        assert frontend in COMMANDS, f"{frontend} must be a registered slash command"

        # ── Step 4: resolve the workflow command → AutoformalizeLaunchPlan
        plan = _resolve_plan(frontend, project, tmp_path, _workflow_config, monkeypatch)

        assert plan.workflow_kind == expected_kind
        assert plan.frontend_command == frontend
        assert plan.backend_command.startswith(expected_backend_prefix), (
            f"Backend command {plan.backend_command!r} does not start with {expected_backend_prefix!r}"
        )
        assert plan.project.root == project.root
        assert plan.managed_context.backend_name == "claude-code"

        staged = plan.staged_paths()
        assert staged["workflow_kind"] == expected_kind
        assert staged["project_root"] == str(project.root)

        # ── Step 4b: spawn the swarm task from the plan
        task = _spawn_from_plan(plan)

        assert task.task_id.startswith("af-"), f"Task ID {task.task_id!r} should start with af-"
        assert task.workflow_kind == expected_kind
        assert task.workflow_command == plan.backend_command
        assert task.project_name == project.label
        assert task.project_root == str(project.root)
        assert task.backend_name == "claude-code"

        # ── Step 5: /swarm — task is tracked and can be queried
        assert swarm.get_task(task.task_id) is task, "SwarmManager must track the spawned task"
        assert len(swarm.tasks) == 1

        # Simulate the task becoming active and cancellable
        task.status = "running"
        assert swarm.cancel(task.task_id) is True
        assert task.status == "cancelled"

        # After cancel the task is still in the registry (it just changes status)
        assert swarm.get_task(task.task_id) is task


# ===========================================================================
# Workflow with extra arguments
# ===========================================================================

class TestCompleteWorkflowWithArgs:
    """Verify extra CLI arguments survive the full pipeline."""

    @pytest.mark.parametrize(
        ("command", "expected_kind", "expected_arg"),
        [
            ("/prove --repair-only Main.lean", "prove", "--repair-only"),
            ("/draft Section3", "draft", "Section3"),
            ("/autoprove --max-cycles=4", "autoprove", "--max-cycles=4"),
            ("/formalize --source paper.pdf", "formalize", "--source"),
            ("/autoformalize --claim-select=first --out=Proof.lean", "autoformalize", "--claim-select=first"),
        ],
    )
    def test_arguments_preserved_through_complete_loop(
        self,
        command: str,
        expected_kind: str,
        expected_arg: str,
        lean_project: GaussProject,
        _workflow_config: dict[str, Any],
        tmp_path: Path,
        monkeypatch,
    ):
        """Arguments passed after the command name reach the backend command."""
        plan = _resolve_plan(command, lean_project, tmp_path, _workflow_config, monkeypatch)

        assert plan.workflow_kind == expected_kind
        assert expected_arg in plan.backend_command, (
            f"Expected {expected_arg!r} in backend_command {plan.backend_command!r}"
        )

        task = _spawn_from_plan(plan)
        # The task theorem/description encodes the backend command (with args)
        assert expected_arg in task.theorem or expected_arg in task.workflow_command, (
            f"Expected {expected_arg!r} in task metadata"
        )


# ===========================================================================
# Project enforcement
# ===========================================================================

class TestCompleteWorkflowProjectEnforcement:
    """The workflow fails safely when no active Gauss project exists."""

    @pytest.mark.parametrize("frontend", ["/prove", "/draft", "/autoprove", "/formalize", "/autoformalize"])
    def test_no_project_raises_preflight_error_before_spawn(
        self,
        frontend: str,
        _workflow_config: dict[str, Any],
        tmp_path: Path,
        monkeypatch,
    ):
        """If /project was never run, the workflow raises before touching the swarm."""
        empty_dir = tmp_path / "no-project"
        empty_dir.mkdir()

        monkeypatch.setattr(autoformalize, "_require_executable", lambda n, _m, _e: f"/usr/bin/{n}")
        monkeypatch.setattr(
            autoformalize,
            "_resolve_uv_runner",
            lambda _e: ("/usr/bin/uvx", "--from", autoformalize.LEAN_LSP_MCP_GIT_SPEC, "lean-lsp-mcp"),
        )

        with pytest.raises(autoformalize.AutoformalizePreflightError, match=r"/project"):
            autoformalize.resolve_autoformalize_request(
                frontend,
                _workflow_config,
                active_cwd=str(empty_dir),
                base_env={"HOME": str(tmp_path / "home"), "PATH": "/usr/bin"},
            )

        # Swarm must remain empty — no task should have been queued
        assert len(SwarmManager().tasks) == 0, "Swarm must not be polluted by a failed workflow launch"


# ===========================================================================
# Multi-agent swarm scenario
# ===========================================================================

class TestMultiAgentSwarmSession:
    """Run all five workflow kinds in one session and verify independent tracking."""

    def test_all_five_workflow_agents_tracked_in_same_session(
        self,
        lean_project: GaussProject,
        _workflow_config: dict[str, Any],
        tmp_path: Path,
        monkeypatch,
    ):
        """Launch all five workflow kinds and confirm each gets its own swarm entry."""
        swarm = SwarmManager()

        commands = [
            ("/prove", "prove"),
            ("/draft", "draft"),
            ("/autoprove", "autoprove"),
            ("/formalize", "formalize"),
            ("/autoformalize", "autoformalize"),
        ]

        plans = [
            _resolve_plan(frontend, lean_project, tmp_path, _workflow_config, monkeypatch)
            for frontend, _ in commands
        ]

        tasks = [_spawn_from_plan(plan) for plan in plans]

        # All five tasks are tracked independently
        assert len(swarm.tasks) == 5
        assert len({t.task_id for t in tasks}) == 5, "Each task must have a unique ID"

        # Task IDs follow the sequential af-001…af-005 pattern
        assert [t.task_id for t in tasks] == ["af-001", "af-002", "af-003", "af-004", "af-005"]

        # Each task records the correct workflow kind
        for task, (_, expected_kind) in zip(tasks, commands):
            assert task.workflow_kind == expected_kind, (
                f"Task {task.task_id} has workflow_kind={task.workflow_kind!r}, expected {expected_kind!r}"
            )

        # Cancelling one task does not affect the others
        tasks[0].status = "running"
        swarm.cancel(tasks[0].task_id)
        assert tasks[0].status == "cancelled"
        for other in tasks[1:]:
            assert other.status != "cancelled", f"Task {other.task_id} should not be cancelled"

        # /swarm can retrieve every task individually
        for task in tasks:
            assert swarm.get_task(task.task_id) is task
