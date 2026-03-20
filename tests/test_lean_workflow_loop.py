"""Tests for the Gauss Lean 4 core workflow loop.

Validates the sequence described in the CLI workflow:
  1. Start CLI with ``gauss``
  2. Create or select the active project with ``/project``
  3. Launch ``/prove``, ``/draft``, ``/autoprove``, ``/formalize``, or
     ``/autoformalize``
  4. Gauss spawns a managed backend child session that runs the corresponding
     lean4-skills workflow command in the active project
  5. Use ``/swarm`` to track or reattach to running workflow agents

These tests confirm the routing, project enforcement, command-kind mapping,
and swarm-task lifecycle that tie those pieces together.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import gauss_cli.autoformalize as autoformalize
from gauss_cli.commands import COMMANDS
from gauss_cli.project import initialize_gauss_project
from swarm_manager import SwarmManager, SwarmTask


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lean_project(tmp_path: Path, name: str = "TestProject") -> Path:
    root = tmp_path / name
    root.mkdir(parents=True)
    (root / "lakefile.lean").write_text("-- lean project\n", encoding="utf-8")
    initialize_gauss_project(root, name=name)
    return root


def _config() -> dict:
    return {
        "gauss": {
            "autoformalize": {
                "backend": "claude-code",
                "handoff_mode": "auto",
                "auth_mode": "auto",
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — /project: Gauss project creation and discovery
# ─────────────────────────────────────────────────────────────────────────────

class TestProjectStep:
    """``/project`` creates or locates the active Lean 4 project."""

    def test_slash_project_registered_in_command_surface(self):
        """``/project`` is a first-class slash command in the COMMANDS registry."""
        assert "/project" in COMMANDS

    def test_project_init_creates_manifest_at_lean_root(self, tmp_path: Path):
        """``/project init`` produces a readable manifest under the Lean root."""
        root = tmp_path / "MyProof"
        root.mkdir()
        (root / "lakefile.lean").write_text("-- lean\n", encoding="utf-8")

        project = initialize_gauss_project(root, name="MyProof")

        assert project.manifest_path.is_file()
        assert project.name == "MyProof"
        assert project.lean_root == root

    def test_project_paths_are_under_gauss_dir(self, tmp_path: Path):
        """Runtime, cache, and workflow dirs all live inside ``.gauss/``."""
        root = _lean_project(tmp_path)
        from gauss_cli.project import load_gauss_project

        project = load_gauss_project(root)
        gauss_dir = root / ".gauss"

        assert project.runtime_dir.is_relative_to(gauss_dir)
        assert project.cache_dir.is_relative_to(gauss_dir)
        assert project.workflows_dir.is_relative_to(gauss_dir)

    def test_project_without_lean_root_raises_error(self, tmp_path: Path):
        """``/project init`` fails when no ``lakefile.lean`` can be found."""
        empty = tmp_path / "Empty"
        empty.mkdir()

        from gauss_cli.project import ProjectCommandError

        with pytest.raises(ProjectCommandError):
            initialize_gauss_project(empty)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — workflow commands registered and correctly routed
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkflowCommandRegistry:
    """All five workflow commands are registered in the slash-command surface."""

    @pytest.mark.parametrize("cmd", ["/prove", "/draft", "/autoprove", "/formalize", "/autoformalize"])
    def test_workflow_command_in_commands_registry(self, cmd: str):
        """Each workflow command appears in the shared COMMANDS dict."""
        assert cmd in COMMANDS, f"Missing slash command: {cmd}"

    def test_all_workflow_commands_mention_backend_agent(self):
        """Each workflow entry's description references a managed backend agent."""
        workflow_cmds = ["/prove", "/draft", "/autoprove", "/formalize", "/autoformalize"]
        for cmd in workflow_cmds:
            desc = COMMANDS[cmd].lower()
            assert "managed backend agent" in desc, (
                f"{cmd} description does not mention 'managed backend agent': {COMMANDS[cmd]!r}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — command parsing: each command maps to the right workflow_kind
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkflowCommandParsing:
    """``_parse_managed_workflow_command`` extracts the correct ``workflow_kind``."""

    @pytest.mark.parametrize(
        ("command", "expected_kind", "expected_backend_prefix"),
        [
            ("/prove", "prove", "/lean4:prove"),
            ("/prove Fermat.lean", "prove", "/lean4:prove"),
            ("/draft", "draft", "/lean4:draft"),
            ("/draft Section3", "draft", "/lean4:draft"),
            ("/autoprove", "autoprove", "/lean4:autoprove"),
            ("/auto_proof Main.lean", "autoprove", "/lean4:autoprove"),
            ("/formalize", "formalize", "/lean4:formalize"),
            ("/formalize --source paper.pdf", "formalize", "/lean4:formalize"),
            ("/autoformalize", "autoformalize", "/lean4:autoformalize"),
            ("/auto_formalize --claim-select=first", "autoformalize", "/lean4:autoformalize"),
        ],
    )
    def test_workflow_kind_and_backend_command(
        self, command: str, expected_kind: str, expected_backend_prefix: str
    ):
        spec = autoformalize._parse_managed_workflow_command(command)
        assert spec.workflow_kind == expected_kind
        assert spec.backend_command.startswith(expected_backend_prefix)

    def test_interactive_commands_group(self):
        """``/prove``, ``/draft``, ``/formalize`` all parse as distinct interactive kinds."""
        kinds = {
            autoformalize._parse_managed_workflow_command(cmd).workflow_kind
            for cmd in ("/prove", "/draft", "/formalize")
        }
        assert kinds == {"prove", "draft", "formalize"}

    def test_auto_commands_group(self):
        """``/autoprove`` and ``/autoformalize`` parse as auto kinds."""
        kinds = {
            autoformalize._parse_managed_workflow_command(cmd).workflow_kind
            for cmd in ("/autoprove", "/autoformalize")
        }
        assert kinds == {"autoprove", "autoformalize"}

    def test_workflow_args_preserved_in_backend_command(self):
        """Extra arguments are appended to the backend command."""
        spec = autoformalize._parse_managed_workflow_command("/prove --repair-only MyLemma.lean")
        assert "--repair-only" in spec.backend_command
        assert "MyLemma.lean" in spec.backend_command

    def test_unknown_command_raises(self):
        """An unrecognised prefix raises ``AutoformalizeUsageError``."""
        with pytest.raises(autoformalize.AutoformalizeUsageError):
            autoformalize._parse_managed_workflow_command("/unknown-command")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — project required: workflow commands fail without an active project
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkflowRequiresProject:
    """Workflow commands raise ``AutoformalizePreflightError`` when no project exists."""

    @pytest.mark.parametrize("cmd", ["/prove", "/draft", "/autoprove", "/formalize", "/autoformalize"])
    def test_raises_when_no_project(self, monkeypatch, tmp_path: Path, cmd: str):
        """Each workflow command requires a `.gauss/project.yaml` to be present."""
        empty_dir = tmp_path / "no-project"
        empty_dir.mkdir()

        monkeypatch.setattr(
            autoformalize, "_require_executable", lambda name, _msg, _env: f"/usr/bin/{name}"
        )
        monkeypatch.setattr(
            autoformalize,
            "_resolve_uv_runner",
            lambda _env: ("/usr/bin/uvx", "--from", autoformalize.LEAN_LSP_MCP_GIT_SPEC, "lean-lsp-mcp"),
        )

        with pytest.raises(autoformalize.AutoformalizePreflightError, match=r"/project"):
            autoformalize.resolve_autoformalize_request(
                cmd,
                _config(),
                active_cwd=str(empty_dir),
                base_env={"HOME": str(tmp_path / "home"), "PATH": "/usr/bin"},
            )


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — project resolves: workflow plan built from active project
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkflowPlanBuiltFromProject:
    """``resolve_autoformalize_request`` carries project metadata into the plan."""

    def test_plan_carries_project_root_and_lean_root(self, monkeypatch, tmp_path: Path):
        """`plan.project` is the discovered project, not a stub."""
        project_root = _lean_project(tmp_path, "ProofProject")

        shared = _make_shared_bundle(tmp_path, project_root)
        runtime = _make_runtime(tmp_path, shared)

        monkeypatch.setattr(autoformalize, "_require_executable", lambda n, _m, _e: f"/usr/bin/{n}")
        monkeypatch.setattr(
            autoformalize,
            "_resolve_uv_runner",
            lambda _e: ("/usr/bin/uvx", "--from", autoformalize.LEAN_LSP_MCP_GIT_SPEC, "lean-lsp-mcp"),
        )
        monkeypatch.setattr(autoformalize, "_prepare_shared_bundle", lambda **_kw: shared)
        monkeypatch.setattr(autoformalize, "_resolve_backend_runtime", lambda **_kw: runtime)
        monkeypatch.setattr(
            autoformalize,
            "build_handoff_request",
            lambda **kw: SimpleNamespace(**kw),
        )

        plan = autoformalize.resolve_autoformalize_request(
            "/prove",
            _config(),
            active_cwd=str(project_root),
            base_env={"HOME": str(tmp_path / "home"), "PATH": "/usr/bin"},
        )

        assert plan.project.root == project_root
        assert plan.workflow_kind == "prove"
        assert plan.frontend_command == "/prove"
        assert plan.backend_command.startswith("/lean4:prove")

    def test_plan_backend_command_uses_lean4_prefix(self, monkeypatch, tmp_path: Path):
        """The lean4-skills CLI receives ``/lean4:<workflow>`` as the backend command."""
        project_root = _lean_project(tmp_path, "AutoProveProject")
        shared = _make_shared_bundle(tmp_path, project_root)
        runtime = _make_runtime(tmp_path, shared)

        monkeypatch.setattr(autoformalize, "_require_executable", lambda n, _m, _e: f"/usr/bin/{n}")
        monkeypatch.setattr(
            autoformalize,
            "_resolve_uv_runner",
            lambda _e: ("/usr/bin/uvx", "--from", autoformalize.LEAN_LSP_MCP_GIT_SPEC, "lean-lsp-mcp"),
        )
        monkeypatch.setattr(autoformalize, "_prepare_shared_bundle", lambda **_kw: shared)
        monkeypatch.setattr(autoformalize, "_resolve_backend_runtime", lambda **_kw: runtime)
        monkeypatch.setattr(
            autoformalize,
            "build_handoff_request",
            lambda **kw: SimpleNamespace(**kw),
        )

        plan = autoformalize.resolve_autoformalize_request(
            "/autoformalize --source paper.pdf",
            _config(),
            active_cwd=str(project_root),
            base_env={"HOME": str(tmp_path / "home"), "PATH": "/usr/bin"},
        )

        assert plan.backend_command == "/lean4:autoformalize --source paper.pdf"


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — /swarm: tracking and reattaching to running workflow agents
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_swarm():
    """Each test gets a fresh SwarmManager singleton."""
    SwarmManager.reset()
    yield
    SwarmManager.reset()


class TestSwarmStep:
    """`/swarm` tracks and manages spawned workflow tasks."""

    def test_swarm_registered_in_command_surface(self):
        """``/swarm`` is a first-class slash command in the COMMANDS registry."""
        assert "/swarm" in COMMANDS

    def test_spawn_creates_task_with_workflow_metadata(self):
        """A spawned task records the ``workflow_kind`` and ``project_name``."""
        swarm = SwarmManager()

        fake_thread = MagicMock(spec=threading.Thread)
        fake_thread.start = MagicMock()

        with patch("swarm_manager._run_claude_code_interactive"):
            with patch("threading.Thread", return_value=fake_thread):
                task = swarm.spawn_interactive(
                    theorem="mu^8 = 1",
                    description="prove mu 8-cycle",
                    argv=["/usr/bin/claude"],
                    cwd="/tmp",
                    env={"HOME": "/tmp", "PATH": "/usr/bin"},
                    workflow_kind="prove",
                    workflow_command="/lean4:prove",
                    project_name="KernelProject",
                    project_root="/tmp/KernelProject",
                    backend_name="claude-code",
                )

        assert task.task_id == "af-001"
        assert task.workflow_kind == "prove"
        assert task.workflow_command == "/lean4:prove"
        assert task.project_name == "KernelProject"
        assert task.backend_name == "claude-code"

    def test_get_task_returns_spawned_task(self):
        """``/swarm <id>`` retrieves a specific task by its task-ID."""
        swarm = SwarmManager()

        fake_thread = MagicMock(spec=threading.Thread)
        fake_thread.start = MagicMock()

        with patch("swarm_manager._run_claude_code_interactive"):
            with patch("threading.Thread", return_value=fake_thread):
                task = swarm.spawn_interactive(
                    theorem="test",
                    description="test task",
                    argv=["/usr/bin/claude"],
                    cwd="/tmp",
                    env={"HOME": "/tmp", "PATH": "/usr/bin"},
                )

        retrieved = swarm.get_task(task.task_id)
        assert retrieved is task

    def test_multiple_workflow_tasks_tracked_independently(self):
        """Multiple spawned agents are tracked separately in the swarm."""
        swarm = SwarmManager()

        fake_thread = MagicMock(spec=threading.Thread)
        fake_thread.start = MagicMock()

        with patch("swarm_manager._run_claude_code_interactive"):
            with patch("threading.Thread", return_value=fake_thread):
                t1 = swarm.spawn_interactive(
                    theorem="A",
                    description="prove A",
                    argv=["/usr/bin/claude"],
                    cwd="/tmp",
                    env={},
                    workflow_kind="prove",
                )
                t2 = swarm.spawn_interactive(
                    theorem="B",
                    description="formalize B",
                    argv=["/usr/bin/claude"],
                    cwd="/tmp",
                    env={},
                    workflow_kind="formalize",
                )

        assert swarm.get_task(t1.task_id) is t1
        assert swarm.get_task(t2.task_id) is t2
        assert t1.task_id != t2.task_id
        assert len(swarm.tasks) == 2

    def test_cancel_transitions_task_to_cancelled(self):
        """``/swarm cancel <id>`` changes the task status to ``cancelled``."""
        swarm = SwarmManager()

        fake_thread = MagicMock(spec=threading.Thread)
        fake_thread.start = MagicMock()
        fake_thread.is_alive = MagicMock(return_value=True)

        with patch("swarm_manager._run_claude_code_interactive"):
            with patch("threading.Thread", return_value=fake_thread):
                task = swarm.spawn_interactive(
                    theorem="test",
                    description="test",
                    argv=["/usr/bin/claude"],
                    cwd="/tmp",
                    env={},
                )

        task.status = "running"
        result = swarm.cancel(task.task_id)
        assert result is True
        assert task.status == "cancelled"

    def test_cancel_unknown_task_returns_false(self):
        """Cancelling a non-existent task-ID returns False (no crash)."""
        swarm = SwarmManager()
        assert swarm.cancel("af-999") is False

    def test_status_bar_shows_active_workflow_count(self):
        """The swarm status bar fragment shows the running task count when tasks are active."""
        swarm = SwarmManager()

        fake_thread = MagicMock(spec=threading.Thread)
        fake_thread.start = MagicMock()
        fake_thread.is_alive = MagicMock(return_value=True)

        with patch("swarm_manager._run_claude_code_interactive"):
            with patch("threading.Thread", return_value=fake_thread):
                task = swarm.spawn_interactive(
                    theorem="test",
                    description="test",
                    argv=["/usr/bin/claude"],
                    cwd="/tmp",
                    env={},
                )

        task.status = "running"
        fragment = swarm.status_bar_fragment()
        # With one running task the fragment is a non-empty string like "af:1/1"
        assert isinstance(fragment, str), f"Expected string fragment, got {fragment!r}"
        assert "1" in fragment, f"Fragment {fragment!r} does not mention the 1 running task"

    def test_empty_swarm_has_no_tasks(self):
        """A freshly reset swarm has no tasks."""
        swarm = SwarmManager()
        assert len(swarm.tasks) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Core loop integration: project → workflow → swarm chain
# ─────────────────────────────────────────────────────────────────────────────

class TestCoreLoopIntegration:
    """Smoke-test the connection between /project, workflow command, and /swarm."""

    def test_workflow_to_swarm_task_id_assignment(self):
        """Task IDs follow the ``af-NNN`` pattern used by the Swarm."""
        swarm = SwarmManager()

        fake_thread = MagicMock(spec=threading.Thread)
        fake_thread.start = MagicMock()

        workflow_kinds = ["prove", "draft", "autoprove", "formalize", "autoformalize"]
        task_ids: list[str] = []

        with patch("swarm_manager._run_claude_code_interactive"):
            with patch("threading.Thread", return_value=fake_thread):
                for kind in workflow_kinds:
                    t = swarm.spawn_interactive(
                        theorem=f"test {kind}",
                        description=f"{kind} task",
                        argv=["/usr/bin/claude"],
                        cwd="/tmp",
                        env={},
                        workflow_kind=kind,
                    )
                    task_ids.append(t.task_id)

        # All IDs are unique, follow the af-NNN pattern, and increment sequentially
        assert len(set(task_ids)) == 5
        assert task_ids == ["af-001", "af-002", "af-003", "af-004", "af-005"]

    def test_project_manifest_path_stored_in_workflow_plan_staged_paths(
        self, monkeypatch, tmp_path: Path
    ):
        """The launched plan's ``staged_paths()`` exposes the project root and lean root."""
        project_root = _lean_project(tmp_path, "IntegrationProject")
        shared = _make_shared_bundle(tmp_path, project_root)
        runtime = _make_runtime(tmp_path, shared)

        monkeypatch.setattr(autoformalize, "_require_executable", lambda n, _m, _e: f"/usr/bin/{n}")
        monkeypatch.setattr(
            autoformalize,
            "_resolve_uv_runner",
            lambda _e: ("/usr/bin/uvx", "--from", autoformalize.LEAN_LSP_MCP_GIT_SPEC, "lean-lsp-mcp"),
        )
        monkeypatch.setattr(autoformalize, "_prepare_shared_bundle", lambda **_kw: shared)
        monkeypatch.setattr(autoformalize, "_resolve_backend_runtime", lambda **_kw: runtime)
        monkeypatch.setattr(
            autoformalize,
            "build_handoff_request",
            lambda **kw: SimpleNamespace(**kw),
        )

        plan = autoformalize.resolve_autoformalize_request(
            "/prove",
            _config(),
            active_cwd=str(project_root),
            base_env={"HOME": str(tmp_path / "home"), "PATH": "/usr/bin"},
        )

        staged = plan.staged_paths()
        assert staged["workflow_kind"] == "prove"
        assert staged["project_root"] == str(project_root)
        assert staged["backend_name"] == "claude-code"

    def test_autoformalize_backend_command_matches_lean4_skills_protocol(
        self, monkeypatch, tmp_path: Path
    ):
        """The backend command sent to lean4-skills uses the ``/lean4:<kind>`` prefix."""
        project_root = _lean_project(tmp_path, "ProtocolProject")
        shared = _make_shared_bundle(tmp_path, project_root)
        runtime = _make_runtime(tmp_path, shared)

        monkeypatch.setattr(autoformalize, "_require_executable", lambda n, _m, _e: f"/usr/bin/{n}")
        monkeypatch.setattr(
            autoformalize,
            "_resolve_uv_runner",
            lambda _e: ("/usr/bin/uvx", "--from", autoformalize.LEAN_LSP_MCP_GIT_SPEC, "lean-lsp-mcp"),
        )
        monkeypatch.setattr(autoformalize, "_prepare_shared_bundle", lambda **_kw: shared)
        monkeypatch.setattr(autoformalize, "_resolve_backend_runtime", lambda **_kw: runtime)
        monkeypatch.setattr(
            autoformalize,
            "build_handoff_request",
            lambda **kw: SimpleNamespace(**kw),
        )

        for frontend, expected_backend_start in [
            ("/prove", "/lean4:prove"),
            ("/draft", "/lean4:draft"),
            ("/autoprove", "/lean4:autoprove"),
            ("/formalize", "/lean4:formalize"),
            ("/autoformalize", "/lean4:autoformalize"),
        ]:
            plan = autoformalize.resolve_autoformalize_request(
                frontend,
                _config(),
                active_cwd=str(project_root),
                base_env={"HOME": str(tmp_path / "home"), "PATH": "/usr/bin"},
            )
            assert plan.backend_command.startswith(expected_backend_start), (
                f"{frontend} → backend_command was {plan.backend_command!r}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Private test helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_shared_bundle(
    tmp_path: Path,
    project_root: Path,
    backend_name: str = "claude-code",
) -> autoformalize.SharedLeanBundle:
    """Build a minimal SharedLeanBundle for testing."""
    from gauss_cli.project import load_gauss_project

    project = load_gauss_project(project_root)
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

    real_home.mkdir(parents=True, exist_ok=True)
    startup_dir.mkdir(parents=True, exist_ok=True)
    mcp_dir.mkdir(parents=True, exist_ok=True)
    scripts_root.mkdir(parents=True, exist_ok=True)
    references_root.mkdir(parents=True, exist_ok=True)
    commands_root.mkdir(parents=True, exist_ok=True)
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
        active_cwd=project_root,
        real_home=real_home,
        plugin_source=plugin_source,
        skill_source=skill_source,
        scripts_root=scripts_root,
        references_root=references_root,
        uv_runner=("/usr/bin/uvx", "--from", autoformalize.LEAN_LSP_MCP_GIT_SPEC, "lean-lsp-mcp"),
    )


def _make_runtime(
    tmp_path: Path,
    shared: autoformalize.SharedLeanBundle,
    backend_name: str = "claude-code",
) -> autoformalize.AutoformalizeBackendRuntime:
    """Build a minimal AutoformalizeBackendRuntime for testing."""
    managed_context = autoformalize.ManagedContext(
        backend_name=backend_name,
        managed_root=shared.managed_root,
        project_root=shared.project.root,
        lean_root=shared.project.lean_root,
        backend_home=shared.managed_root / "backend-home",
        plugin_root=shared.managed_root / "backend-home" / ".claude" / "plugins" / "lean4",
        mcp_config_path=shared.mcp_dir / "lean-lsp.mcp.json",
        startup_context_path=shared.startup_dir / "context.md",
        assets_root=shared.assets_root,
        project_manifest_path=shared.project.manifest_path,
        backend_config_path=shared.managed_root / "backend-home" / ".claude.json",
    )
    return autoformalize.AutoformalizeBackendRuntime(
        argv=["/usr/bin/claude", "--model", autoformalize.CLAUDE_MODEL],
        child_env={"HOME": str(managed_context.backend_home), "PATH": "/usr/bin"},
        managed_context=managed_context,
    )
