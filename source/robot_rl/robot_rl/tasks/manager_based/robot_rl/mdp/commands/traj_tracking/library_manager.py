# Copyright 2026 Zachary Olkin. All rights reserved.

import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .manager_base import ManagerBase
from .trajectory_manager import TrajectoryManager

# TODO: Test with new changes
class LibraryManager(ManagerBase):
    """Manages a library of trajectories, selecting the appropriate one based on a conditioning variable."""

    def __init__(self, library_folder_path: str, hf_repo: str, device,
                 env=None, conditioner_generator_name: str = None):
        self.folder_path = library_folder_path
        self.device = device
        self.env = env
        self.conditioner_generator_name = conditioner_generator_name
        self.trajectory_managers = []
        self.conditioning_vars = None
        self.num_pos_outputs = None
        self.num_vel_outputs = None

        # Per-step cache to avoid redundant conditioner/index/unique computation
        self._cache_valid = False
        self._cache_env_ids = None  # None means full batch
        self._cached_indices: torch.Tensor | None = None
        self._cached_unique_cpu: list[int] | None = None
        self._cached_env_indices: dict[int, torch.Tensor] | None = None

        # Pre-computed contiguous conditioning var for searchsorted
        self._conditioning_vars_sorted: torch.Tensor | None = None

        self.load_library(hf_repo)

        # Full (n, 3) conditioner table for the nearest-gait lookup
        # (tinh-lpa-clfrl.7.7): keying on vel_x alone made any
        # turning gait sharing a vel_x silently unreachable.
        self._conditioning_vars_sorted = self.conditioning_vars.contiguous()

    def invalidate_cache(self):
        """Invalidate the per-step cache. Call once at the start of each step."""
        self._cache_valid = False

    def _ensure_cache(self, env_ids: torch.Tensor | None = None):
        """Ensure the per-step cache is populated for the given env_ids.

        Computes conditioner, trajectory indices, unique indices, and per-trajectory
        env_indices once. Subsequent calls with the same env_ids are no-ops.

        Args:
            env_ids: Optional environment indices. None means full batch.
        """
        # Check if cache is valid for this env_ids configuration
        if self._cache_valid and self._cache_env_ids is env_ids:
            return

        conditioner = self.get_conditioner_var()
        ref_ids = self.get_ref_id_var()
        if env_ids is not None:
            conditioner = conditioner[env_ids]
            ref_ids = ref_ids[env_ids]

        indices = self.get_traj_indices(conditioner, ref_ids)

        # Single CPU-GPU sync: convert unique indices to CPU list
        unique_gpu = torch.unique(indices)
        unique_cpu = unique_gpu.tolist()

        # Pre-compute env_indices for each unique trajectory
        env_indices_map = {}
        for idx in unique_cpu:
            env_indices_map[idx] = torch.where(indices == idx)[0]

        self._cached_indices = indices
        self._cached_unique_cpu = unique_cpu
        self._cached_env_indices = env_indices_map
        self._cache_env_ids = env_ids
        self._cache_valid = True

    def load_library(self, hf_repo: str):
        """Load all trajectory files from the library folder into a list."""
        if hf_repo is None:
            library_path = Path(self.folder_path)
        else:
            library_path = self._get_from_hugging_face(hf_repo, self.folder_path)

        if not library_path.exists():
            raise FileNotFoundError(f"Library folder not found: {self.folder_path}")

        if not library_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.folder_path}")

        # Find all YAML files in the directory
        yaml_files = list(library_path.glob("*.yaml")) + list(library_path.glob("*.yml"))

        if len(yaml_files) == 0:
            raise ValueError(f"No YAML trajectory files found in: {self.folder_path}")

        # Make a trajectory manager object for each file
        traj_conditioner_pairs = []
        for yaml_file in yaml_files:
            traj_manager = TrajectoryManager(str(yaml_file), None, self.device)
            traj_conditioner_pairs.append((traj_manager, traj_manager.traj_data.conditioner))

        # Sort by first conditioning variable to allow searchsorted in get_output
        traj_conditioner_pairs.sort(key=lambda x: x[1][0])

        # Separate back into lists while maintaining sorted order
        self.trajectory_managers = [pair[0] for pair in traj_conditioner_pairs]
        conditioner_list = [pair[1] for pair in traj_conditioner_pairs]

        # Create conditioning tensor of shape (n_traj, 2)
        self.conditioning_vars = torch.tensor(conditioner_list, device=self.device)

        # Multi-skill registry (tinh-lpa-clfrl.8.5b): a library may
        # mix velocity-conditioned periodic gaits (locomotion) with
        # named behavior segments (episodic graph edges, 8.5a
        # exports). Selection: an env with active_ref_id >= 0 gets
        # that trajectory directly; -1 means locomotion -> weighted
        # nearest-gait over the PERIODIC pool only (an episodic
        # segment's zeroed velocity conditioner must never win a
        # velocity command).
        from .trajectory_manager import TrajectoryType
        self.ref_names = [m.traj_data.name for m in self.trajectory_managers]
        if len(set(self.ref_names)) != len(self.ref_names):
            raise ValueError(
                f"Duplicate trajectory names in library: {self.ref_names}")
        self.name_to_idx = {n: i for i, n in enumerate(self.ref_names)}
        self.episodic_mask = torch.tensor(
            [m.traj_data.trajectory_type == TrajectoryType.EPISODIC
             for m in self.trajectory_managers], device=self.device)
        self.periodic_indices = torch.nonzero(
            ~self.episodic_mask).flatten()
        if len(self.periodic_indices) == 0:
            # Behavior-only library: locomotion fallback is traj 0.
            self.periodic_indices = torch.zeros(
                1, dtype=torch.long, device=self.device)
        self.total_times = torch.tensor(
            [m.traj_data.total_time for m in self.trajectory_managers],
            device=self.device)
        # Skill/param tables for the policy observations (8.5d).
        # Slots and channels are DECLARED by the env cfg (stable obs
        # layout across libraries); trajectories carrying an
        # undeclared skill fail loudly at load, missing params
        # default to 0.
        self.skill_names = [m.traj_data.skill
                            for m in self.trajectory_managers]
        self.skill_slot_of_traj = None
        self.params_of_traj = None

    def build_skill_tables(self, skill_slots: list[str],
                           param_channels: list[str]):
        """Map each trajectory to its skill slot + param channels
        (tinh-lpa-clfrl.8.5d). Called by the owning TrajectoryCommand
        when the env cfg declares skill observations."""
        slots = []
        for n, s in zip(self.ref_names, self.skill_names):
            if s not in skill_slots:
                raise ValueError(
                    f"trajectory {n!r} carries skill {s!r}, not in "
                    f"declared skill_slots {skill_slots}")
            slots.append(skill_slots.index(s))
        self.skill_slot_of_traj = torch.tensor(
            slots, dtype=torch.long, device=self.device)
        table = torch.zeros(len(self.trajectory_managers),
                            len(param_channels), device=self.device)
        for i, m in enumerate(self.trajectory_managers):
            for k, v in (m.traj_data.params or {}).items():
                if k in param_channels:
                    table[i, param_channels.index(k)] = float(v)
        self.params_of_traj = table

        # Verify the trajectories are compatible (num_outputs, type, reference_frames)
        ref_manager = self.trajectory_managers[-1]
        num_pos_outputs = ref_manager.traj_data.num_pos_outputs
        num_vel_outputs = ref_manager.traj_data.num_vel_outputs
        pos_output_names = ref_manager.traj_data.pos_output_names
        vel_output_names = ref_manager.traj_data.vel_output_names
        trajectory_type = ref_manager.traj_data.trajectory_type
        ref_frames = ref_manager.traj_data.reference_frames
        for manager in self.trajectory_managers:
            if manager.traj_data.num_pos_outputs != num_pos_outputs:
                raise ValueError(f"Trajectories in the library are not compatible! Varying number of pos outputs!")
            if manager.traj_data.num_vel_outputs != num_vel_outputs:
                raise ValueError(f"Trajectories in the library are not compatible! Varying number of vel outputs!")
            # if manager.traj_data.trajectory_type != trajectory_type:
            #     raise ValueError(f"Trajectories in the library are not compatible! Varying trajectory_type!")
            if manager.traj_data.pos_output_names != pos_output_names:
                raise ValueError(f"Trajectories in the library are not compatible! Varying pos_output_names!"
                                 f"Expected names: {pos_output_names}, \ngot: {manager.traj_data.pos_output_names}")
            if manager.traj_data.vel_output_names != vel_output_names:
                raise ValueError(f"Trajectories in the library are not compatible! Varying vel_output_names!"
                                 f"Expected names: {vel_output_names}, \ngot: {manager.traj_data.vel_output_names}")
            # TODO: Consider putting back!
            # if manager.traj_data.reference_frames != ref_frames:
            #     raise ValueError(f"Trajectories in the library are not compatible! Varying reference_frames!"
            #                      f"Expected frames: {manager.traj_data.reference_frames}, \ngot: {ref_frames}")
        # Mixed library (8.5b): the library-level type is the
        # LOCOMOTION (periodic) type when any periodic gait exists —
        # per-env episodic handling keys on episodic_mask via
        # get_episodic_mask, not this scalar.
        if bool(self.episodic_mask.all()):
            self.trajectory_type = trajectory_type
        else:
            first_periodic = int(self.periodic_indices[0])
            self.trajectory_type = self.trajectory_managers[
                first_periodic].traj_data.trajectory_type
        self.num_pos_outputs = num_pos_outputs
        self.num_vel_outputs = num_vel_outputs
        self.pos_output_names = pos_output_names
        self.vel_output_names = vel_output_names
        self.ref_frames = ref_frames

    def _get_from_hugging_face(self, hf_repo: str, hf_path: str) -> Path:
        """
        Load the trajectory library folder from hugging face. Download it into the hf folder.
        Make the /hf/ folder if it doesn't exist.

        Args:
            hf_repo: hugging face repo to use (e.g., 'username/repo-name')
            hf_path: the path to the trajectory folder in the hf repo (e.g., 'trajectories/library')

        Returns:
            Local Path to the downloaded trajectory folder
        """
        import os

        # Get the robot_rl root directory and go two folders above it
        root = os.getcwd() #os.environ.get("ROBOT_RL_ROOT", os.getcwd())
        hf_base = os.path.join(root)
        hf_base = os.path.abspath(hf_base)  # Resolve to absolute path

        # Create cache directory in the hf folder
        cache_dir = os.path.join(hf_base, "hf")
        os.makedirs(cache_dir, exist_ok=True)

        # The local path to the trajectory folder
        local_folder_path = os.path.join(cache_dir, hf_path)

        # Check if folder already exists locally and has YAML files
        if os.path.exists(local_folder_path) and os.path.isdir(local_folder_path):
            yaml_files = list(Path(local_folder_path).glob("*.yaml")) + list(Path(local_folder_path).glob("*.yml"))
            if len(yaml_files) > 0:
                print(f"Using cached trajectory library from {local_folder_path}")
                return Path(local_folder_path)

        # Download from Hugging Face
        try:
            from huggingface_hub import snapshot_download

            print(f"Downloading trajectory library {hf_path} from {hf_repo}...")

            # Download the entire repo or specific folder
            snapshot_download(
                repo_id=hf_repo,
                allow_patterns=f"{hf_path}/*",  # Download only files in the specified folder
                local_dir=cache_dir,
            )

            print(f"Successfully downloaded trajectory library to {local_folder_path}")
            return Path(local_folder_path)

        except ImportError:
            raise RuntimeError("huggingface_hub is required for downloading trajectories. Install with: pip install huggingface_hub")
        except Exception as e:
            raise RuntimeError(f"Failed to download trajectory library from Hugging Face: {e}")

    def get_conditioner_var(self) -> torch.Tensor:
        """Get the conditioner variable from the environment's command manager.

        Returns:
            torch.Tensor: The conditioning variable for each environment, shape [N].
        """
        cond_term = self.env.command_manager.get_term(self.conditioner_generator_name)
        # (vel_x, vel_y, ang_vel_z) — the full command triple.
        return cond_term.command[:, :3]

    def get_ref_id_var(self) -> torch.Tensor:
        """Per-env explicit reference selection (8.5b): the OWNING
        TrajectoryCommand (which sets self.owner after construction,
        8.5c) exposes active_ref_id (LongTensor [N]; index into
        ref_names, -1 = velocity-conditioned locomotion). Absent ->
        all -1, byte-identical to the pre-multi-skill behavior."""
        owner = getattr(self, "owner", None)
        ref_ids = getattr(owner, "active_ref_id", None)
        if ref_ids is None:
            cond_term = self.env.command_manager.get_term(self.conditioner_generator_name)
            n = cond_term.command.shape[0]
            return torch.full((n,), -1, dtype=torch.long,
                              device=self.device)
        return ref_ids

    @property
    def get_output_names(self):
        """Get position output names (for backwards compatibility)."""
        return self.pos_output_names

    @property
    def get_pos_output_names(self):
        """Get position output names (includes ori_w)."""
        return self.pos_output_names

    @property
    def get_vel_output_names(self):
        """Get velocity output names (excludes ori_w)."""
        return self.vel_output_names

    def get_reference_frames(self):
        return self.ref_frames

    def get_num_outputs(self) -> int:
        """Get the total number of position outputs in the trajectory.

        Returns:
            The number of position outputs (includes ori_w).
        """
        return self.num_pos_outputs

    def get_num_pos_outputs(self) -> int:
        """Get the total number of position outputs in the trajectory.

        Returns:
            The number of position outputs (includes ori_w).
        """
        return self.num_pos_outputs

    def get_num_vel_outputs(self) -> int:
        """Get the total number of velocity outputs in the trajectory.

        Returns:
            The number of velocity outputs (excludes ori_w).
        """
        return self.num_vel_outputs

    def get_num_domains(self):
        """Get the number of domains for each environment."""
        self._ensure_cache()

        N = self._cached_indices.shape[0]
        domains = torch.zeros(N, device=self.device)

        for idx in self._cached_unique_cpu:
            env_indices = self._cached_env_indices[idx]
            manager_domains = self.trajectory_managers[idx].get_num_domains()
            domains[env_indices] = manager_domains

        return domains

    def get_trajectory_type(self):
        return self.trajectory_type

    def get_phasing_var(self, t: torch.Tensor, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Compute the phasing variable for each environment.

        Args:
            t: Time tensor of shape [N].
            env_ids: Optional environment indices of shape [N]. If provided, only compute
                for those environments (conditioner is sliced to match t.shape[0]).

        Returns:
            Phasing variable tensor of shape [N].
        """
        self._ensure_cache(env_ids)

        phasing_var = torch.zeros(t.shape[0], device=self.device)

        for idx in self._cached_unique_cpu:
            env_indices = self._cached_env_indices[idx]
            t_for_manager = t[env_indices]
            manager_phasing_var = self.trajectory_managers[idx].get_phasing_var(t_for_manager)
            phasing_var[env_indices] = manager_phasing_var

        return phasing_var

    def get_output(self, t: torch.Tensor, env_ids: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the outputs to be tracked by the RL.

        Args:
            t: Time in each env, shape [N].
            env_ids: Optional environment indices of shape [N]. If provided, only compute
                for those environments (conditioner is sliced to match t.shape[0]).

        Returns:
            pos_outputs: shape [N, num_pos_outputs] position outputs
            vel_outputs: shape [N, num_vel_outputs] velocity outputs
        """
        self._ensure_cache(env_ids)

        N = t.shape[0]
        pos_outputs = torch.zeros(N, self.num_pos_outputs, device=self.device)
        vel_outputs = torch.zeros(N, self.num_vel_outputs, device=self.device)

        for idx in self._cached_unique_cpu:
            env_indices = self._cached_env_indices[idx]
            t_for_manager = t[env_indices]
            manager_pos_outputs, manager_vel_outputs = self.trajectory_managers[idx].get_output(t_for_manager)
            pos_outputs[env_indices] = manager_pos_outputs
            vel_outputs[env_indices] = manager_vel_outputs

        return pos_outputs, vel_outputs

    def get_acceleration(self, t: torch.Tensor, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Compute the acceleration outputs for each environment.

        Args:
            t: Time in each env, shape [N].
            env_ids: Optional environment indices of shape [N]. If provided, only compute
                for those environments (conditioner is sliced to match t.shape[0]).

        Returns:
            Acceleration outputs, shape [N, num_vel_outputs].
        """
        self._ensure_cache(env_ids)

        N = t.shape[0]
        accelerations = torch.zeros(N, self.num_vel_outputs, device=self.device)

        for idx in self._cached_unique_cpu:
            env_indices = self._cached_env_indices[idx]
            t_for_manager = t[env_indices]
            manager_accelerations = self.trajectory_managers[idx].get_acceleration(t_for_manager)
            accelerations[env_indices] = manager_accelerations

        return accelerations

    def get_ref_frames_in_use(self, t: torch.Tensor,
                              ref_frames: list[str], env_ids: torch.Tensor = None) -> torch.Tensor:
        """Determine the reference frame in use for each environment.

        Args:
            t: Time in each env, shape [N].
            ref_frames: List of reference frame names.
            env_ids: Optional environment indices of shape [N]. If provided, only compute
                for those environments (conditioner is sliced to match t.shape[0]).

        Returns:
            Frame indices into ref_frames for the active frame in each env, shape [N].
        """
        self._ensure_cache(env_ids)

        N = t.shape[0]
        frame_indices = torch.zeros(N, dtype=torch.long, device=self.device)

        for idx in self._cached_unique_cpu:
            env_indices = self._cached_env_indices[idx]
            t_for_manager = t[env_indices]
            manager_frame_indices = self.trajectory_managers[idx].get_ref_frames_in_use(t_for_manager, ref_frames)
            frame_indices[env_indices] = manager_frame_indices

        return frame_indices

    def get_contact_state(self, t: torch.Tensor,
                          contact_frames: list[str], env_ids: torch.Tensor = None) -> torch.Tensor:
        """Get the contact states for each frame from the trajectory.

        Args:
            t: Time in each env, shape [N].
            contact_frames: List of contact frame names.
            env_ids: Optional environment indices of shape [N]. If provided, only compute
                for those environments (conditioner is sliced to match t.shape[0]).

        Returns:
            Contact states for each frame from the trajectory, shape [N, num_contacts].
        """
        self._ensure_cache(env_ids)

        N = t.shape[0]
        contact_states = torch.zeros(N, len(contact_frames), device=self.device)

        for idx in self._cached_unique_cpu:
            env_indices = self._cached_env_indices[idx]
            t_for_manager = t[env_indices]
            manager_states = self.trajectory_managers[idx].get_contact_state(t_for_manager, contact_frames)
            contact_states[env_indices] = manager_states

        return contact_states

    def get_current_domains(self, t: torch.Tensor, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Return the domain index for each env.

        Args:
            t: Time in each env, shape [N].
            env_ids: Optional environment indices of shape [N]. If provided, only compute
                for those environments (conditioner is sliced to match t.shape[0]).

        Returns:
            Domain indices, shape [N].
        """
        self._ensure_cache(env_ids)

        domain_idx = torch.zeros(t.shape[0], dtype=torch.long, device=self.device)

        for idx in self._cached_unique_cpu:
            env_indices = self._cached_env_indices[idx]
            t_for_manager = t[env_indices]
            manager_states = self.trajectory_managers[idx].get_current_domains(t_for_manager)
            domain_idx[env_indices] = manager_states

        return domain_idx


    # L2 weights for the nearest-gait lookup. vel_yaw weighted 2.5x:
    # the standing command (0,0,0) must select the STRAIGHT gait
    # (0.56,0,0) over a turn gait (0,0,0.289) — the phase-freeze then
    # holds a stand, not a mid-pivot pose. 2.5 * 0.289 > 0.56.
    _COND_WEIGHTS = (1.0, 1.0, 2.5)

    def get_traj_indices(self, conditioner: torch.Tensor,
                         ref_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Nearest gait in (vel_x, vel_y, vel_yaw) under a weighted
        L2 over the PERIODIC pool. Replaces the 1D floor-searchsorted
        on vel_x (which could never distinguish gaits sharing a vel_x
        — turning gaits). Envs with ref_ids >= 0 (8.5b) bypass the
        velocity lookup and get that trajectory index directly.
        NOTE: consumers replicating selection semantics (the MuJoCo
        gate's period_for floor-select) must be updated in lockstep
        when a multi-gait library ships."""
        w = torch.tensor(self._COND_WEIGHTS, device=conditioner.device,
                         dtype=conditioner.dtype)
        pool = self._conditioning_vars_sorted[self.periodic_indices]
        # [n_env, 1, 3] - [1, n_pool, 3] -> [n_env, n_pool]
        d = ((conditioner.unsqueeze(1) - pool.unsqueeze(0)) * w
             ).pow(2).sum(dim=-1)
        indices = self.periodic_indices[torch.argmin(d, dim=1)]
        if ref_ids is not None:
            explicit = ref_ids >= 0
            if explicit.any():
                n_traj = len(self.trajectory_managers)
                indices = torch.where(
                    explicit, ref_ids.clamp(0, n_traj - 1), indices)
        return indices

    def get_domain_times(self, t: torch.Tensor, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Get the duration of the current domain for each environment.

        Args:
            t: Time in each env, shape [N].
            env_ids: Optional environment indices of shape [N]. If provided, only compute
                for those environments (conditioner is sliced to match t.shape[0]).

        Returns:
            Domain durations, shape [N].
        """
        self._ensure_cache(env_ids)

        domain_times = torch.zeros(t.shape[0], dtype=torch.float, device=self.device)

        for idx in self._cached_unique_cpu:
            env_indices = self._cached_env_indices[idx]
            t_for_manager = t[env_indices]
            domain_time = self.trajectory_managers[idx].get_domain_times(t_for_manager)
            domain_times[env_indices] = domain_time

        return domain_times

    def get_total_time(self):
        """
        Gets the total time for the trajectory. Assumes all trajectories in the library have the same total time.
        """

        return self.trajectory_managers[-1].get_total_time()

    def get_total_times(self, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Per-env total time of the ACTIVE trajectory (8.5b — a
        mixed library has no single period)."""
        self._ensure_cache(env_ids)
        return self.total_times[self._cached_indices]

    def get_episodic_mask(self, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Per-env flag: the active trajectory is an episodic segment
        (8.5b). The phi=1 hold / handoff logic (8.5c) keys on this,
        not on the library-level trajectory_type."""
        self._ensure_cache(env_ids)
        return self.episodic_mask[self._cached_indices]

    def ref_id_of(self, name: str) -> int:
        """Trajectory index for a named behavior segment (8.5b)."""
        return self.name_to_idx[name]

    def order_outputs(self, pos_output_names: list[str], vel_output_names: list[str]):
        """Order outputs for all trajectory managers in the library.

        Args:
            pos_output_names: Ordered list of position output names (includes ori_w).
            vel_output_names: Ordered list of velocity output names (excludes ori_w).
        """
        for manager in self.trajectory_managers:
            manager.order_outputs(pos_output_names, vel_output_names)

        # Update the library's output name lists
        self.pos_output_names = pos_output_names
        self.vel_output_names = vel_output_names
        self.num_pos_outputs = len(pos_output_names)
        self.num_vel_outputs = len(vel_output_names)

    def log_v_on_phasing_var(self, phi, v):
        """Log the value of the CLF at its value in the phasing variable."""
        self._ensure_cache()

        for idx in self._cached_unique_cpu:
            env_indices = self._cached_env_indices[idx]
            self.trajectory_managers[idx].log_v_on_phasing_var(phi[env_indices], v[env_indices])

    def get_v_log(self) -> tuple[Tensor, Any]:
        """Get the V log for all environments."""
        self._ensure_cache()

        phi_keys = self.trajectory_managers[0].phi_keys
        N = self._cached_indices.shape[0]
        v_log = torch.zeros(N, len(phi_keys), dtype=torch.float, device=self.device)

        for idx in self._cached_unique_cpu:
            env_indices = self._cached_env_indices[idx]
            v_log[env_indices, :] = self.trajectory_managers[idx].v_log

        return v_log, phi_keys

    def get_v_log_avg(self) -> torch.Tensor:
        """
        Compute the average V value for each of the references
        """

        v_mean = torch.zeros(len(self.trajectory_managers), device=self.device)

        for i, manager in enumerate(self.trajectory_managers):
            v = manager.v_log
            v_mean[i] = torch.mean(v)

        return v_mean