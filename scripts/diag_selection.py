"""Which gait does the nearest-gait lookup select under the TRAINING
command distribution? (pendulum9b turn-gate failure diagnosis)"""
import importlib.util, sys, types
import torch

_PKG = "robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking"
_DIR = "source/robot_rl/robot_rl/tasks/manager_based/robot_rl/mdp/commands/traj_tracking"
parts = _PKG.split(".")
for i in range(len(parts)):
    n = ".".join(parts[:i+1])
    if n not in sys.modules:
        m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m
sys.modules[_PKG].__path__ = [_DIR]
def _load(name):
    full = f"{_PKG}.{name}"
    spec = importlib.util.spec_from_file_location(full, f"{_DIR}/{name}.py")
    mod = importlib.util.module_from_spec(spec); sys.modules[full] = mod
    spec.loader.exec_module(mod); return mod
_load("manager_base"); _load("trajectory_manager")
LM = _load("library_manager").LibraryManager

lib = LM(sys.argv[1], None, "cpu")
names = lib.ref_names
# Training spans: lin_vel_x U(0.0, 0.58), lin_vel_y 0, ang_vel_z U(-0.29, 0.29)
g = torch.Generator().manual_seed(0)
vx = torch.rand(200000, generator=g) * 0.58
wz = torch.rand(200000, generator=g) * 0.58 - 0.29
cond = torch.stack([vx, torch.zeros_like(vx), wz], dim=1)
idx = lib.get_traj_indices(cond)
print("selection fractions under training distribution:")
for i, n in enumerate(names):
    frac = float((idx == i).float().mean())
    print(f"  {n:<16} {frac*100:5.1f}%")
# And at the GATE commands specifically:
for c in ([0.56,0,0],[0,0,0.289],[0,0,-0.289]):
    i = int(lib.get_traj_indices(torch.tensor([c]))[0])
    print(f"gate cmd {c} -> {names[i]}")
# Hold logic: cmd_mag for pure-turn commands
print("cmd_mag at (0,0,0.289):", float(torch.tensor([0.,0.,0.289]).norm()),
      "(hold_phi_threshold = 0.1 -> holds if <)")
