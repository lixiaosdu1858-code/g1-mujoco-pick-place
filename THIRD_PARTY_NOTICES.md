# Third-party sources and attribution

Updated: 2026-10-23, Asia/Shanghai. This is the **v0.2.0 source distribution**. Original project code and documentation are MIT-licensed by lixiaosdu1858-code; that license does not relicense the third-party assets described below.

## Robot and gripper assets in this distribution

The 39 STL files in `assets/` are byte-identical copies from the locally frozen [lerobot/unitree-g1-mujoco](https://huggingface.co/lerobot/unitree-g1-mujoco/tree/905111699ef28bdeb1c69c6fe8ada8405e963a48) snapshot, revision `905111699ef28bdeb1c69c6fe8ada8405e963a48`. Per-file source URLs, sizes, and SHA-256 hashes are in [assets/manifest.json](assets/manifest.json). Only meshes referenced by the successful scene are included.

The frozen model repository's README identifies:

- Base model/runtime: `lerobot/unitree-g1-mujoco`, revision `a38dc8617f0fca51b38e9354dc58ee35ad850fb5`.
- Dex1-1 URDF/meshes and wrist camera geometry: [Hxxxz0/HIW-500-controoler](https://github.com/Hxxxz0/HIW-500-controoler/tree/c69d89d88bb51774fe9a3684b90d9ff1abf801da), revision `c69d89d88bb51774fe9a3684b90d9ff1abf801da`.
- Unitree mesh assets retain their upstream terms.

The preserved [HIW-500 NOTICE](licenses/HIW-500-NOTICE.txt) attributes Dex1-1 URDF and mesh assets to Unitree's `unitree_ros` project and explicitly excludes relicensing them under the HIW-500 project's Apache-2.0 license. The [HIW-500 Apache-2.0 text](licenses/HIW-500-Apache-2.0.txt) is retained as provenance evidence, **not as a blanket license for these meshes**.

All 39 mesh files have now been matched by Git blob SHA-1 and byte length to `robots/g1_description/meshes/` in the official [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259) commit `5994d4faef0a9cadd3287f8de0199a67eeb2a259`. The repository's [fixed-revision LICENSE](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/LICENSE) is BSD-3-Clause. Its original text is included in [licenses/unitree_ros-BSD-3-Clause.txt](licenses/unitree_ros-BSD-3-Clause.txt). The complete tree was checked for applicable nearer license files. Per-file official paths, blob identifiers, sizes and SHA-256 hashes are retained in [assets/upstream-license-map.json](assets/upstream-license-map.json). This confirms the mesh attribution; it does not resolve the derived MJCF's license.

## Derived scene

The locally used scene has SHA-256 `9d71754df03ab06c4fbed3010ef9a883ce71a3e6ca04de01815d7d1c7ca2630c`, matching successful run `manual_user_20261021_175535_099`. Neither `assets/scene.xml` nor `configs/scene.xml` is included in this source distribution. They are generated locally from separately supplied upstream XML inputs.

The local project derived this scene from upstream `assets/g1_29dof_with_grippers.xml` (SHA-256 `f177bbfecb2bc9e1cbae9dee875a121e4d19de7a34523bf3ba8db527edc8fa54`) and `assets/scene_33dof.xml` (SHA-256 `f9f18b6ed53688b3bfe8b0c5a2597fc9b7480cc2313b48035c32e7c006b6cb1d`). Local changes fix the robot body and unused joints, retain the active arm/gripper, add the table/tray/free cube, set the task layout and contact geometry, and define actuators and simulation settings. These form a local task model, not an unchanged official Unitree model.

The fixed HF revision does not supply a root license for these XML inputs. They are not bundled or relicensed under MIT or the mesh BSD license. The original local scene-generation program is provided under MIT, but running it does not grant rights to its input or output models. Users must separately obtain the inputs under applicable upstream terms. See [scene preparation](docs/scene-preparation.md).

## Earlier Unitree deployment reference

Earlier learning used [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym/tree/276801e46c5d433564f24658bac64f254b7d2d4b), commit `276801e46c5d433564f24658bac64f254b7d2d4b`. The local copy is a minimal deployment snapshot, not a Git checkout. Its [BSD-3-Clause license](licenses/unitree_rl_gym-BSD-3-Clause.txt) is preserved. That repository's pretrained policy, training stack, and original mesh set are not bundled by this candidate's asset closure. Its license is not substituted for the licensing of the HF-derived assets above.

## Software dependencies

MuJoCo, NumPy and their runtime dependencies are external installed packages, not redistributed Python environments or binaries inside this candidate. Follow the dependency/version instructions accompanying the package. No DDS/SDK bridge, ROS deployment, SONIC policy, learned locomotion policy or third-party training dataset is included by the asset packaging operation.

## Publication status

The project repository and v0.1.0 release are public. The v0.2.0 materials retain MIT project code, BSD-3-Clause meshes and the local scene-generation instructions, and add demonstration recording and dynamic replay tools with one portable example. The relocated candidate completed a user-observed dynamic replay on the existing local environment; this does not establish fresh installation on another machine. Unresolved XML assets and Python environments remain excluded. Public release status is recorded on the repository's [Releases page](https://github.com/lixiaosdu1858-code/g1-mujoco-pick-place/releases), separately from the local preparation and validation evidence.
