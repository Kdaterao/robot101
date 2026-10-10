
from robot101.paths import SO101_DIR
from pathlib import Path

import mujoco


DEFAULT_SCENE = SO101_DIR / "scene.xml"
RUNTIME_SCENE = SO101_DIR / "_scene_runtime.xml"

# Tabletop poses (REST / RAND) put the gripper at z≈0, which is also the
# floor plane. Raise the robot onto a pedestal so the jaw is not buried.
BASE_MOUNT_HEIGHT = 0.06


def _rgb_axes_xml(length: float, radius: float, indent: str) -> str:
    return "\n".join(
        [
            f'{indent}<geom type="cylinder" fromto="0 0 0 {length} 0 0" size="{radius}"'
            ' rgba="1 0 0 1" contype="0" conaffinity="0"/>',
            f'{indent}<geom type="cylinder" fromto="0 0 0 0 {length} 0" size="{radius}"'
            ' rgba="0 1 0 1" contype="0" conaffinity="0"/>',
            f'{indent}<geom type="cylinder" fromto="0 0 0 0 0 {length}" size="{radius}"'
            ' rgba="0 0 1 1" contype="0" conaffinity="0"/>',
        ]
    )


class SceneLoader:
    """Load a MuJoCo MJCF scene. Includes and meshdir resolve from the file path."""

    def __init__(self, scene_path: str | Path | None = None):
        self.scene_path = Path(scene_path) if scene_path is not None else DEFAULT_SCENE

    def load(self, objects: list[dict] | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
        '''
        Loads the scene XML file and returns the model and data.
        SO101 is always in the default scene. Optional Lily objects
        are injected as mocap bodies.
        '''
        if not self.scene_path.exists():
            raise FileNotFoundError(f"Scene XML not found: {self.scene_path}")

        xml = self.scene_path.read_text()
        load_path = self.scene_path
        if objects:
            xml = self.add_object(xml, objects)
            RUNTIME_SCENE.write_text(xml)
            load_path = RUNTIME_SCENE

        model = mujoco.MjModel.from_xml_path(str(load_path))
        self._mount_base(model)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    @staticmethod
    def _mount_base(model: mujoco.MjModel) -> None:
        '''
        Raises the robot onto a pedestal so the jaw is not buried. (DOES NOT CREEATE THE PEDESTAL)
        '''

        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_id < 0:
            return
        model.body_pos[base_id, 2] = BASE_MOUNT_HEIGHT

    def add_object(self, xml: str, objects: list[dict]) -> str:
        '''
            Adds Lily objects as mocap bodies to a scene XML string.
        '''
        asset_lines = []
        body_lines = []
        for obj in objects:
            name = obj["name"]
            mesh_file = Path(obj["mesh_path"]).resolve().as_posix()
            asset_lines.append(
                f'    <mesh name="{name}" file="{mesh_file}"/>'
            )
            tag_bodies = []
            for tag_id, tag in sorted((obj.get("tags") or {}).items()):
                pos = tag["pos"]
                quat = tag["quat"]
                tag_bodies.append(
                    f'''        <body name="{name}_lilytag_{tag_id}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">
{_rgb_axes_xml(0.025, 0.0012, "          ")}
        </body>'''
                )
            tags_xml = ("\n" + "\n".join(tag_bodies) + "\n") if tag_bodies else "\n"
            if obj.get("collide"):
                geom_contact = 'contype="1" conaffinity="1"'
            else:
                geom_contact = 'contype="0" conaffinity="0"'
            body_lines.append(
                f'''    <body name="{name}" mocap="true">
        <geom name="{name}_mesh" type="mesh" mesh="{name}" {geom_contact} rgba="0.6 0.65 0.8 1"/>
{_rgb_axes_xml(0.03, 0.0015, "        ")}{tags_xml}    </body>'''
            )

        body_lines.append(
            f'''    <body name="floor_origin" mocap="true">
{_rgb_axes_xml(0.08, 0.002, "      ")}
    </body>
    <body name="floor_tag_10" mocap="true">
{_rgb_axes_xml(0.04, 0.0015, "      ")}
    </body>
    <body name="floor_tag_11" mocap="true">
{_rgb_axes_xml(0.04, 0.0015, "      ")}
    </body>'''
        )

        assets = "\n".join(asset_lines)
        bodies = "\n".join(body_lines)
        if "</asset>" not in xml:
            raise ValueError("Scene XML has no </asset> to inject meshes into")
        if "</worldbody>" not in xml:
            raise ValueError("Scene XML has no </worldbody> to inject bodies into")
        xml = xml.replace("</asset>", f"{assets}\n    </asset>", 1)
        xml = xml.replace("</worldbody>", f"{bodies}\n    </worldbody>", 1)
        return xml
