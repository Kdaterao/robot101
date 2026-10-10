
from robot101.paths import SO101_DIR
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from trimesh.transformations import quaternion_from_matrix
from robot101.calibration.transforms import rigid_origin


class LilyObjectManager:

    """
        Class which handles storage of 3d objects for mujoco (xml, stl, ect..)

    ====== EXAMPLE STRUCTURE ========
            LilyObjects/
    │       ├── blue_box/
    │       │   ├── model.glb
    │       │   ├── model.obj
    │       │   └── model.xml
    │       ├── cup/
    │       │   ├── model.glb
    │       │   ├── model.obj
    │       │   └── model.xml
    │       └── bottle/
    │           ├── model.glb
    │           ├── model.obj
    │           └── model.xml

    """

    LILY_DIR = SO101_DIR.parent / "lily_objects"
    GLB_NAME = "model.glb"
    OBJ_NAME = "model.obj"
    XML_NAME = "model.xml"
    TAG_NAME = re.compile(r"LilyTag_Surface_(\d+)", re.IGNORECASE)
    EXACT_TAG = re.compile(r"^LilyTag_Surface_(\d+)$", re.IGNORECASE)

    def __init__(self):

        # create lilyobject dir just in case
        self.LILY_DIR.mkdir(parents=True, exist_ok=True)

    def _obj_dir(self, name: str) -> Path:
        return self.LILY_DIR / name

    def _write_model_xml(
        self,
        folder: Path,
        name: str,
        mass: float,
        friction: float,
        tags: dict | None = None,
    ) -> None:
        tags = tags or {}
        site_lines = []
        for tag_id in sorted(tags):
            tag = tags[tag_id]
            pos = tag["pos"]
            quat = tag["quat"]
            pos_s = f"{pos[0]} {pos[1]} {pos[2]}"
            quat_s = f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"
            site_lines.append(
                f'      <site name="lilytag_{tag_id}" pos="{pos_s}" quat="{quat_s}" size="0.005"/>'
            )
        sites_xml = ("\n" + "\n".join(site_lines) + "\n") if site_lines else "\n"
        xml = f'''<mujoco model="{name}">
  <compiler meshdir="." angle="radian"/>
  <asset>
    <mesh name="{name}" file="{self.OBJ_NAME}"/>
  </asset>
  <worldbody>
    <body name="{name}">
      <inertial pos="0 0 0" mass="{mass}"/>
      <geom type="mesh" mesh="{name}" friction="{friction} 0.005 0.0001"/>{sites_xml}    </body>
  </worldbody>
</mujoco>
'''
        (folder / self.XML_NAME).write_text(xml)

    def _read_model_xml(self, folder: Path) -> dict:
        xml_path = folder / self.XML_NAME
        if not xml_path.exists():
            return {}
        root = ET.parse(xml_path).getroot()
        out: dict = {"tags": {}}
        inertial = root.find(".//inertial")
        if inertial is not None and inertial.get("mass") is not None:
            out["mass"] = float(inertial.get("mass"))
        geom = root.find(".//geom")
        if geom is not None and geom.get("friction"):
            out["friction"] = float(geom.get("friction").split()[0])
        for site in root.findall(".//site"):
            sname = site.get("name") or ""
            if not sname.startswith("lilytag_"):
                continue
            tag_id = int(sname.split("_")[-1])
            pos = [float(x) for x in site.get("pos", "0 0 0").split()]
            quat = [float(x) for x in site.get("quat", "1 0 0 0").split()]
            out["tags"][tag_id] = {"pos": pos, "quat": quat}
        return out

    def _import_glb(self, glb_path: Path, folder: Path) -> dict[int, dict]:
        """Copy is already done. Parse LilyTag_Surface_NNN poses vs main-mesh origin."""
        loaded = trimesh.load(str(glb_path), force="scene")
        if isinstance(loaded, trimesh.Trimesh):
            scene = trimesh.Scene(loaded)
        else:
            scene = loaded

        tag_world: dict[int, np.ndarray] = {}
        main_nodes: list[tuple[np.ndarray, str]] = []
        tag_node_names: set[str] = set()

        for node_name in scene.graph.nodes:
            match = self.EXACT_TAG.search(str(node_name))
            if not match:
                continue
            transform, _ = scene.graph.get(node_name)
            tag_world[int(match.group(1))] = np.array(transform, dtype=float)
            tag_node_names.add(str(node_name))

        for node_name in scene.graph.nodes_geometry:
            transform, geom_name = scene.graph.get(node_name)
            if str(node_name) in tag_node_names or self.TAG_NAME.search(str(node_name)):
                continue
            main_nodes.append((np.array(transform, dtype=float), geom_name))

        if not main_nodes:
            raise ValueError(f"No main mesh found in {glb_path}")

        t_origin = rigid_origin(main_nodes[0][0])
        t_origin_inv = np.linalg.inv(t_origin)

        meshes = []
        for transform, geom_name in main_nodes:
            geom = scene.geometry[geom_name]
            if not hasattr(geom, "copy"):
                continue
            piece = geom.copy()
            piece.apply_transform(t_origin_inv @ transform)
            meshes.append(piece)

        if not meshes:
            raise ValueError(f"Could not export a mesh from {glb_path}")

        combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        combined.export(folder / self.OBJ_NAME)

        tags: dict[int, dict] = {}
        for tag_id, t_tag in tag_world.items():
            t_rel = rigid_origin(t_origin_inv @ t_tag)
            quat = quaternion_from_matrix(t_rel)
            tags[tag_id] = {
                "pos": t_rel[:3, 3].tolist(),
                "quat": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
            }

        if not tags:
            print("Warning: no LilyTag_Surface_NNN nodes found in the GLB.")
        else:
            print(f"Found LilyTags: {sorted(tags)}")

        return tags

    def create_obj(self):
        '''
            asks for name (all names must be unique)
            asks for glb path  (will then copy glb to proper location)
            asks for mass
            asks for friction

        '''

        #======= varaibles =============
        edit = False
        defaults: dict = {}

        #======== Get Name =============

        name = input("Enter name of object: ").strip()
        if not name:
            print("Name cannot be empty.")
            return None

        obj_dir = self._obj_dir(name)

        # if exists already
        if obj_dir.is_dir():

            cont = input("This object already exists, do you want to edit it?(y/n) ").strip().lower()

            if cont == "y":
                edit = True
                defaults = self._read_model_xml(obj_dir)

            else:
                return

        #========= Get GLB ================
        copy_glb = True
        glb_src = None
        if edit:
            cont = input("Do you want to overwrite glb?(y/n) ").strip().lower()
            copy_glb = cont == "y"

        if copy_glb:
            while True:
                raw_glb_path = input("Enter Path to GLB: ").strip()
                glb_path = Path(raw_glb_path).expanduser()

                if glb_path.is_file():
                    glb_src = glb_path
                    break
                else:
                    cont = input("Path did not work, do you want to try again?(y/n) ").strip().lower()
                    if cont == "n":
                        break

            if glb_src is None:
                if not edit:
                    print("No GLB provided. Object was not created.")
                    return None
                print("Keeping existing GLB.")
                copy_glb = False

        #========= Get mass + friction ================
        def prompt_float(label: str, default: float | None) -> float:
            hint = f" [{default}]" if default is not None else ""
            while True:
                raw = input(f"Enter {label}{hint}: ").strip()
                if raw == "" and default is not None:
                    return default
                try:
                    return float(raw)
                except ValueError:
                    print("Enter a number.")

        mass = prompt_float("mass", defaults.get("mass"))
        friction = prompt_float("friction", defaults.get("friction"))

        #========= Save ================
        obj_dir.mkdir(parents=True, exist_ok=True)
        tags = defaults.get("tags") or {}

        if copy_glb and glb_src is not None:
            shutil.copy2(glb_src, obj_dir / self.GLB_NAME)
            tags = self._import_glb(obj_dir / self.GLB_NAME, obj_dir)

        if not (obj_dir / self.OBJ_NAME).is_file() and (obj_dir / self.GLB_NAME).is_file():
            tags = self._import_glb(obj_dir / self.GLB_NAME, obj_dir)

        if not (obj_dir / self.OBJ_NAME).is_file():
            print("No mesh on disk. Object was not saved.")
            return None

        self._write_model_xml(obj_dir, name, mass, friction, tags)
        print(f"Saved {obj_dir / self.XML_NAME}")
        return obj_dir

    def list_obj(self):
        names = sorted(
            path.name
            for path in self.LILY_DIR.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        if names:
            print("Lily objects:")
            for name in names:
                print(f"  {name}")
        else:
            print("No Lily objects yet.")
        return names

    def load_obj(self, name):
        folder = self._obj_dir(name)
        mesh_path = folder / self.OBJ_NAME
        xml_path = folder / self.XML_NAME
        glb_path = folder / self.GLB_NAME
        if not folder.is_dir():
            raise FileNotFoundError(f"Object '{name}' not found in {self.LILY_DIR}")
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Missing mesh for '{name}': {mesh_path}")
        if not xml_path.is_file():
            raise FileNotFoundError(f"Missing XML for '{name}': {xml_path}")

        meta = self._read_model_xml(folder)
        return {
            "name": name,
            "folder": folder,
            "glb_path": glb_path if glb_path.is_file() else None,
            "mesh_path": mesh_path,
            "xml_path": xml_path,
            "mass": meta.get("mass"),
            "friction": meta.get("friction"),
            "tags": meta.get("tags") or {},
        }


if __name__ == "__main__":
    mgr = LilyObjectManager()
    print("LilyObjectManager")
    print("  1) create_obj")
    print("  2) list_obj")
    choice = input("Choose: ").strip()
    if choice == "1":
        mgr.create_obj()
    mgr.list_obj()
