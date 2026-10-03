
'''


NOTE:
AI GENERATED BLENDER SCRIPT 
TO MAKE TAGGING MODELS IN 
BLENDER WITH LILYTAGS MUCH EASIER


NOTE: 
If using a model generated from a LiDAR scan, 
denoise/smooth the scanned mesh before using this tool. 
Scanned meshes can contain vertex noise and irregular 
triangulation that may interfere with surface and edge detection.
 A CAD/mesh-processing tool such as AutoCAD can be used to clean
 and denoise the mesh before importing it.

'''



import bpy
import math

from mathutils import Vector, Quaternion, Matrix

import gpu
from gpu_extras.batch import batch_for_shader


# ============================================================
# GLOBAL STATE
# ============================================================

EDGE_1 = None
EDGE_2 = None
SURFACE = None

DRAW_HANDLER = None


# ============================================================
# SETTINGS
# ============================================================

class LilyTagSettings(bpy.types.PropertyGroup):

    # --------------------------------------------------------
    # EDGE 1
    # --------------------------------------------------------

    edge1_endpoint: bpy.props.EnumProperty(
        name="Edge 1 Endpoint",
        items=[
            (
                'A',
                "Endpoint A",
                "Measure from endpoint A"
            ),
            (
                'B',
                "Endpoint B",
                "Measure from endpoint B"
            ),
        ],
        default='A',
        update=lambda self, context: tag_redraw()
    )

    edge1_distance: bpy.props.FloatProperty(
        name="Edge 1 Distance",
        description="Distance from selected endpoint",
        default=0.01,
        min=0.0,
        unit='LENGTH',
        update=lambda self, context: tag_redraw()
    )

    # --------------------------------------------------------
    # EDGE 2
    # --------------------------------------------------------

    edge2_endpoint: bpy.props.EnumProperty(
        name="Edge 2 Endpoint",
        items=[
            (
                'A',
                "Endpoint A",
                "Measure from endpoint A"
            ),
            (
                'B',
                "Endpoint B",
                "Measure from endpoint B"
            ),
        ],
        default='A',
        update=lambda self, context: tag_redraw()
    )

    edge2_distance: bpy.props.FloatProperty(
        name="Edge 2 Distance",
        description="Distance from selected endpoint",
        default=0.01,
        min=0.0,
        unit='LENGTH',
        update=lambda self, context: tag_redraw()
    )

    # --------------------------------------------------------
    # LILLYTAG
    # --------------------------------------------------------

    lilytag_id: bpy.props.IntProperty(
        name="LilyTag ID",
        default=1,
        min=0
    )

    lilytag_size: bpy.props.FloatProperty(
        name="LilyTag Size",
        default=0.01,
        min=0.0001,
        unit='LENGTH'
    )

    lilytag_rotation: bpy.props.FloatProperty(
        name="Rotation",
        description=(
            "Spin the LilyTag around the face normal. "
            "Red arrow is +X, green arrow is +Y."
        ),
        default=0.0,
        min=0.0,
        max=math.tau,
        subtype='ANGLE',
        unit='ROTATION',
        update=lambda self, context: tag_redraw()
    )


# ============================================================
# REDRAW
# ============================================================

def tag_redraw():

    if not bpy.context.screen:
        return

    for area in bpy.context.screen.areas:

        if area.type == 'VIEW_3D':

            area.tag_redraw()


# ============================================================
# EDGE POINT
# ============================================================

def get_edge_point(
    edge_data,
    endpoint,
    distance
):

    a = Vector(edge_data["a"])
    b = Vector(edge_data["b"])

    direction = b - a

    if direction.length < 1e-8:

        return a.copy()

    length = direction.length

    direction.normalize()

    distance = max(
        0.0,
        min(distance, length)
    )

    if endpoint == 'A':

        return (
            a +
            direction * distance
        )

    return (
        b -
        direction * distance
    )


# ============================================================
# EDGE DATA
# ============================================================

def make_edge_data(
    obj,
    edge
):

    mesh = obj.data

    v0_index = edge.vertices[0]
    v1_index = edge.vertices[1]

    v0 = mesh.vertices[v0_index]
    v1 = mesh.vertices[v1_index]

    mw = obj.matrix_world

    a = mw @ v0.co
    b = mw @ v1.co

    return {
        "obj": obj,
        "edge_index": edge.index,
        "a": a.copy(),
        "b": b.copy(),
        "length": (b - a).length
    }


# ============================================================
# SURFACE DATA
# ============================================================

def make_surface_data(
    obj,
    face_or_index
):

    mesh = obj.data

    if isinstance(
        face_or_index,
        int
    ):

        face = mesh.polygons[
            face_or_index
        ]

    else:

        face = face_or_index

    world_verts = []

    for vertex_index in face.vertices:

        vertex = mesh.vertices[
            vertex_index
        ]

        world_verts.append(
            obj.matrix_world @ vertex.co
        )

    if len(world_verts) == 0:

        return None

    center = (
        sum(
            world_verts,
            Vector()
        )
        /
        len(world_verts)
    )

    normal = face_normal_from_world_verts(
        world_verts
    )

    if normal is None:

        normal = (
            obj.matrix_world.to_3x3()
            @
            face.normal
        ).normalized()

    return {
        "obj": obj,
        "face_index": face.index,
        "verts": world_verts,
        "center": center,
        "normal": normal,
        "edge_keys": list(
            face.edge_keys
        )
    }


# ============================================================
# PROJECT POINT ONTO PLANE
# ============================================================

def project_point_to_plane(
    point,
    plane_point,
    plane_normal
):

    distance = (
        point -
        plane_point
    ).dot(
        plane_normal
    )

    return (
        point -
        plane_normal * distance
    )


# ============================================================
# PERPENDICULAR DIRECTION IN FACE PLANE
# ============================================================

def get_face_perpendicular(
    edge_direction,
    face_normal
):

    edge_direction = Vector(
        edge_direction
    ).normalized()

    face_normal = Vector(
        face_normal
    ).normalized()

    edge_in_face = (
        edge_direction
        -
        face_normal *
        edge_direction.dot(
            face_normal
        )
    )

    if edge_in_face.length < 1e-8:

        return None

    edge_in_face.normalize()

    perpendicular = (
        face_normal.cross(
            edge_in_face
        )
    )

    if perpendicular.length < 1e-8:

        return None

    perpendicular.normalize()

    return perpendicular


# ============================================================
# GET PERPENDICULAR MARKER
# ============================================================

def get_perpendicular_marker(
    edge_data,
    endpoint,
    distance,
    surface,
    marker_length=0.06
):

    if edge_data is None:
        return None

    if surface is None:
        return None

    point = get_edge_point(
        edge_data,
        endpoint,
        distance
    )

    point = project_point_to_plane(
        point,
        surface["center"],
        surface["normal"]
    )

    a = Vector(
        edge_data["a"]
    )

    b = Vector(
        edge_data["b"]
    )

    edge_direction = (
        b - a
    ).normalized()

    perpendicular = (
        get_face_perpendicular(
            edge_direction,
            surface["normal"]
        )
    )

    if perpendicular is None:

        return None

    half = (
        marker_length *
        0.5
    )

    p1 = (
        point -
        perpendicular * half
    )

    p2 = (
        point +
        perpendicular * half
    )

    return p1, p2


# ============================================================
# DRAW LINE
# ============================================================

def draw_line(
    points,
    color,
    width=8,
    depth_test=False
):

    if points is None:
        return

    if len(points) < 2:
        return

    shader = gpu.shader.from_builtin(
        'UNIFORM_COLOR'
    )

    batch = batch_for_shader(
        shader,
        'LINES',
        {
            "pos": points
        }
    )

    if not depth_test:

        gpu.state.depth_test_set(
            'NONE'
        )

    else:

        gpu.state.depth_test_set(
            'LESS_EQUAL'
        )

    gpu.state.blend_set(
        'ALPHA'
    )

    shader.bind()

    shader.uniform_float(
        "color",
        color
    )

    gpu.state.line_width_set(
        width
    )

    batch.draw(shader)

    gpu.state.line_width_set(
        1.0
    )

    gpu.state.blend_set(
        'NONE'
    )

    gpu.state.depth_test_set(
        'LESS_EQUAL'
    )


# ============================================================
# TRIANGULATE POLYGON
# ============================================================

def triangulate_polygon(
    vertices
):

    if len(vertices) < 3:

        return []

    triangles = []

    for i in range(
        1,
        len(vertices) - 1
    ):

        triangles.append(
            (
                0,
                i,
                i + 1
            )
        )

    return triangles


# ============================================================
# DRAW FACE
# ============================================================

def draw_face(
    vertices,
    color
):

    if len(vertices) < 3:

        return

    indices = triangulate_polygon(
        vertices
    )

    shader = gpu.shader.from_builtin(
        'UNIFORM_COLOR'
    )

    batch = batch_for_shader(
        shader,
        'TRIS',
        {
            "pos": vertices
        },
        indices=indices
    )

    gpu.state.blend_set(
        'ALPHA'
    )

    gpu.state.depth_test_set(
        'NONE'
    )

    shader.bind()

    shader.uniform_float(
        "color",
        color
    )

    batch.draw(shader)

    gpu.state.blend_set(
        'NONE'
    )

    gpu.state.depth_test_set(
        'LESS_EQUAL'
    )


# ============================================================
# DRAW HIGHLIGHTS
# ============================================================

def draw_highlights():

    global EDGE_1
    global EDGE_2
    global SURFACE

    settings = getattr(
        bpy.context.scene,
        "lilytag_settings",
        None
    )

    # ========================================================
    # TARGET FACE
    # ========================================================

    if SURFACE is not None:

        verts = SURFACE[
            "verts"
        ]

        draw_face(
            verts,
            (
                0.1,
                1.0,
                0.1,
                0.12
            )
        )

        outline = []

        for i in range(
            len(verts)
        ):

            outline.append(
                verts[i]
            )

            outline.append(
                verts[
                    (i + 1) %
                    len(verts)
                ]
            )

        draw_line(
            outline,
            (
                0.1,
                1.0,
                0.1,
                1.0
            ),
            width=6,
            depth_test=False
        )

    # ========================================================
    # EDGE 1
    # ========================================================

    if EDGE_1 is not None:

        draw_line(
            [
                Vector(
                    EDGE_1["a"]
                ),
                Vector(
                    EDGE_1["b"]
                )
            ],
            (
                1.0,
                0.08,
                0.02,
                1.0
            ),
            width=12,
            depth_test=False
        )

        if (
            SURFACE is not None
            and
            settings is not None
        ):

            marker = (
                get_perpendicular_marker(
                    EDGE_1,
                    settings.edge1_endpoint,
                    settings.edge1_distance,
                    SURFACE
                )
            )

            if marker is not None:

                draw_line(
                    marker,
                    (
                        1.0,
                        1.0,
                        0.0,
                        1.0
                    ),
                    width=10,
                    depth_test=False
                )

    # ========================================================
    # EDGE 2
    # ========================================================

    if EDGE_2 is not None:

        draw_line(
            [
                Vector(
                    EDGE_2["a"]
                ),
                Vector(
                    EDGE_2["b"]
                )
            ],
            (
                0.02,
                0.25,
                1.0,
                1.0
            ),
            width=12,
            depth_test=False
        )

        if (
            SURFACE is not None
            and
            settings is not None
        ):

            marker = (
                get_perpendicular_marker(
                    EDGE_2,
                    settings.edge2_endpoint,
                    settings.edge2_distance,
                    SURFACE
                )
            )

            if marker is not None:

                draw_line(
                    marker,
                    (
                        1.0,
                        1.0,
                        0.0,
                        1.0
                    ),
                    width=10,
                    depth_test=False
                )


# ============================================================
# RAYCAST
# ============================================================

def raycast_from_mouse(
    context,
    event
):

    from bpy_extras.view3d_utils import (
        region_2d_to_origin_3d,
        region_2d_to_vector_3d
    )

    region = context.region

    region_3d = (
        context.space_data.region_3d
    )

    coord = (
        event.mouse_region_x,
        event.mouse_region_y
    )

    origin = (
        region_2d_to_origin_3d(
            region,
            region_3d,
            coord
        )
    )

    direction = (
        region_2d_to_vector_3d(
            region,
            region_3d,
            coord
        )
    )

    depsgraph = (
        context.evaluated_depsgraph_get()
    )

    (
        hit,
        location,
        normal,
        face_index,
        obj,
        matrix
    ) = context.scene.ray_cast(
        depsgraph,
        origin,
        direction
    )

    if not hit:

        return None

    return {
        "location": location,
        "normal": normal,
        "face_index": face_index,
        "object": obj
    }


# ============================================================
# FIND CLOSEST EDGE ON SELECTED FACE
# ============================================================

def closest_edge_on_selected_face(
    obj,
    face_index,
    location
):

    mesh = obj.data

    if (
        face_index < 0
        or
        face_index >= len(
            mesh.polygons
        )
    ):

        return None

    face = mesh.polygons[
        face_index
    ]

    mw = obj.matrix_world

    best_edge = None

    best_distance = (
        float("inf")
    )

    for edge_key in face.edge_keys:

        v1_index = edge_key[0]
        v2_index = edge_key[1]

        a = (
            mw @
            mesh.vertices[
                v1_index
            ].co
        )

        b = (
            mw @
            mesh.vertices[
                v2_index
            ].co
        )

        ab = b - a

        if (
            ab.length_squared
            <
            1e-12
        ):

            continue

        t = (
            (location - a).dot(ab)
            /
            ab.length_squared
        )

        t = max(
            0.0,
            min(1.0, t)
        )

        closest = (
            a +
            ab * t
        )

        distance = (
            location -
            closest
        ).length

        if distance < best_distance:

            best_distance = distance

            actual_edge_index = None

            for edge in mesh.edges:

                if {
                    edge.vertices[0],
                    edge.vertices[1]
                } == {
                    v1_index,
                    v2_index
                }:

                    actual_edge_index = (
                        edge.index
                    )

                    break

            if actual_edge_index is None:

                continue

            best_edge = {
                "obj": obj,
                "edge_index": (
                    actual_edge_index
                ),
                "a": a.copy(),
                "b": b.copy(),
                "length": ab.length
            }

    return best_edge


# ============================================================
# PICK SURFACE
# ============================================================

class MESH_OT_lilytag_pick_surface(
    bpy.types.Operator
):

    bl_idname = (
        "mesh.lilytag_pick_surface"
    )

    bl_label = (
        "Pick Target Surface"
    )

    bl_description = (
        "Select the face that will define "
        "the LilyTag plane"
    )

    def invoke(
        self,
        context,
        event
    ):

        context.window_manager.modal_handler_add(
            self
        )

        self.report(
            {'INFO'},
            "Click the target face first"
        )

        return {'RUNNING_MODAL'}

    def modal(
        self,
        context,
        event
    ):

        global SURFACE
        global EDGE_1
        global EDGE_2

        if event.type == 'ESC':

            return {'CANCELLED'}

        if (
            event.type == 'LEFTMOUSE'
            and
            event.value == 'PRESS'
        ):

            hit = (
                raycast_from_mouse(
                    context,
                    event
                )
            )

            if hit is None:

                return {'RUNNING_MODAL'}

            obj = hit[
                "object"
            ]

            if obj.type != 'MESH':

                return {'RUNNING_MODAL'}

            face_index = hit[
                "face_index"
            ]

            if face_index < 0:

                return {'RUNNING_MODAL'}

            face = (
                obj.data.polygons[
                    face_index
                ]
            )

            SURFACE = (
                make_surface_data(
                    obj,
                    face
                )
            )

            EDGE_1 = None
            EDGE_2 = None

            self.report(
                {'INFO'},
                "Target surface selected. "
                "Now pick edges from this face."
            )

            tag_redraw()

            return {'FINISHED'}

        return {'RUNNING_MODAL'}


# ============================================================
# PICK EDGE
# ============================================================

class MESH_OT_lilytag_pick_edge(
    bpy.types.Operator
):

    bl_idname = (
        "mesh.lilytag_pick_edge"
    )

    bl_label = (
        "Pick Reference Edge"
    )

    bl_description = (
        "Pick an edge belonging to "
        "the selected face"
    )

    edge_number: bpy.props.IntProperty(
        default=1
    )

    def invoke(
        self,
        context,
        event
    ):

        if SURFACE is None:

            self.report(
                {'ERROR'},
                "Select the target surface first"
            )

            return {'CANCELLED'}

        context.window_manager.modal_handler_add(
            self
        )

        self.report(
            {'INFO'},
            f"Click an edge on the selected face "
            f"for Edge {self.edge_number}"
        )

        return {'RUNNING_MODAL'}

    def modal(
        self,
        context,
        event
    ):

        global EDGE_1
        global EDGE_2
        global SURFACE

        if event.type == 'ESC':

            return {'CANCELLED'}

        if (
            event.type == 'LEFTMOUSE'
            and
            event.value == 'PRESS'
        ):

            if SURFACE is None:

                self.report(
                    {'ERROR'},
                    "Select a target surface first"
                )

                return {'CANCELLED'}

            hit = (
                raycast_from_mouse(
                    context,
                    event
                )
            )

            if hit is None:

                return {'RUNNING_MODAL'}

            obj = hit[
                "object"
            ]

            if obj != SURFACE["obj"]:

                self.report(
                    {'WARNING'},
                    "Edge must belong to the "
                    "selected surface"
                )

                return {'RUNNING_MODAL'}

            selected_face_index = (
                SURFACE[
                    "face_index"
                ]
            )

            edge = (
                closest_edge_on_selected_face(
                    obj,
                    selected_face_index,
                    hit["location"]
                )
            )

            if edge is None:

                self.report(
                    {'WARNING'},
                    "Could not find an eligible edge"
                )

                return {'RUNNING_MODAL'}

            if self.edge_number == 1:

                EDGE_1 = edge

                self.report(
                    {'INFO'},
                    "Edge 1 selected"
                )

            else:

                EDGE_2 = edge

                self.report(
                    {'INFO'},
                    "Edge 2 selected"
                )

            tag_redraw()

            return {'FINISHED'}

        return {'RUNNING_MODAL'}


# ============================================================
# LINE INTERSECTION
# ============================================================

def line_line_intersection_3d(
    p1,
    d1,
    p2,
    d2
):

    w0 = p1 - p2

    a = d1.dot(d1)

    b = d1.dot(d2)

    c = d2.dot(d2)

    d = d1.dot(w0)

    e = d2.dot(w0)

    denominator = (
        a * c -
        b * b
    )

    if abs(
        denominator
    ) < 1e-8:

        return None

    s = (
        b * e -
        c * d
    ) / denominator

    t = (
        a * e -
        b * d
    ) / denominator

    point1 = (
        p1 +
        d1 * s
    )

    point2 = (
        p2 +
        d2 * t
    )

    return (
        point1 +
        point2
    ) * 0.5


# ============================================================
# FACE NORMAL FROM WORLD VERTS
# ============================================================

def face_normal_from_world_verts(
    world_verts
):

    if len(world_verts) < 3:

        return None

    normal = Vector()

    count = len(world_verts)

    for i in range(count):

        a = world_verts[i]
        b = world_verts[(i + 1) % count]

        normal.x += (
            (a.y - b.y) *
            (a.z + b.z)
        )

        normal.y += (
            (a.z - b.z) *
            (a.x + b.x)
        )

        normal.z += (
            (a.x - b.x) *
            (a.y + b.y)
        )

    if normal.length < 1e-12:

        edge_a = (
            world_verts[1] -
            world_verts[0]
        )

        edge_b = (
            world_verts[2] -
            world_verts[0]
        )

        normal = edge_a.cross(
            edge_b
        )

    if normal.length < 1e-12:

        return None

    return normal.normalized()


# ============================================================
# TAG BASIS ALIGNED TO FACE
# ============================================================

def lilytag_axes_from_face(
    world_verts,
    face_normal
):

    # Tag +Z points into the face, not along the outward normal.
    z_axis = -(
        Vector(
            face_normal
        ).normalized()
    )

    x_axis = None
    best_length = 0.0

    count = len(world_verts)

    for i in range(count):

        edge = (
            world_verts[(i + 1) % count]
            -
            world_verts[i]
        )

        edge = (
            edge
            -
            z_axis *
            edge.dot(z_axis)
        )

        if edge.length > best_length:

            best_length = edge.length
            x_axis = edge

    if x_axis is None or best_length < 1e-12:

        helper = Vector((0.0, 0.0, 1.0))

        if abs(z_axis.dot(helper)) > 0.9:

            helper = Vector((1.0, 0.0, 0.0))

        x_axis = helper.cross(
            z_axis
        )

    x_axis.normalize()

    y_axis = z_axis.cross(
        x_axis
    )

    y_axis.normalize()

    x_axis = y_axis.cross(
        z_axis
    )

    x_axis.normalize()

    return x_axis, y_axis, z_axis


def lilytag_world_rotation(
    world_verts,
    face_normal,
    extra_spin=0.0
):

    x_axis, y_axis, z_axis = lilytag_axes_from_face(
        world_verts,
        face_normal
    )

    rotation = Matrix((
        (x_axis.x, y_axis.x, z_axis.x),
        (x_axis.y, y_axis.y, z_axis.y),
        (x_axis.z, y_axis.z, z_axis.z),
    )).to_quaternion()

    if extra_spin:

        rotation = (
            Quaternion(
                Vector(
                    face_normal
                ).normalized(),
                extra_spin
            )
            @
            rotation
        )

    return rotation


def get_or_create_lilytag_material(
    name,
    color
):

    mat = bpy.data.materials.get(
        name
    )

    if mat is None:

        mat = bpy.data.materials.new(
            name
        )

        mat.use_nodes = True

        principled = mat.node_tree.nodes.get(
            "Principled BSDF"
        )

        if principled is not None:

            principled.inputs[
                "Base Color"
            ].default_value = (
                color[0],
                color[1],
                color[2],
                1.0
            )

        mat.diffuse_color = (
            color[0],
            color[1],
            color[2],
            1.0
        )

    mat.use_backface_culling = False

    return mat


# ============================================================
# CREATE LILLYTAG
# ============================================================

def create_lilytag(
    position,
    normal,
    settings,
    face_verts=None
):

    size = (
        settings.lilytag_size
    )

    half = (
        size * 0.5
    )

    lift = max(
        size * 0.04,
        0.0004
    )

    # Arrows sit on the outward side of the plane so they stay
    # visible after the tag is flipped to face into the surface.
    arrow_z = -lift
    arrow_base = half * 0.08
    arrow_width = half * 0.22
    arrow_tip = half * 0.92

    verts = [
        (-half, -half, 0.0),
        ( half, -half, 0.0),
        ( half,  half, 0.0),
        (-half,  half, 0.0),
        ( arrow_tip,   0.0,         arrow_z),
        ( arrow_base, -arrow_width, arrow_z),
        ( arrow_base,  arrow_width, arrow_z),
        ( 0.0,         arrow_tip,   arrow_z),
        (-arrow_width, arrow_base,  arrow_z),
        ( arrow_width, arrow_base,  arrow_z),
    ]

    faces = [
        (0, 1, 2, 3),
        (4, 6, 5),
        (7, 9, 8),
    ]

    mesh = bpy.data.meshes.new(
        f"LilyTag_{settings.lilytag_id}_Mesh"
    )

    mesh.from_pydata(
        verts,
        [],
        faces
    )

    body_mat = get_or_create_lilytag_material(
        "LilyTagBody",
        (0.12, 0.12, 0.12)
    )

    x_arrow_mat = get_or_create_lilytag_material(
        "LilyTagX",
        (0.95, 0.08, 0.08)
    )

    y_arrow_mat = get_or_create_lilytag_material(
        "LilyTagY",
        (0.08, 0.85, 0.12)
    )

    mesh.materials.append(
        body_mat
    )

    mesh.materials.append(
        x_arrow_mat
    )

    mesh.materials.append(
        y_arrow_mat
    )

    if len(mesh.polygons) >= 1:

        mesh.polygons[0].material_index = 0

    if len(mesh.polygons) >= 2:

        mesh.polygons[1].material_index = 1

    if len(mesh.polygons) >= 3:

        mesh.polygons[2].material_index = 2

    mesh.update()

    obj = bpy.data.objects.new(
        f"LilyTag_Surface_"
        f"{settings.lilytag_id:03d}",
        mesh
    )

    bpy.context.collection.objects.link(
        obj
    )

    obj.location = position

    normal = (
        normal.normalized()
    )

    if face_verts and len(face_verts) >= 3:

        rotation = lilytag_world_rotation(
            face_verts,
            normal,
            settings.lilytag_rotation
        )

    else:

        rotation = lilytag_world_rotation(
            [
                position,
                position + Vector((1.0, 0.0, 0.0)),
                position + Vector((0.0, 1.0, 0.0)),
            ],
            normal,
            settings.lilytag_rotation
        )

    obj.rotation_mode = (
        'QUATERNION'
    )

    obj.rotation_quaternion = (
        rotation
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    parent_obj = SURFACE[
        "obj"
    ]

    obj["lilytag_id"] = (
        settings.lilytag_id
    )

    obj["lilytag_type"] = (
        "LilyTag"
    )

    obj["lilytag_parent"] = (
        parent_obj.name
    )

    obj["lilytag_size"] = (
        settings.lilytag_size
    )

    obj["surface_face_index"] = (
        SURFACE[
            "face_index"
        ]
    )

    obj["world_position"] = (
        list(position)
    )

    obj["world_normal"] = (
        list(normal)
    )

    forward = (
        rotation
        @
        Vector((1.0, 0.0, 0.0))
    )

    obj["world_forward"] = (
        list(forward)
    )

    obj["rotation"] = (
        settings.lilytag_rotation
    )

    # --------------------------------------------------------
    # Edge 1 metadata
    # --------------------------------------------------------

    obj["edge_1_index"] = (
        EDGE_1[
            "edge_index"
        ]
    )

    obj["edge_1_endpoint"] = (
        settings.edge1_endpoint
    )

    obj["edge_1_distance"] = (
        settings.edge1_distance
    )

    # --------------------------------------------------------
    # Edge 2 metadata
    # --------------------------------------------------------

    obj["edge_2_index"] = (
        EDGE_2[
            "edge_index"
        ]
    )

    obj["edge_2_endpoint"] = (
        settings.edge2_endpoint
    )

    obj["edge_2_distance"] = (
        settings.edge2_distance
    )

    # --------------------------------------------------------
    # Parent
    # --------------------------------------------------------

    obj.parent = (
        parent_obj
    )

    obj.matrix_parent_inverse = (
        parent_obj.matrix_world.inverted()
    )

    return obj


# ============================================================
# SOLVE
# ============================================================

class MESH_OT_lilytag_solve_constraints(
    bpy.types.Operator
):

    bl_idname = (
        "mesh.lilytag_solve_constraints"
    )

    bl_label = (
        "Calculate & Place LilyTag"
    )

    bl_description = (
        "Calculate the intersection of "
        "the two perpendiculars"
    )

    def execute(
        self,
        context
    ):

        global EDGE_1
        global EDGE_2
        global SURFACE

        settings = (
            context.scene.lilytag_settings
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if SURFACE is None:

            self.report(
                {'ERROR'},
                "Select a target surface first"
            )

            return {'CANCELLED'}

        if EDGE_1 is None:

            self.report(
                {'ERROR'},
                "Select Edge 1"
            )

            return {'CANCELLED'}

        if EDGE_2 is None:

            self.report(
                {'ERROR'},
                "Select Edge 2"
            )

            return {'CANCELLED'}

        # ----------------------------------------------------
        # DUPLICATE ID GUARD
        #
        # Prevent creation of two LilyTags
        # with the same ID.
        # ----------------------------------------------------

        for obj in bpy.data.objects:

            if (
                obj.get("lilytag_type") == "LilyTag"
                and
                obj.get("lilytag_id") == settings.lilytag_id
            ):

                self.report(
                    {'ERROR'},
                    f"LilyTag ID "
                    f"{settings.lilytag_id} "
                    f"already exists"
                )

                return {'CANCELLED'}

        # ----------------------------------------------------
        # Distance points
        # ----------------------------------------------------

        p1 = get_edge_point(
            EDGE_1,
            settings.edge1_endpoint,
            settings.edge1_distance
        )

        p2 = get_edge_point(
            EDGE_2,
            settings.edge2_endpoint,
            settings.edge2_distance
        )

        # ----------------------------------------------------
        # Surface
        # ----------------------------------------------------

        surface_center = (
            SURFACE["center"]
        )

        surface_normal = (
            SURFACE["normal"]
            .normalized()
        )

        # ----------------------------------------------------
        # Project points onto face
        # ----------------------------------------------------

        p1 = project_point_to_plane(
            p1,
            surface_center,
            surface_normal
        )

        p2 = project_point_to_plane(
            p2,
            surface_center,
            surface_normal
        )

        # ----------------------------------------------------
        # Edge directions
        # ----------------------------------------------------

        d1 = (
            Vector(
                EDGE_1["b"]
            )
            -
            Vector(
                EDGE_1["a"]
            )
        ).normalized()

        d2 = (
            Vector(
                EDGE_2["b"]
            )
            -
            Vector(
                EDGE_2["a"]
            )
        ).normalized()

        # ----------------------------------------------------
        # Perpendiculars inside face
        # ----------------------------------------------------

        perp1 = (
            get_face_perpendicular(
                d1,
                surface_normal
            )
        )

        perp2 = (
            get_face_perpendicular(
                d2,
                surface_normal
            )
        )

        if perp1 is None:

            self.report(
                {'ERROR'},
                "Edge 1 cannot produce a "
                "face-plane perpendicular"
            )

            return {'CANCELLED'}

        if perp2 is None:

            self.report(
                {'ERROR'},
                "Edge 2 cannot produce a "
                "face-plane perpendicular"
            )

            return {'CANCELLED'}

        # ----------------------------------------------------
        # Intersection
        # ----------------------------------------------------

        intersection = (
            line_line_intersection_3d(
                p1,
                perp1,
                p2,
                perp2
            )
        )

        if intersection is None:

            self.report(
                {'ERROR'},
                "The two perpendiculars are parallel"
            )

            return {'CANCELLED'}

        # ----------------------------------------------------
        # Force final point onto face
        # ----------------------------------------------------

        intersection = (
            project_point_to_plane(
                intersection,
                surface_center,
                surface_normal
            )
        )

        # ----------------------------------------------------
        # Create LilyTag
        # ----------------------------------------------------

        tag = create_lilytag(
            intersection,
            surface_normal,
            settings,
            SURFACE["verts"]
        )

        # ----------------------------------------------------
        # Select new tag
        # ----------------------------------------------------

        bpy.ops.object.select_all(
            action='DESELECT'
        )

        tag.select_set(
            True
        )

        context.view_layer.objects.active = (
            tag
        )

        # ----------------------------------------------------
        # Clear construction highlights
        # ----------------------------------------------------

        clear_constraints()

        self.report(
            {'INFO'},
            f"LilyTag "
            f"{settings.lilytag_id} "
            f"placed"
        )

        return {'FINISHED'}


# ============================================================
# CLEAR CONSTRAINTS
# ============================================================

def clear_constraints():

    global EDGE_1
    global EDGE_2
    global SURFACE

    EDGE_1 = None
    EDGE_2 = None
    SURFACE = None

    tag_redraw()


class MESH_OT_lilytag_clear_constraints(
    bpy.types.Operator
):

    bl_idname = (
        "mesh.lilytag_clear_constraints"
    )

    bl_label = (
        "Clear Constraints"
    )

    def execute(
        self,
        context
    ):

        clear_constraints()

        self.report(
            {'INFO'},
            "Constraints cleared"
        )

        return {'FINISHED'}


# ============================================================
# PANEL
# ============================================================

class VIEW3D_PT_lilytag_panel(
    bpy.types.Panel
):

    bl_label = (
        "LilyTag Surfaces"
    )

    bl_idname = (
        "VIEW3D_PT_lilytag_surfaces"
    )

    bl_space_type = (
        'VIEW_3D'
    )

    bl_region_type = (
        'UI'
    )

    bl_category = (
        'LilyTag Surfaces'
    )

    def draw(
        self,
        context
    ):

        layout = self.layout

        settings = (
            context.scene.lilytag_settings
        )

        # ====================================================
        # STEP 1
        # ====================================================

        box = layout.box()

        box.label(
            text="1. Target Surface",
            icon='OBJECT_DATA'
        )

        if SURFACE is None:

            box.label(
                text="Not selected",
                icon='INFO'
            )

        else:

            box.label(
                text="Surface selected",
                icon='CHECKMARK'
            )

            box.label(
                text=(
                    f"Face: "
                    f"{SURFACE['face_index']}"
                )
            )

        box.operator(
            "mesh.lilytag_pick_surface",
            text="Select Face",
            icon='ADD'
        )

        # ====================================================
        # STEP 2
        # ====================================================

        box = layout.box()

        box.label(
            text="2. Reference Edge 1",
            icon='MESH_DATA'
        )

        if SURFACE is None:

            box.label(
                text="Select surface first",
                icon='INFO'
            )

        elif EDGE_1 is None:

            box.label(
                text="Not selected",
                icon='INFO'
            )

        else:

            box.label(
                text="Selected",
                icon='CHECKMARK'
            )

            box.label(
                text=(
                    f"Length: "
                    f"{EDGE_1['length']:.4f} m"
                )
            )

        op = box.operator(
            "mesh.lilytag_pick_edge",
            text="Select Edge 1",
            icon='ADD'
        )

        op.edge_number = 1

        box.prop(
            settings,
            "edge1_endpoint"
        )

        box.prop(
            settings,
            "edge1_distance"
        )

        # ====================================================
        # STEP 3
        # ====================================================

        box = layout.box()

        box.label(
            text="3. Reference Edge 2",
            icon='MESH_DATA'
        )

        if SURFACE is None:

            box.label(
                text="Select surface first",
                icon='INFO'
            )

        elif EDGE_2 is None:

            box.label(
                text="Not selected",
                icon='INFO'
            )

        else:

            box.label(
                text="Selected",
                icon='CHECKMARK'
            )

            box.label(
                text=(
                    f"Length: "
                    f"{EDGE_2['length']:.4f} m"
                )
            )

        op = box.operator(
            "mesh.lilytag_pick_edge",
            text="Select Edge 2",
            icon='ADD'
        )

        op.edge_number = 2

        box.prop(
            settings,
            "edge2_endpoint"
        )

        box.prop(
            settings,
            "edge2_distance"
        )

        # ====================================================
        # LILLYTAG SETTINGS
        # ====================================================

        box = layout.box()

        box.label(
            text="LilyTag Settings",
            icon='CONSTRAINT'
        )

        box.prop(
            settings,
            "lilytag_id"
        )

        box.prop(
            settings,
            "lilytag_size"
        )

        box.prop(
            settings,
            "lilytag_rotation"
        )

        box.label(
            text="Red = +X, green = +Y",
            icon='EMPTY_SINGLE_ARROW'
        )

        box.label(
            text="Tag +Z faces into the surface"
        )

        # ====================================================
        # ACTIONS
        # ====================================================

        layout.separator()

        layout.operator(
            "mesh.lilytag_solve_constraints",
            text="Calculate & Place",
            icon='ADD'
        )

        layout.operator(
            "mesh.lilytag_clear_constraints",
            text="Clear Constraints",
            icon='TRASH'
        )


# ============================================================
# REGISTER
# ============================================================

classes = (

    LilyTagSettings,

    MESH_OT_lilytag_pick_surface,

    MESH_OT_lilytag_pick_edge,

    MESH_OT_lilytag_solve_constraints,

    MESH_OT_lilytag_clear_constraints,

    VIEW3D_PT_lilytag_panel,
)


def register():

    global DRAW_HANDLER

    for cls in classes:

        bpy.utils.register_class(
            cls
        )

    bpy.types.Scene.lilytag_settings = (
        bpy.props.PointerProperty(
            type=LilyTagSettings
        )
    )

    DRAW_HANDLER = (
        bpy.types.SpaceView3D.draw_handler_add(
            draw_highlights,
            (),
            'WINDOW',
            'POST_VIEW'
        )
    )

    tag_redraw()


def unregister():

    global DRAW_HANDLER

    if DRAW_HANDLER is not None:

        bpy.types.SpaceView3D.draw_handler_remove(
            DRAW_HANDLER,
            'WINDOW'
        )

        DRAW_HANDLER = None

    if hasattr(
        bpy.types.Scene,
        "lilytag_settings"
    ):

        del bpy.types.Scene.lilytag_settings

    for cls in reversed(classes):

        bpy.utils.unregister_class(
            cls
        )


if __name__ == "__main__":

    register()