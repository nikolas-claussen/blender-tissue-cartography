import numpy as np
import bpy
from scipy import ndimage, interpolate


def normalize_quantiles(image, quantiles=(0.01, 0.99), channel_axis=None, clip=False,
                        data_type=None):
    """
    Normalize a multi-dimensional image by setting given quantiles to 0 and 1.
    
    Parameters
    ----------
    image : np.array
        Multi-dimensional image.
    quantiles : tuple
        Image quantile to set to 0 and 1.
    channel_axis : int or None
        If None, the image is assumed to have only a single channel.
        If int, indicates the position of the channel axis. 
        Each channel is normalized separately.
    clip : bool
        Whether to clip image to 0-1. Automatically enabled if converting to int dtype.
    data_type : None, np.unit8 or np.uint16
        If not None, image is converted to give data type.
    
    Returns
    -------
    image_normalized : np.array
        Normalized image, the same shape as input
    """
    if channel_axis is None:
        image_normalized = image - np.nanquantile(image, quantiles[0])
        image_normalized /= np.nanquantile(image_normalized, quantiles[1])
        image_normalized = np.nan_to_num(image_normalized)
    else:
        image_normalized = np.moveaxis(image, channel_axis, 0)
        image_normalized = np.stack([ch - np.nanquantile(ch, quantiles[0]) for ch in image_normalized])
        image_normalized = np.stack([ch / np.nanquantile(ch, quantiles[1]) for ch in image_normalized])
        image_normalized = np.moveaxis(np.nan_to_num(image_normalized), 0, channel_axis)
    if clip or (data_type is not None):
        image_normalized = np.clip(image_normalized, 0, 1)
    if data_type is np.uint8:
        image_normalized = np.round((2**8-1)*image_normalized).astype(np.uint8)
    if data_type is np.uint16:
        image_normalized = np.round((2**16-1)*image_normalized).astype(np.uint16)
    return image_normalized


def compute_edge_lengths(obj):
    """
    Computes the lengths of all edges in a mesh object as a numpy array.

    Args:
        obj (bpy.types.Object): The mesh object to compute edge lengths for.

    Returns:
        numpy.ndarray: A 1D array containing the lengths of all edges in the mesh.
    """
    # Ensure the object is a mesh
    if obj.type != 'MESH':
        raise ValueError("The selected object is not a mesh.")
    # Ensure the mesh is in edit mode for accurate vertex data
    bpy.context.view_layer.objects.active = obj
    if obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    edge_lengths = []
    for edge in obj.data.edges:
        v1 = obj.data.vertices[edge.vertices[0]].co
        v2 = obj.data.vertices[edge.vertices[1]].co
        edge_lengths.append((v1 - v2).length)
    return np.array(edge_lengths)


def get_image_to_vertex_interpolator(obj, image_3d, resolution_array, quantiles=(0.01, 0.99)):
    """
    Get interpolator that maps vertex position -> image intensity.
    
    Returns a list of interpolators, one for each channel.
    To avoid aliasing, the 3d image is smoothed with
    sigma=median edge length /2. The image data is also normalized to
    range from 0-1 using the provided quantiles.
    """
    anti_aliasing_scale = np.median(compute_edge_lengths(obj))/2
    image_3d_smoothed = np.stack([ndimage.gaussian_filter(ch, anti_aliasing_scale/resolution_array)
                                  for ch in image_3d])
    image_3d_smoothed = normalize_quantiles(image_3d_smoothed,
                                            quantiles=quantiles, clip=True, data_type=None)
    x, y, z = [np.arange(ni)*resolution_array[i]
               for i, ni in enumerate(image_3d.shape[1:])]
    return [interpolate.RegularGridInterpolator((x,y,z), ch, method='linear', bounds_error=False)
            for ch in image_3d_smoothed]


def assign_vertex_colors(obj, colors):
    """
    Assigns an RGB color to each vertex in the given object.
    Args:
        obj: The mesh object.
        colors: A list or dict of (R, G, B) tuples for each vertex.
    """
    if obj.type != 'MESH':
        print("Object is not a mesh!")
        return
    if not obj.data.vertex_colors:
        obj.data.vertex_colors.new()
    color_layer = obj.data.vertex_colors.active
    # Assign colors to each loop (face corner)
    for loop in obj.data.loops:    
        color_layer.data[loop.index].color = (*colors[loop.vertex_index], 1.0)  # RGBA
    return None


def create_vertex_color_material(object, material_name="VertexColorMaterial"):
    """
    Creates a material for an object that uses vertex colors.
    The R, G, and B channels are processed through separate "Map Range" nodes
    to edit their brightness, and then combined into a Principled BSDF.

    Args:
        object (bpy.types.Object): The object to which the material will be applied.
        material_name (str): Name of the new material.
    """
    # Ensure the object has a vertex color layer
    if not object.data.vertex_colors:
        raise ValueError("The object has no vertex color layers.")

    # Create a new material
    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Add nodes
    vertex_color_node = nodes.new(type="ShaderNodeVertexColor")
    vertex_color_node.layer_name = object.data.vertex_colors[0].name
    vertex_color_node.location = (-1000, 0)

    separate_color_node = nodes.new(type="ShaderNodeSeparateRGB")
    separate_color_node.location = (-800, 0)

    map_range_r = nodes.new(type="ShaderNodeMapRange")
    map_range_r.label = "Map Range R"
    map_range_r.location = (-600, 300)

    map_range_g = nodes.new(type="ShaderNodeMapRange")
    map_range_g.label = "Map Range G"
    map_range_g.location = (-600, 0)

    map_range_b = nodes.new(type="ShaderNodeMapRange")
    map_range_b.label = "Map Range B"
    map_range_b.location = (-600, -300)

    combine_rgb = nodes.new(type="ShaderNodeCombineRGB")
    combine_rgb.location = (-200, 0)

    bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf_node.location = (000, 0)

    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (400, 0)

    # Connect nodes
    links.new(vertex_color_node.outputs["Color"], separate_color_node.inputs["Image"])
    links.new(separate_color_node.outputs["R"], map_range_r.inputs["Value"])
    links.new(separate_color_node.outputs["G"], map_range_g.inputs["Value"])
    links.new(separate_color_node.outputs["B"], map_range_b.inputs["Value"])

    links.new(map_range_r.outputs["Result"], combine_rgb.inputs["R"])
    links.new(map_range_g.outputs["Result"], combine_rgb.inputs["G"])
    links.new(map_range_b.outputs["Result"], combine_rgb.inputs["B"])

    links.new(combine_rgb.outputs["Image"], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

    # Set default map range values for each channel
    for map_range_node in [map_range_r, map_range_g, map_range_b]:
        map_range_node.inputs["From Min"].default_value = 0.0
        map_range_node.inputs["From Max"].default_value = 1.0
        map_range_node.inputs["To Min"].default_value = 0.0
        map_range_node.inputs["To Max"].default_value = 1.0

    # Assign the material to the object
    if object.data.materials:
        object.data.materials[0] = material
    else:
        object.data.materials.append(material)

    return None


# Example usage
obj = bpy.context.active_object

interpolators = get_image_to_vertex_interpolator(obj, bpy.context.scene.tissue_cartography_data,
                    bpy.context.scene.tissue_cartography_resolution_array)


# compute the intensities at the vertices
normal_offset = 3
intensities = {v.index: [f(v.co + normal_offset*v.normal) for f in interpolators]
                         for v in obj.data.vertices}

# channel 0 + gray to RGB
colors = {key: 3*(val[0], ) for key, val in intensities.items()}
#colors = [(0.0, 1.0, 0.0),] * len(obj.data.vertices)

assign_vertex_colors(obj, colors)
create_vertex_color_material(obj)