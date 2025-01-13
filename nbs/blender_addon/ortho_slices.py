import bpy
import numpy as np

def create_slice_plane(length, width, height, axis='z', position=0.0):
    """
    Creates a 2D plane as a slice of a rectangular box along a specified axis.
    The plane lies within the bounds of the box.

    Args:
        length (float): Length of the box along the X-axis.
        width (float): Width of the box along the Y-axis.
        height (float): Height of the box along the Z-axis.
        axis (str): Axis along which to slice ('x', 'y', or 'z').
        position (float): Position along the chosen axis for the slice plane.
                          Should be within the range of the box dimensions.
    """
    current_active = bpy.context.active_object
    # Validate axis and position
    if axis not in {'x', 'y', 'z'}:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")
    
    axis_limits = {'x': length, 'y': width, 'z': height}
    if not (0.0 <= position <= axis_limits[axis]):
        raise ValueError(f"Position must be within [0, {axis_limits[axis]}] for axis {axis}.")

    # Create the plane's dimensions based on the slicing axis
    if axis == 'x':
        plane_size = (height, width) #(width, height)
        location =  (position, width / 2, height / 2)
        rotation = (0, 1.5708, 0)  # Rotate to align with the YZ-plane
    elif axis == 'y':
        plane_size = (length, height)
        location = (length / 2, position, height / 2)
        rotation = (1.5708, 0, 0)  # Rotate to align with the XZ-plane
    else:  # 'z'
        plane_size = (length, width)
        location = (length / 2, width / 2, position)
        rotation = (0, 0, 0)  # No rotation needed for the XY-plane

    # Add a plane
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = f"SlicePlane_{axis.upper()}_{position:.2f}"

    # Scale and position the plane
    plane.scale = (plane_size[0] / 2, plane_size[1] / 2, 1)
    plane.location = location
    plane.rotation_euler = rotation

    # Apply transformations (scale, location, rotation)
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

    # Restore the previously active object
    if current_active:
        bpy.context.view_layer.objects.active = current_active

    return plane

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


def get_slice_image(image_3d, resolution, axis='z', position=0.0):
    """Get slice of 3d image along axis for ortho-slice visualization.
    image_3d must be a 4d array (channels, x, y, z). Position in microns."""
    if axis == 'x':
        ind = int(np.round(position / resolution[0]))
        slice_img = image_3d[:,ind,:,::-1]
    elif axis == 'y':
        ind = int(np.round(position / resolution[1]))
        slice_img = image_3d[:,:,ind,:].transpose((0,2,1))
    elif axis == 'z': 
        ind = int(np.round(position / resolution[0]))
        slice_img = image_3d[:,:,:,ind].transpose((0,2,1))
    return slice_img


def create_material_from_array(slice_plane, array, material_name="SliceMaterial"):
    """
    Creates a material for a plane using a 2D numpy array as a texture.

    Args:
        slice_plane (bpy.types.Object): The plane object to which the material will be applied.
        array (numpy.ndarray): 2D array representing grayscale values (0-1), or 3D array representing RGBA values (0-1).
        material_name (str): Name of the new material.
    """
    # Validate input array
    if not len(array.shape) in [2,3]:
        raise ValueError("Input array must be 2D.")
    
    # Normalize array to range [0, 1] and convert to a flat list
    image_height, image_width = array.shape[:2]
    pixel_data = np.zeros((image_height, image_width, 4), dtype=np.float32)  # RGBA
    if len(array.shape) == 2:
        pixel_data[..., 0] =  pixel_data[..., 1] = pixel_data[..., 2] = array
        pixel_data[..., 3] = 1.0  # Alpha
    else:
        pixel_data[...] = array
    pixel_data = pixel_data.flatten()

    # Create a new image in Blender
    image = bpy.data.images.new(name="SliceTexture", width=image_width, height=image_height)
    image.pixels = pixel_data.tolist()

    # Create a new material
    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Clear default nodesx
    for node in nodes:
        nodes.remove(node)

    # Add required nodes
    texture_node = nodes.new(type="ShaderNodeTexImage")
    texture_node.image = image
    bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # Arrange nodes
    texture_node.location = (-400, 0)
    bsdf_node.location = (0, 0)
    output_node.location = (400, 0)

    # Connect nodes
    links.new(texture_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

    # Assign the material to the plane
    slice_plane.data.materials.append(material)
    return None

# Example usage: Create a slice of a box with dimensions 4x5x6 along the Z-axis at position 3

length, width, height = (np.array(bpy.context.scene.tissue_cartography_data.shape[1:]) *
                         bpy.context.scene.tissue_cartography_resolution_array)

slice_plane_z = create_slice_plane(length, width, height, axis='z', position=50)
slice_img_z = get_slice_image(bpy.context.scene.tissue_cartography_data,
                              bpy.context.scene.tissue_cartography_resolution_array,
                              axis='z', position=50)
                    
slice_img_z = normalize_quantiles(slice_img_z, quantiles=(0.01, 0.99),
                                  channel_axis=0, clip=True, data_type=None)     
create_material_from_array(slice_plane_z, slice_img_z[0],
                           material_name="SliceMaterial_z_50")  

slice_plane_y = create_slice_plane(length, width, height, axis='y', position=100)
slice_img_y = get_slice_image(bpy.context.scene.tissue_cartography_data,
                              bpy.context.scene.tissue_cartography_resolution_array,
                              axis='y', position=100)
slice_img_y = normalize_quantiles(slice_img_y, quantiles=(0.01, 0.99),
                                  channel_axis=0, clip=True, data_type=None)     
create_material_from_array(slice_plane_y, slice_img_y[0],
                           material_name="SliceMaterial_y_100")

slice_plane_x = create_slice_plane(length, width, height, axis='x', position=100)
slice_img_x = get_slice_image(bpy.context.scene.tissue_cartography_data,
                              bpy.context.scene.tissue_cartography_resolution_array,
                              axis='x', position=100)
slice_img_x = normalize_quantiles(slice_img_x, quantiles=(0.01, 0.99),
                                  channel_axis=0, clip=True, data_type=None)     
create_material_from_array(slice_plane_x, slice_img_x[0],
                           material_name="SliceMaterial_x_100")  



#create_box_from_cube(190, 509, 188)
