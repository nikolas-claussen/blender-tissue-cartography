import bpy
import numpy as np

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


def create_material_from_multilayer_array(mesh, array, material_name="ProjectedMaterial"):
    """
    Creates a material for a mesh using multi-channel, multi-layer projection.

    Args:
        obj (bpy.types.Object): The mesh object to which the material will be applied.
        array (numpy.ndarray): 4D array of shape (channels, layers, U, V)
        material_name (str): Name of the new material.
    """
    # Validate and normalize input array
    if not len(array.shape) == 4:
        raise ValueError("Input array must have 4 axes.")
    array_normalized = normalize_quantiles(array, quantiles=(0.01, 0.99), channel_axis=0,
                                           clip=True, data_type=None)
    # Create a new image in Blender for each layer and channel
    image_height, image_width = array.shape[-2:]
    n_channels, n_layers = array.shape[:2]
    images = {}
    for ic, chanel in enumerate(array_normalized):
        for il, layer in enumerate(chanel):
            pixel_data = np.zeros((image_height, image_width, 4), dtype=np.float32)
            pixel_data[..., 0] =  pixel_data[..., 1] = pixel_data[..., 2] = layer[::-1]
            pixel_data[..., 3] = 1.0  # Alpha
            pixel_data = pixel_data.flatten()
            images[(ic, il)] = bpy.data.images.new(name=f"Channel_{ic}_Layer_{il}",
                                                   width=image_width, height=image_height)
            images[(ic, il)].pixels = pixel_data.tolist()
    # Create a new material
    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    # Clear default nodesx
    for node in nodes:
        nodes.remove(node)
    # Add required nodes
    texture_nodes = {}
    for (ic, il), image in images.items():
        texture_nodes[(ic, il)] = nodes.new(type="ShaderNodeTexImage")
        texture_nodes[(ic, il)].image = image
        texture_nodes[(ic, il)].location = (-400, ic*400 + il*300)
    
    bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # Arrange nodes
    bsdf_node.location = (0, 0)
    output_node.location = (400, 0)

    # Connect nodes
    links.new(texture_nodes[(0,0)].outputs["Color"], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

    # Assign the material to the mesh
    mesh.data.materials.append(material)
    mesh.active_material = material
    return None

obj = bpy.context.active_object
array = np.array(obj["baked_data"][0]).reshape(obj["baked_data"][1])


create_material_from_multilayer_array(obj, array, material_name="ProjectedMaterial")