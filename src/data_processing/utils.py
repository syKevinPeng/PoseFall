from math import e
import torch
import torch.nn.functional as F
import functools


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)


def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(matrix, convention: str = "XYZ"):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).

    Petrovich, M., Black, M. J., &amp; Varol, G. (2021).
    Action-conditioned 3D human motion synthesis with Transformer Vae.
    2021 IEEE/CVF International Conference on Computer Vision (ICCV).
    Retrived from https://doi.org/10.1109/iccv48922.2021.01080
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention: str):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).

    Petrovich, M., Black, M. J., &amp; Varol, G. (2021).
    Action-conditioned 3D human motion synthesis with Transformer Vae.
    2021 IEEE/CVF International Conference on Computer Vision (ICCV).
    Retrived from https://doi.org/10.1109/iccv48922.2021.01080
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)


def parse_output(sequences):
    """
    Parse the output from the model
    @param sequences: Tensor, shape (batch_size, seq_len, feature_dim=153 )
    @return: dict, keys are arm_rot, arm_loc, bone_rot
    """
    batch_size, seq_len, num_joints, feature_dim = sequences.size()
    sequences = sequences.reshape(batch_size, seq_len, -1, 6)
    if num_joints == 25:
        # meanning bone rots + body locs
        arm_loc = sequences[:, :, -1, :3]
        bone_rot = sequences[:, :, :-1, :]
        arm_rot = torch.zeros(batch_size, seq_len, feature_dim)
    elif num_joints == 24:
        # meanning bone rots only
        arm_loc = torch.zeros(batch_size, seq_len, feature_dim)
        bone_rot = sequences
        arm_rot = torch.zeros(batch_size, seq_len, feature_dim)
    elif num_joints == 26:
        # meanning bone rots + body locs + arm rots
        arm_loc = sequences[:, :, -2, :3]
        bone_rot = sequences[:, :, :-2, :]
        arm_rot = sequences[:, :, -1, :]

    return {"arm_rot": arm_rot, "arm_loc": arm_loc, "bone_rot": bone_rot}


def convert_binary_label_to_category(binary_str):
    category = [
        "Impact_loc_Arms",
        "Impact_loc_Head",
        "Impact_loc_Legs",
        "Impact_loc_Torso",
        "Impact_attr_Contraction",
        "Impact_attr_Explosion",
        "Impact_attr_Prick",
        "Impact_attr_Push",
        "Impact_attr_Shot",
        "Glitch_attr_contort",
        "Glitch_attr_flail",
        "Glitch_attr_flash",
        "Glitch_attr_freeze",
        "Glitch_attr_shake",
        "Glitch_attr_short",
        "Glitch_attr_spin",
        "Glitch_attr_stumble",
        "Glitch_attr_stutter",
        "Fall_Attribute_hinge",
        "Fall_Attribute_let go",
        "Fall_Attribute_release",
        "Fall_Attribute_surrender",
        "Fall_Attribute_suspend",
    ]
    assert len(binary_str) == len(category)
    attrs = [category[i] for i, c in enumerate(binary_str) if c == "1"]
    return "-".join(attrs)


if __name__ == "__main__":
    # # test the 6D rotation conversion
    # euler_angles = torch.rand(2, 3) * 2 * 3.1415926 - 3.1415926
    # # euler_angles = torch.tensor([[0.0, 0.0, 0.0]])
    # print(f"generated euler angles: {euler_angles}")
    # rotation_matrix = euler_angles_to_matrix(euler_angles, "XYZ")
    # print(f"generated rotation matrix: \n{rotation_matrix}")
    # rep_6D = matrix_to_rotation_6d(rotation_matrix)
    # print(f"6D representation: {rep_6D}")
    # # convert it back
    # rotation_matrix_2 = rotation_6d_to_matrix(rep_6D)
    # print(f"converted rotation matrix: \n{rotation_matrix_2}")
    # # convert it to euler angles
    # euler_angles_2 = matrix_to_euler_angles(rotation_matrix_2)
    # print(f"converted euler angles: {euler_angles_2}")
    # # check the difference
    # print(f"difference: {euler_angles_2 - euler_angles}")
    result = convert_binary_label_to_category("00100010000001000000100")
    print("-".join(result))
