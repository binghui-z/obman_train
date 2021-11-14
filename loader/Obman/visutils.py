from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def visualize_joints_2d(ax,
                        joints,
                        joint_idxs=True,
                        links=None,
                        alpha=1,
                        scatter=True,
                        linewidth=2):
    if links is None:
        links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    if scatter:
        ax.scatter(x, y, 1, 'r')

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, links, alpha=alpha, linewidth=linewidth)
    ax.axis('equal')


def _draw2djoints(ax, annots, links, alpha=1, linewidth=1):
    colors = ['r', 'm', 'b', 'c', 'g']

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha,
                linewidth=linewidth)


def _draw2dseg(ax, annot, idx1, idx2, c='r', alpha=1, linewidth=1):
    ax.plot([annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
            c=c,
            alpha=alpha,
            linewidth=linewidth)

def visualize_2d(img,
                 hand_joints=None,
                 hand_verts=None,
                 obj_verts=None,
                 links=[(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                        (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.axis('off')
    if hand_joints is not None:
        visualize_joints_2d(
            ax, hand_joints, joint_idxs=False, links=links)
    if obj_verts is not None:
        ax.scatter(obj_verts[:, 0], obj_verts[:, 1], alpha=0.1, c='r')
    if hand_verts is not None:
        ax.scatter(hand_verts[:, 0], hand_verts[:, 1], alpha=0.1, c='b')
    plt.show()


def visualize_3d(img,
                 hand_verts=None,
                 hand_faces=None,
                 obj_verts=None,
                 obj_faces=None):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(img)
    ax.axis('off')
    ax = fig.add_subplot(122, projection='3d')
    add_mesh(ax, hand_verts, hand_faces)
    add_mesh(ax, obj_verts, obj_faces, c='r')
    cam_equal_aspect_3d(ax, hand_verts)
    plt.show()


def add_mesh(ax, verts, faces, alpha=0.1, c='b'):
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    if c == 'b':
        face_color = (141 / 255, 184 / 255, 226 / 255)
    elif c == 'r':
        face_color = (226 / 255, 184 / 255, 141 / 255)
    edge_color = (50 / 255, 50 / 255, 50 / 255)
    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)


def get_coords_2d(coords3d, cam_extr=None, cam_calib=None):
    if cam_extr is None:
        coords2d_hom = np.dot(cam_calib, coords3d.transpose())
    else:
        coords3d_hom = np.concatenate(
            [coords3d, np.ones((coords3d.shape[0], 1))], 1)
        coords3d_hom = coords3d_hom.transpose()
        coords2d_hom = np.dot(cam_calib, np.dot(cam_extr, coords3d_hom))
    coords2d = coords2d_hom / coords2d_hom[2, :]
    coords2d = coords2d[:2, :]
    return coords2d.transpose()